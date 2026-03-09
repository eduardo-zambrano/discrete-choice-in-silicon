#!/usr/bin/env python3
"""
Attention Diagnostics for "Discrete Choice in Silicon"

Forward-pass diagnostics on GPT-2 small (117M parameters).
Computes:
  1. Inclusive value trajectories across layers
  2. IIA ratio tests (context perturbation)
  3. Effective temperature estimation per head
  4. Attention concentration (HHI) across layers
  5. Head aggregation and diversification
  6. Inclusive value vs. logit lens comparison

All computations are forward-pass only — no training or fine-tuning.

Usage:
    python attention_diagnostics.py --all         # Run all diagnostics
    python attention_diagnostics.py --inclusive    # Inclusive value only
    python attention_diagnostics.py --iia          # IIA test only
    python attention_diagnostics.py --temperature  # Temperature estimation only
    python attention_diagnostics.py --hhi          # HHI only
    python attention_diagnostics.py --head-agg     # Head aggregation only
    python attention_diagnostics.py --iv-lens      # IV vs logit lens only

Output: Figures saved to ../output/figures/
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIGURES_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
MODEL_NAME = "gpt2"  # GPT-2 small, 117M parameters
DEVICE = "cpu"

# Corpus for diagnostics
SENTENCES = [
    "The Federal Reserve raised interest rates to combat inflation.",
    "Machine learning models require large amounts of training data.",
    "The cat sat on the mat and watched the birds in the garden.",
    "Central banks use monetary policy to stabilize the economy.",
    "Attention mechanisms allow neural networks to focus on relevant inputs.",
    "The stock market reacted sharply to the unexpected earnings report.",
    "Transformers have revolutionized natural language processing tasks.",
    "Fiscal policy and monetary policy are the two main tools of macroeconomics.",
]

# IIA test sentences: base sentence + extensions
IIA_BASE = "The central bank raised rates"
IIA_EXTENSIONS = [
    "",  # baseline
    " yesterday",
    " yesterday after the meeting",
    " yesterday after the meeting amid global uncertainty",
    " yesterday after the meeting amid global uncertainty and market volatility",
]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model():
    """Load GPT-2 small and tokenizer."""
    print(f"Loading {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2Model.from_pretrained(MODEL_NAME, output_attentions=True)
    model.eval()
    model.to(DEVICE)
    print(f"Loaded. {sum(p.numel() for p in model.parameters()):,} parameters.")
    return tokenizer, model


def get_attention_and_logits(model, tokenizer, text):
    """
    Forward pass: return attention weights and pre-softmax logits for all layers/heads.

    Returns
    -------
    attentions : list of (n_heads, seq_len, seq_len) arrays, one per layer
    logits_raw : list of (n_heads, seq_len, seq_len) arrays (pre-softmax scores)
    tokens : list of token strings
    """
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    tokens = [tokenizer.decode(t) for t in inputs["input_ids"][0]]

    # Single forward pass: hooks capture pre-softmax logits during the same pass
    # that produces post-softmax attention weights
    logits_raw, hooks = _register_logit_hooks(model)

    with torch.no_grad():
        outputs = model(**inputs)

    for h in hooks:
        h.remove()

    # outputs.attentions is a tuple of (batch, heads, seq, seq) tensors
    attentions = [a[0].cpu().numpy() for a in outputs.attentions]

    # Validate: softmax of extracted logits should match HuggingFace attention weights
    _validate_logits(attentions, logits_raw)

    return attentions, logits_raw, tokens


_validation_done = False  # only validate once to avoid repeated output


def _validate_logits(attentions, logits_raw):
    """Check that softmax(logits_raw) ≈ attentions from HuggingFace."""
    global _validation_done
    if _validation_done:
        return
    _validation_done = True

    max_err = 0.0
    for l in range(len(attentions)):
        logits_t = torch.tensor(logits_raw[l])
        # Apply causal mask
        seq_len = logits_t.shape[-1]
        causal = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        logits_t = logits_t.masked_fill(causal, float("-inf"))
        recon = torch.softmax(logits_t, dim=-1).numpy()
        err = np.max(np.abs(recon - attentions[l]))
        max_err = max(max_err, err)

    if max_err > 1e-4:
        print(f"  WARNING: logit extraction mismatch (max error = {max_err:.6f})")
    else:
        print(f"  Logit extraction validated (max error = {max_err:.2e})")


def _register_logit_hooks(model):
    """
    Register forward hooks to capture pre-softmax attention logits (QK^T / sqrt(d))
    during the model's own forward pass.

    Returns (logits_list, hooks) — logits_list is populated during forward pass.
    """
    logits_all = []
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            # args[0] is already layer-normed (block applies ln_1 before attn)
            hidden_ln = args[0]

            # QKV projection
            qkv = module.c_attn(hidden_ln)
            q, k, v = qkv.split(module.split_size, dim=-1)

            batch, seq_len, _ = q.shape
            n_heads = module.num_heads
            d_head = q.shape[-1] // n_heads

            q = q.view(batch, seq_len, n_heads, d_head).transpose(1, 2)
            k = k.view(batch, seq_len, n_heads, d_head).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)

            # Apply causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=scores.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            logits_all.append(scores[0].detach().cpu().numpy())
        return hook_fn

    for i, block in enumerate(model.h):
        h = block.attn.register_forward_hook(make_hook(i), with_kwargs=True)
        hooks.append(h)

    return logits_all, hooks


# ---------------------------------------------------------------------------
# Diagnostic 1: Inclusive Value Trajectories
# ---------------------------------------------------------------------------


def compute_inclusive_values(logits_raw):
    """
    Compute inclusive value log Z_i = log sum_j exp(u_ij / tau) for each token i.

    For standard attention, tau = 1 (scaling already applied in logits).

    Parameters
    ----------
    logits_raw : list of (n_heads, seq_len, seq_len) arrays

    Returns
    -------
    inclusive_values : (n_layers, n_heads, seq_len) array
    """
    n_layers = len(logits_raw)
    n_heads, seq_len, _ = logits_raw[0].shape
    iv = np.zeros((n_layers, n_heads, seq_len))

    for l, logits in enumerate(logits_raw):
        for h in range(n_heads):
            for i in range(seq_len):
                # Only consider non-masked positions (finite logits)
                scores = logits[h, i, :]
                finite_mask = np.isfinite(scores)
                if finite_mask.any():
                    s = scores[finite_mask]
                    # Log-sum-exp for numerical stability
                    max_s = np.max(s)
                    iv[l, h, i] = max_s + np.log(np.sum(np.exp(s - max_s)))

    return iv


def diagnostic_inclusive_value(model, tokenizer):
    """
    Plot inclusive value trajectories across layers.

    For each sentence, compute the mean inclusive value per layer (averaged
    over heads and tokens), then plot the trajectory.
    """
    print("\n=== Diagnostic 1: Inclusive Value Trajectories ===")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Mean inclusive value per layer, one line per sentence
    ax = axes[0]
    for sent in SENTENCES[:4]:
        _, logits_raw, tokens = get_attention_and_logits(model, tokenizer, sent)
        iv = compute_inclusive_values(logits_raw)
        # Average over heads and tokens
        mean_iv = iv.mean(axis=(1, 2))
        label = sent[:40] + "..." if len(sent) > 40 else sent
        ax.plot(range(len(mean_iv)), mean_iv, marker="o", markersize=3, label=label)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Inclusive Value (log $Z_i$)")
    ax.set_title("(a) Inclusive Value Across Layers")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # Panel B: Per-head inclusive value at selected layers for one sentence
    ax = axes[1]
    _, logits_raw, tokens = get_attention_and_logits(model, tokenizer, SENTENCES[0])
    iv = compute_inclusive_values(logits_raw)
    n_layers = iv.shape[0]
    layers_to_show = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    for l in layers_to_show:
        mean_iv_per_head = iv[l].mean(axis=1)  # average over tokens
        ax.plot(
            range(len(mean_iv_per_head)),
            mean_iv_per_head,
            marker="s",
            markersize=4,
            label=f"Layer {l}",
        )

    ax.set_xlabel("Head")
    ax.set_ylabel("Mean Inclusive Value (log $Z_i$)")
    ax.set_title("(b) Inclusive Value by Head")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "inclusive_value.pdf", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'inclusive_value.pdf'}")


# ---------------------------------------------------------------------------
# Diagnostic 2: IIA Test
# ---------------------------------------------------------------------------


def diagnostic_iia(model, tokenizer):
    """
    Test Independence of Irrelevant Alternatives.

    Key insight: IIA is a property of softmax, so it holds *exactly* within any
    single head at any single layer (by construction). The interesting test is
    whether IIA holds for the *effective* attention from input to a given layer,
    which involves composition across layers.

    We test this by using the *last* token in the sequence as the query (so it
    can see all tokens), and tracking its attention ratio to two fixed early
    tokens (j, k) as we extend the context. Within a single layer, IIA predicts
    the ratio is invariant to added tokens. Across composed layers, the keys
    and values are transformed by prior layers, so adding context changes the
    representations that token i attends over — breaking IIA.
    """
    print("\n=== Diagnostic 2: IIA Test ===")

    n_heads = 12  # GPT-2 small

    # j and k: two early token positions in the base sentence
    j_idx = 0
    k_idx = 1

    # Compute ratios across extensions
    # i_idx = last token in each extension (varies with context length)
    all_ratios = []  # shape will be (n_extensions, n_layers, n_heads)

    for ext in IIA_EXTENSIONS:
        text = IIA_BASE + ext
        attentions, _, tokens = get_attention_and_logits(model, tokenizer, text)

        i_idx = len(tokens) - 1  # last token sees entire context

        ratios = np.zeros((len(attentions), n_heads))
        for l, attn in enumerate(attentions):
            for h in range(n_heads):
                a_ij = attn[h, i_idx, j_idx]
                a_ik = attn[h, i_idx, k_idx]
                if a_ik > 1e-10:
                    ratios[l, h] = a_ij / a_ik
                else:
                    ratios[l, h] = np.nan
        all_ratios.append(ratios)

    all_ratios = np.array(all_ratios)  # (n_ext, n_layers, n_heads)

    # Compute IIA violation: std of log-ratio across extensions, per layer
    # Under IIA, log(a_ij/a_ik) should be constant across extensions
    log_ratios = np.log(np.where(all_ratios > 0, all_ratios, np.nan))
    iia_violation = np.nanstd(log_ratios, axis=0)  # (n_layers, n_heads)
    mean_violation_by_layer = np.nanmean(iia_violation, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Mean IIA violation by layer
    ax = axes[0]
    ax.bar(range(len(mean_violation_by_layer)), mean_violation_by_layer, color="steelblue", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("IIA Violation (Std of log ratio)")
    ax.set_title("(a) IIA Violation by Layer")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: Heatmap of IIA violation by layer and head
    ax = axes[1]
    im = ax.imshow(iia_violation.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Head")
    ax.set_title("(b) IIA Violation by Layer and Head")
    plt.colorbar(im, ax=ax, label="Std of log ratio")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "iia_test.pdf", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'iia_test.pdf'}")


# ---------------------------------------------------------------------------
# Diagnostic 3: Effective Temperature Estimation
# ---------------------------------------------------------------------------


def diagnostic_temperature(model, tokenizer):
    """
    Measure effective attention sharpness for each head.

    Two complementary measures:
    1. Entropy of the attention distribution H(a_i) = -sum_j a_{ij} log a_{ij}.
       Lower entropy = sharper attention = lower effective temperature.
    2. Entropy ratio: H(a_i) / log(n_i), where n_i is the number of available
       positions. This normalizes for sequence position (later tokens have more
       alternatives). Ratio of 1 = uniform; ratio near 0 = concentrated.

    The entropy ratio maps to effective temperature: in a logit model with equal
    utilities, the entropy is (tau-1)/tau * log(n) + log(n)/n * ... ≈ log(n)
    when tau is large. More precisely, for a softmax with scores spread sigma,
    the entropy reveals the effective temperature relative to the score spread.
    """
    print("\n=== Diagnostic 3: Effective Temperature (Attention Entropy) ===")

    all_entropy = []
    all_entropy_ratio = []

    for sent in SENTENCES:
        attentions, _, tokens = get_attention_and_logits(model, tokenizer, sent)
        n_layers = len(attentions)
        n_heads = attentions[0].shape[0]
        seq_len = attentions[0].shape[1]

        entropy = np.zeros((n_layers, n_heads))
        entropy_ratio = np.zeros((n_layers, n_heads))

        for l in range(n_layers):
            for h in range(n_heads):
                entropies = []
                ratios = []
                for i in range(seq_len):
                    a = attentions[l][h, i, : i + 1]
                    a = a[a > 1e-15]  # filter zeros for log
                    n_i = len(a)
                    if n_i < 2:
                        continue
                    H = -np.sum(a * np.log(a))
                    H_max = np.log(n_i)
                    entropies.append(H)
                    ratios.append(H / H_max)

                if entropies:
                    entropy[l, h] = np.mean(entropies)
                    entropy_ratio[l, h] = np.mean(ratios)

        all_entropy.append(entropy)
        all_entropy_ratio.append(entropy_ratio)

    mean_entropy = np.mean(np.array(all_entropy), axis=0)  # (n_layers, n_heads)
    mean_ratio = np.mean(np.array(all_entropy_ratio), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Heatmap of entropy ratio (normalized sharpness)
    ax = axes[0]
    im = ax.imshow(mean_ratio.T, aspect="auto", cmap="coolwarm_r", interpolation="nearest",
                   vmin=0, vmax=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Head")
    ax.set_title("(a) Attention Entropy Ratio $H/H_{\\max}$ by Head")
    plt.colorbar(im, ax=ax, label="$H / H_{\\max}$  (0 = sharp, 1 = uniform)")

    # Panel B: Mean entropy ratio by layer
    ax = axes[1]
    layer_means = mean_ratio.mean(axis=1)
    layer_stds = mean_ratio.std(axis=1)
    layers = np.arange(len(layer_means))
    ax.errorbar(layers, layer_means, yerr=layer_stds, marker="o", capsize=3, color="steelblue")
    ax.axhline(y=1.0, color="grey", linestyle=":", alpha=0.5, label="Uniform attention")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Entropy Ratio $H / H_{\\max}$")
    ax.set_title("(b) Mean Entropy Ratio by Layer ($\\pm$ 1 SD)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "temperature.pdf", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'temperature.pdf'}")


# ---------------------------------------------------------------------------
# Diagnostic 4: Attention Concentration (HHI)
# ---------------------------------------------------------------------------


def compute_hhi(weights):
    """Herfindahl-Hirschman Index: sum of squared shares."""
    return np.sum(weights ** 2)


def diagnostic_hhi(model, tokenizer):
    """
    Compute attention concentration (HHI) per row and per column.

    Row HHI: How concentrated is each token's attention spending?
    Column HHI: How concentrated is demand for each token as a source?
    """
    print("\n=== Diagnostic 4: Attention Concentration (HHI) ===")

    all_row_hhi = []
    all_col_hhi = []

    for sent in SENTENCES:
        attentions, _, tokens = get_attention_and_logits(model, tokenizer, sent)
        n_layers = len(attentions)
        n_heads = attentions[0].shape[0]
        seq_len = attentions[0].shape[1]

        row_hhi = np.zeros((n_layers, n_heads))
        col_hhi = np.zeros((n_layers, n_heads))

        for l in range(n_layers):
            for h in range(n_heads):
                attn = attentions[l][h]  # (seq_len, seq_len)
                # Row HHI: average across query tokens
                row_hhis = [compute_hhi(attn[i, : i + 1]) for i in range(seq_len)]
                row_hhi[l, h] = np.mean(row_hhis)

                # Column HHI: normalize columns, compute HHI
                # Column sum gives total attention received by each key token
                col_sums = attn.sum(axis=0)
                col_shares = col_sums / col_sums.sum() if col_sums.sum() > 0 else col_sums
                col_hhi[l, h] = compute_hhi(col_shares)

        all_row_hhi.append(row_hhi)
        all_col_hhi.append(col_hhi)

    mean_row_hhi = np.mean(np.array(all_row_hhi), axis=0)  # (n_layers, n_heads)
    mean_col_hhi = np.mean(np.array(all_col_hhi), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Row HHI by layer (mean and spread across heads)
    ax = axes[0]
    row_layer_mean = mean_row_hhi.mean(axis=1)
    row_layer_std = mean_row_hhi.std(axis=1)
    col_layer_mean = mean_col_hhi.mean(axis=1)
    col_layer_std = mean_col_hhi.std(axis=1)

    layers = np.arange(len(row_layer_mean))
    ax.errorbar(
        layers, row_layer_mean, yerr=row_layer_std,
        marker="o", capsize=3, label="Row HHI (attention spending)", color="steelblue"
    )
    ax.errorbar(
        layers, col_layer_mean, yerr=col_layer_std,
        marker="s", capsize=3, label="Column HHI (attention demand)", color="coral"
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("HHI")
    ax.set_title("(a) Attention Concentration by Layer")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Heatmap of row HHI by layer and head
    ax = axes[1]
    im = ax.imshow(mean_row_hhi.T, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Head")
    ax.set_title("(b) Row HHI by Layer and Head")
    plt.colorbar(im, ax=ax, label="HHI")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hhi.pdf", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'hhi.pdf'}")


# ---------------------------------------------------------------------------
# Diagnostic 5: Head Aggregation and Diversification
# ---------------------------------------------------------------------------


def diagnostic_head_aggregation(model, tokenizer):
    """
    Measure how averaging across heads affects IIA deviations.

    Compare IIA deviations in individual heads vs head-averaged attention.
    Each head computes a logit with its own utility function; averaging
    across heads diversifies, stabilizing the aggregate ratio.
    """
    print("\n=== Diagnostic 5: Head Aggregation and Diversification ===")

    n_heads = 12

    j_idx = 0
    k_idx = 1

    # Collect per-head ratios and head-averaged ratios across extensions
    all_per_head_ratios = []   # (n_ext, n_layers, n_heads)
    all_avg_ratios = []        # (n_ext, n_layers)

    for ext in IIA_EXTENSIONS:
        text = IIA_BASE + ext
        attentions, _, tokens = get_attention_and_logits(model, tokenizer, text)

        i_idx = len(tokens) - 1
        n_layers = len(attentions)

        per_head = np.zeros((n_layers, n_heads))
        avg = np.zeros(n_layers)

        for l, attn in enumerate(attentions):
            # Per-head ratios
            for h in range(n_heads):
                a_ij = attn[h, i_idx, j_idx]
                a_ik = attn[h, i_idx, k_idx]
                if a_ik > 1e-10:
                    per_head[l, h] = a_ij / a_ik
                else:
                    per_head[l, h] = np.nan

            # Head-averaged attention
            avg_attn = attn.mean(axis=0)  # (seq_len, seq_len)
            a_ij_avg = avg_attn[i_idx, j_idx]
            a_ik_avg = avg_attn[i_idx, k_idx]
            if a_ik_avg > 1e-10:
                avg[l] = a_ij_avg / a_ik_avg
            else:
                avg[l] = np.nan

        all_per_head_ratios.append(per_head)
        all_avg_ratios.append(avg)

    all_per_head_ratios = np.array(all_per_head_ratios)  # (n_ext, n_layers, n_heads)
    all_avg_ratios = np.array(all_avg_ratios)            # (n_ext, n_layers)

    # Compute violations
    log_ph = np.log(np.where(all_per_head_ratios > 0, all_per_head_ratios, np.nan))
    log_avg = np.log(np.where(all_avg_ratios > 0, all_avg_ratios, np.nan))

    ph_violation = np.nanstd(log_ph, axis=0)  # (n_layers, n_heads)
    avg_violation = np.nanstd(log_avg, axis=0)  # (n_layers,)

    mean_ph_by_layer = np.nanmean(ph_violation, axis=1)  # (n_layers,)

    fig, ax = plt.subplots(figsize=(8, 5))

    layers = np.arange(len(mean_ph_by_layer))
    width = 0.35
    ax.bar(layers - width / 2, mean_ph_by_layer, width,
           label="Individual heads (mean)", color="steelblue", alpha=0.8)
    ax.bar(layers + width / 2, avg_violation, width,
           label="Head-averaged attention", color="darkorange", alpha=0.8)

    ax.set_xlabel("Layer")
    ax.set_ylabel("IIA Deviation (Std of log ratio)")
    ax.set_title("Head Aggregation: Individual Heads vs Head-Averaged Attention")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "head_aggregation.pdf", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'head_aggregation.pdf'}")


# ---------------------------------------------------------------------------
# Diagnostic 6: Inclusive Value vs. Logit Lens
# ---------------------------------------------------------------------------


def diagnostic_iv_vs_logit_lens(model, tokenizer):
    """
    Compare inclusive value trajectories with logit lens prediction convergence.

    Inclusive value (from the discrete choice framework) measures the richness of
    the attention information environment at each layer. The logit lens
    (nostalgebraist 2020) projects the residual stream at each layer to vocabulary
    space, measuring how close the intermediate representation is to the final
    prediction. These track different dimensions of processing.
    """
    print("\n=== Diagnostic 6: Inclusive Value vs. Logit Lens ===")

    # Load LM head model for logit lens
    model_lm = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model_lm.eval()
    model_lm.to(DEVICE)
    lm_model = model_lm.transformer
    ln_f = lm_model.ln_f
    lm_head = model_lm.lm_head

    all_iv = []
    all_kl = []

    for sent in SENTENCES:
        inputs = tokenizer(sent, return_tensors="pt").to(DEVICE)
        seq_len = inputs["input_ids"].shape[1]

        # --- Inclusive value via hooks on the original model ---
        logits_all, hooks = _register_logit_hooks(model)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for h in hooks:
            h.remove()

        iv_per_layer = []
        for l in range(len(logits_all)):
            logits_l = logits_all[l]
            n_heads = logits_l.shape[0]
            ivs = []
            for h_idx in range(n_heads):
                for i in range(seq_len):
                    scores = logits_l[h_idx, i, :]
                    finite = scores[np.isfinite(scores)]
                    if len(finite) > 0:
                        max_s = np.max(finite)
                        iv = max_s + np.log(np.sum(np.exp(finite - max_s)))
                        ivs.append(iv)
            iv_per_layer.append(np.mean(ivs))

        # --- Logit lens via hidden states from the LM model ---
        logits_all2, hooks2 = _register_logit_hooks(lm_model)
        with torch.no_grad():
            outputs_lm = lm_model(**inputs, output_hidden_states=True)
        for h in hooks2:
            h.remove()

        hidden_states = outputs_lm.hidden_states
        n_layers = len(hidden_states) - 1

        # Final-layer vocab distribution (last token)
        h_final = hidden_states[-1][0, -1]
        logits_final = lm_head(ln_f(h_final))
        probs_final = torch.softmax(logits_final, dim=-1)
        log_p_final = torch.log_softmax(logits_final, dim=-1)

        # KL divergence from final layer at each layer
        kl_per_layer = []
        for layer_idx in range(1, n_layers + 1):
            h_l = hidden_states[layer_idx][0, -1]
            logits_l = lm_head(ln_f(h_l))
            log_p_l = torch.log_softmax(logits_l, dim=-1)
            kl = torch.sum(probs_final * (log_p_final - log_p_l)).item()
            kl_per_layer.append(kl)

        all_iv.append(np.array(iv_per_layer))
        all_kl.append(np.array(kl_per_layer))

    mean_iv = np.mean(all_iv, axis=0)
    std_iv = np.std(all_iv, axis=0)
    mean_kl = np.mean(all_kl, axis=0)
    std_kl = np.std(all_kl, axis=0)
    layers = np.arange(len(mean_iv))

    # Key statistics
    iv_abs_change = np.abs(np.diff(mean_iv))
    kl_abs_change = np.abs(np.diff(mean_kl))
    iv_frac = iv_abs_change / iv_abs_change.sum()
    kl_frac = kl_abs_change / kl_abs_change.sum()
    corr = np.corrcoef(iv_frac, kl_frac)[0, 1]

    print(f"  IV fraction in layers 0-4:  {iv_frac[:4].sum():.1%}")
    print(f"  KL fraction in layers 9-11: {kl_frac[8:].sum():.1%}")
    print(f"  Correlation |ΔIV| vs |ΔKL|: {corr:.2f}")

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: dual y-axis trajectories
    ax1 = axes[0]
    color_iv = "steelblue"
    color_kl = "coral"

    ln1 = ax1.plot(layers, mean_iv, "o-", color=color_iv, linewidth=2, markersize=5,
                   label="Inclusive value $\\mathcal{V}$")
    ax1.fill_between(layers, mean_iv - std_iv, mean_iv + std_iv,
                     color=color_iv, alpha=0.15)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean Inclusive Value (log $Z_i$)", color=color_iv)
    ax1.tick_params(axis="y", labelcolor=color_iv)

    ax2 = ax1.twinx()
    ln2 = ax2.plot(layers, mean_kl, "s-", color=color_kl, linewidth=2, markersize=5,
                   label="KL from final layer")
    ax2.fill_between(layers, mean_kl - std_kl, mean_kl + std_kl,
                     color=color_kl, alpha=0.15)
    ax2.set_ylabel("KL$(p_{\\mathrm{final}} \\| p_{\\ell})$", color=color_kl)
    ax2.tick_params(axis="y", labelcolor=color_kl)
    ax2.invert_yaxis()

    all_lines = ln1 + ln2
    all_labels = [line.get_label() for line in all_lines]
    ax1.legend(all_lines, all_labels, loc="center right", fontsize=9)
    ax1.set_title("(a) Inclusive Value vs. Prediction Convergence")
    ax1.grid(True, alpha=0.3)

    # Panel B: fraction of total change per transition
    ax = axes[1]
    x = np.arange(len(iv_frac))
    width = 0.35
    ax.bar(x - width / 2, iv_frac, width, label="Inclusive value",
           color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, kl_frac, width, label="Prediction (KL)",
           color="coral", alpha=0.8)

    ax.set_xlabel("Layer Transition")
    ax.set_ylabel("Fraction of Total Change")
    ax.set_title("(b) Where Processing Happens")
    labels_x = [f"{l}\u2192{l+1}" for l in range(len(iv_frac))]
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=7)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    ax.axvspan(-0.5, 3.5, alpha=0.06, color="steelblue")
    ax.axvspan(len(iv_frac) - 3.5, len(iv_frac) - 0.5, alpha=0.06, color="coral")
    ylim = ax.get_ylim()
    ax.text(1.5, ylim[1] * 0.92, "IV-dominated", fontsize=8, color="steelblue",
            ha="center", style="italic")
    ax.text(len(iv_frac) - 2, ylim[1] * 0.92, "Prediction-\ndominated", fontsize=8,
            color="coral", ha="center", style="italic")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "iv_vs_logit_lens.pdf", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'iv_vs_logit_lens.pdf'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Attention diagnostics for 'Discrete Choice in Silicon'"
    )
    parser.add_argument("--all", action="store_true", help="Run all diagnostics")
    parser.add_argument("--inclusive", action="store_true", help="Inclusive value trajectories")
    parser.add_argument("--iia", action="store_true", help="IIA test")
    parser.add_argument("--temperature", action="store_true", help="Temperature estimation")
    parser.add_argument("--hhi", action="store_true", help="Concentration (HHI)")
    parser.add_argument("--head-agg", action="store_true", help="Head aggregation diagnostic")
    parser.add_argument("--iv-lens", action="store_true", help="IV vs logit lens")
    args = parser.parse_args()

    # Default to all if nothing specified
    if not any([args.all, args.inclusive, args.iia, args.temperature, args.hhi,
                args.head_agg, args.iv_lens]):
        args.all = True

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model()

    if args.all or args.inclusive:
        diagnostic_inclusive_value(model, tokenizer)

    if args.all or args.iia:
        diagnostic_iia(model, tokenizer)

    if args.all or args.temperature:
        diagnostic_temperature(model, tokenizer)

    if args.all or args.hhi:
        diagnostic_hhi(model, tokenizer)

    if args.all or args.head_agg:
        diagnostic_head_aggregation(model, tokenizer)

    if args.all or args.iv_lens:
        diagnostic_iv_vs_logit_lens(model, tokenizer)

    print("\nAll diagnostics complete.")


if __name__ == "__main__":
    main()
