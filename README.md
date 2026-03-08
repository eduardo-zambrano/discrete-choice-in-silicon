# Replication Package

## "Rational Inattention in Silicon"

Eduardo Zambrano
Department of Economics, California Polytechnic State University
Email: ezambran@calpoly.edu

---

## Overview

This package contains the code to reproduce all figures in "Rational Inattention in Silicon." The code runs forward-pass diagnostics on GPT-2 small (117M parameters) and requires no external datasets. All computations are deterministic.

## Data Availability and Provenance

### Statement about Rights

The author has legitimate access to all data and code used in this paper.

### Summary of Availability

No external datasets are used. The pretrained GPT-2 small model (117M parameters) is downloaded automatically from HuggingFace on first execution. The input corpus consists of 8 English sentences hardcoded in the script (lines 47--56 of `code/attention_diagnostics.py`). All data used in this paper are publicly available.

| Data Source | Provider | Access |
|---|---|---|
| GPT-2 small pretrained weights | HuggingFace / OpenAI | https://huggingface.co/openai-community/gpt2 |

### Data Citations

- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Blog*.
- Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., Drame, M., Lhoest, Q., & Rush, A. (2020). Transformers: State-of-the-Art Natural Language Processing. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 38--45.

## Computational Requirements

### Software Requirements

- Python 3.10 or higher
- Required packages (with tested versions):

| Package | Version |
|---|---|
| torch | 2.10.0 |
| transformers | 5.3.0 |
| numpy | 2.2.6 |
| matplotlib | 3.10.8 |

All packages are installed automatically via `pip install -r requirements.txt`.

### Controlled Randomness

No pseudo-random number generators are used. All results are fully deterministic given the same model weights and input text. Minor floating-point differences may arise across CPU architectures (x86 vs. ARM) but do not affect the qualitative results or visual appearance of the figures.

### Memory, Runtime, and Storage

| Resource | Requirement |
|---|---|
| Runtime | Approximately 2--5 minutes (CPU only) |
| Memory | ~2 GB RAM |
| Storage | ~600 MB (model weights ~500 MB, cached by HuggingFace; output figures ~0.5 MB) |

The GPT-2 small model weights (~500 MB) are downloaded from HuggingFace on first execution and cached locally (`~/.cache/huggingface/hub/` by default). Subsequent runs use the cached model and do not require internet access.

Tested on: macOS (Apple Silicon, 16 GB RAM).

## Description of Programs

| File | Description |
|---|---|
| `code/attention_diagnostics.py` | Main Python script. Runs 5 diagnostic computations on GPT-2 small and produces 5 PDF figures. |
| `code/run_all.sh` | Master shell script. Creates virtual environment, installs dependencies, runs all diagnostics. |
| `requirements.txt` | Pinned Python dependencies. |

No data cleaning or preparation step is needed. The script loads the pretrained model, runs forward passes on the hardcoded corpus, extracts attention weights and pre-softmax logits, computes diagnostics, and saves figures.

## Instructions to Replicators

### Quick Start

```bash
cd replication
bash code/run_all.sh
```

### Manual Steps

```bash
cd replication
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python code/attention_diagnostics.py --all
```

Output figures will appear in `output/figures/`.

### Running Individual Diagnostics

```bash
python code/attention_diagnostics.py --inclusive     # Figure 1
python code/attention_diagnostics.py --iia           # Figure 2
python code/attention_diagnostics.py --temperature   # Figure 3
python code/attention_diagnostics.py --head-agg      # Figure 4
python code/attention_diagnostics.py --hhi           # Figure A.1
```

## List of Figures and Programs

| Figure in Paper | Output File | Program | CLI Flag |
|---|---|---|---|
| Figure 1: Inclusive value trajectories | `output/figures/inclusive_value.pdf` | `code/attention_diagnostics.py` | `--inclusive` |
| Figure 2: IIA deviations | `output/figures/iia_test.pdf` | `code/attention_diagnostics.py` | `--iia` |
| Figure 3: Attention entropy ratio | `output/figures/temperature.pdf` | `code/attention_diagnostics.py` | `--temperature` |
| Figure 4: Head aggregation | `output/figures/head_aggregation.pdf` | `code/attention_diagnostics.py` | `--head-agg` |
| Figure A.1: Attention concentration (HHI) | `output/figures/hhi.pdf` | `code/attention_diagnostics.py` | `--hhi` |

There are no tables in this paper.

## License

Code is licensed under the Modified BSD License (3-Clause). Documentation and output are licensed under Creative Commons Attribution 4.0 International. See `LICENSE.txt` for details.

## References

- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Blog*.
- Wolf, T., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. *EMNLP System Demonstrations*.
