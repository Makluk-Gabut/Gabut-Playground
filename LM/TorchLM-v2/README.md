# TorchLM-v2 — Bob

**Bob is the successor to the original PyTorch TorchLM project** in `LM/TorchLM`.

Bob keeps the original model's decoder-only Transformer, rotary position embeddings (RoPE), byte-level BPE tokenizer, tied input/output embeddings, PyTorch training, checkpointing, and sampling.

## Model size and activation

The default model uses an 8,192-token maximum vocabulary, 512 hidden dimensions, 8 attention heads, 6 Transformer blocks, and a 512-token context. Input and output embeddings are tied.

**SwiGLU is the default activation.** It uses an FFN width of 1,024. At exactly 8,192 vocabulary entries, the model has **19,963,904 trainable parameters** (about 19.96M). GELU remains available as an optional baseline, with an FFN width of 1,536 and **19,960,832 trainable parameters**.

The script enforces both the 8,192-token vocabulary cap and the 20,000,000-parameter limit. The actual tokenizer vocabulary may be smaller than the requested size, and the script prints and checks the actual parameter count at startup.

Use `--activation gelu` to compare against the GELU baseline. Override FFN size with `--ff-dim` if needed, but configurations over 20M parameters will be rejected.

## Setup

```bash
python -m pip install -r requirements.txt
```

A CUDA GPU is recommended. Reduce `--batch-size` or `--block-size` if GPU memory is insufficient.

## Prepare data

On Google Colab, mount Drive and install requirements, then stream the dataset and clean it:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
pip install -r requirements.txt
python download_hf_wikipedia.py \
  --output /content/drive/MyDrive/TorchLM-v2/data/idwiki_raw.jsonl
python clean_wikipedia.py \
  --input /content/drive/MyDrive/TorchLM-v2/data/idwiki_raw.jsonl \
  --output /content/drive/MyDrive/TorchLM-v2/data/idwiki.jsonl
```

Check the Hugging Face dataset card for the source snapshot's license and attribution requirements. Retain appropriate Wikipedia attribution and comply with the applicable license terms.

## Train Bob

```bash
python Bob.py \
  --data /content/drive/MyDrive/TorchLM-v2/data/idwiki.jsonl \
  --output-dir /content/drive/MyDrive/TorchLM-v2/runs/Bob \
  --steps 98000 --vocab-size 8192 --require-gpu --precision fp16
```

Bob uses FP32 model parameters and AdamW master weights with FP16 autocast on a T4, plus activation checkpointing. CUDA runs use fused AdamW when supported by the installed PyTorch. By default, validation runs every 1,000 steps with 5 batches per split, and the resumable `bob_latest.pt` checkpoint is saved every 1,000 steps (and at the final step). The best validation checkpoint, CSV loss log, and generated sample are also saved. Checkpoints use a temporary file followed by an atomic rename to reduce the chance of leaving a partially written checkpoint.

Adjust `--checkpoint-interval` to change save frequency. More frequent validation/checkpoint writes can slow training, especially when the output directory is on Google Drive. Resume from the latest checkpoint with `--resume /path/to/bob_latest.pt`.
