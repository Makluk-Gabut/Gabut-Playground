# TorchLM-v2 — Bob

**Bob is the successor to the original PyTorch TorchLM project** in `LM/TorchLM`.

Bob keeps the original model's decoder-only Transformer, rotary position embeddings (RoPE), byte-level BPE tokenizer, tied input/output embeddings, PyTorch training, checkpointing, and sampling.

## Model size

Defaults: vocabulary capped at 8,192, 512 hidden dimensions, 8 attention heads, 6 Transformer blocks, a 1,536-wide MLP, and a 512-token context. With an 8,192-token vocabulary, this configuration has **19,960,832 trainable parameters** (about 19.96M). The vocabulary embedding and output projection share weights. The script enforces both the 8,192-token vocabulary cap and the 20,000,000-parameter model limit; actual parameter count is printed at startup.

## Setup

```bash
python -m pip install -r requirements.txt
```

A CUDA GPU is recommended. Reduce `--batch-size` or `--block-size` if GPU memory is insufficient.

## Prepare Indonesian Wikipedia

```bash
python prepare_wikipedia.py --output /content/drive/MyDrive/TorchLM-v2/data/idwiki.jsonl
```

The helper downloads Wikimedia's current-only article dump, extracts namespace-0 non-redirect articles, strips wikitext markup, and writes JSONL with title, revision ID, timestamp, URL, and text. The `latest` URL selects the dump available at download time.

Wikipedia text is generally under CC BY-SA. Retain article attribution and follow applicable license terms.

## Train

```bash
python Bob.py \
  --data /content/drive/MyDrive/TorchLM-v2/data/idwiki.jsonl \
  --output-dir /content/drive/MyDrive/TorchLM-v2/runs/Bob \
  --steps 12000 --vocab-size 8192 --require-gpu
```

Training uses a 1-in-20 article validation split, memory-mapped token caches, `bob_latest.pt`, `bob_best.pt`, CSV loss logs, and `sample.txt`.
