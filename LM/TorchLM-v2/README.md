# TorchLM-v2 — Bob

**Bob is the successor to the original PyTorch TorchLM project** in `LM/TorchLM`.

Bob retains the original model's decoder-only Transformer, rotary position embeddings (RoPE), byte-level BPE tokenizer, tied input/output embeddings, PyTorch training, checkpointing, and sampling.

## Model size

Defaults: 16,384 vocabulary, 384 hidden dimensions, 6 attention heads, 6 Transformer blocks, 1,024-wide MLP, and 512-token context. With tied embeddings this is approximately **14.5 million parameters**. The exact count is printed by `Bob.py` and depends on the resulting tokenizer vocabulary size.

## Setup

```bash
python -m pip install -r requirements.txt
```

Use a CUDA GPU when available. Reduce `--batch-size` or `--block-size` if VRAM is insufficient.

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
  --steps 12000 --vocab-size 16384 --require-gpu
```

Training uses a 1-in-20 article validation split, memory-mapped token caches, `bob_latest.pt`, `bob_best.pt`, CSV loss logs, and `sample.txt`.
