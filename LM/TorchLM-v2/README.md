# TorchLM-v2 — Bob

**Bob is the successor to the original PyTorch TorchLM project** in `LM/TorchLM`.

Bob keeps the original model's decoder-only Transformer, rotary position embeddings (RoPE), byte-level BPE tokenizer, tied input/output embeddings, PyTorch training, checkpointing, and sampling.

## Model size and activations

The default model uses an 8,192-token maximum vocabulary, 512 hidden dimensions, 8 attention heads, 6 Transformer blocks, and a 512-token context. Input and output embeddings are tied. By default, GELU uses an FFN width of 1,536, for **19,960,832 trainable parameters** at exactly 8,192 vocabulary entries.

SwiGLU is also supported. Its default FFN width is 1,024 to account for its gate and up projections; with an 8,192 vocabulary it has **19,963,904 trainable parameters**. Both stay under the hard 20,000,000-parameter limit. The script prints and checks the actual count; the actual tokenizer vocabulary can be smaller than the requested cap.

Select the activation with `--activation gelu` or `--activation swiglu`. You can override the FFN width with `--ff-dim`, but the script rejects configurations over the parameter limit.

## Gradient checkpointing

Use `--gradient-checkpointing` to recompute Transformer blocks during backpropagation instead of retaining all their activations. It reduces activation-memory use at the cost of extra compute/time. It is off by default; enable it if GPU memory is tight.

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

## Train Bob

```bash
python Bob.py \
  --data /content/drive/MyDrive/TorchLM-v2/data/idwiki.jsonl \
  --output-dir /content/drive/MyDrive/TorchLM-v2/runs/Bob \
  --steps 12000 --vocab-size 8192 --activation swiglu \
  --gradient-checkpointing --require-gpu
```

Use `--activation gelu` for the GELU baseline. Training uses a 1-in-20 article validation split, memory-mapped token caches, `bob_latest.pt`, `bob_best.pt`, CSV loss logs, and `sample.txt`.
