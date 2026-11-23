# TorchLM-v2 — Bob

**Bob is the successor to the original PyTorch TorchLM project** in `LM/TorchLM`.

Bob keeps the original model's decoder-only Transformer, rotary position embeddings (RoPE), byte-level BPE tokenizer, tied input/output embeddings, PyTorch training, checkpointing, and sampling.

## Model size and activation

The default model uses an 8,192-token maximum vocabulary, 512 hidden dimensions, 8 attention heads, 6 Transformer blocks, and a 512-token context. Input and output embeddings are tied.

**SwiGLU is the default activation.** It uses an FFN width of 1,024. At exactly 8,192 vocabulary entries, the model has **19,963,904 trainable parameters** (about 19.96M). GELU remains available as an optional baseline, with an FFN width of 1,536 and **19,960,832 trainable parameters**.

The script enforces both the 8,192-token vocabulary cap and the 20,000,000-parameter limit. The actual tokenizer vocabulary may be smaller than the requested size, and the script prints and checks the actual parameter count at startup.

Use `--activation swiglu` explicitly if desired. Use `--activation gelu` to compare against the GELU baseline. Override FFN size with `--ff-dim` if needed, but configurations over 20M parameters will be rejected.

## Gradient checkpointing

Gradient checkpointing is disabled by default. Enable it with `--gradient-checkpointing` to recompute Transformer block activations during backpropagation instead of retaining them. It reduces activation memory at the cost of extra compute/time.

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

Training uses a 1-in-20 article validation split, memory-mapped token caches, `bob_latest.pt`, `bob_best.pt`, CSV loss logs, and `sample.txt`.
