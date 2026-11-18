# TorchLM — Shakespeare language-model experiment

TorchLM is a small, decoder-only Transformer language model written in PyTorch. This project is a proof of concept: it demonstrates that a compact model can learn surface-level patterns from a Shakespeare text corpus, including Early Modern English spellings, speaker labels, and play-like dialogue formatting. It is not intended to produce reliable or consistently coherent prose.

## Project contents

- `torchlm.py` — tokenizer preparation, model definition, training, checkpoint saving, and text generation.
- `nanolm_bpe_2k.json` — the trained byte-level BPE tokenizer used for the experiment.
- `loss.csv` and `loss.png` — recorded training and validation loss.
- `training_12000.log` — training log for the 12,000-step run.
- `torchlm_inference_tuning.csv` — sample outputs from decoding-parameter experiments.

The training corpus and model checkpoint are not included. Supply the corpus yourself and keep it consistent with the tokenizer/checkpoint when reproducing inference.

## Requirements

- Python 3.10 or later
- PyTorch 2.3 or later
- `tokenizers` (see `requirements.txt`)
- CUDA GPU is recommended for training; CPU is supported but can be much slower.

## Installation

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Prepare a tokenizer

If you have the text corpus and want to train a tokenizer:

```bash
python torchlm.py \\
  --data /path/to/shakespeare_corpus.txt \\
  --prepare-tokenizer \\
  --tokenizer nanolm_bpe_2k.json \\
  --vocab-size 2000
```

The repository already includes `nanolm_bpe_2k.json` from the experiment, so you can reuse it when the corpus/checkpoint pairing is appropriate.

## Train

```bash
python torchlm.py \\
  --data /path/to/shakespeare_corpus.txt \\
  --tokenizer nanolm_bpe_2k.json \\
  --steps 12000 \\
  --batch-size 32 \\
  --block-size 128 \\
  --eval-interval 250 \\
  --checkpoint nanolm_2k_12000.pt \\
  --best-checkpoint nanolm_2k_12000_best.pt
```

The script splits the token stream into training and validation portions, evaluates periodically, and saves the best-validation and final checkpoints. On a CUDA GPU, automatic precision selection uses BF16 when supported, otherwise FP16. You can select `--device` and `--precision` explicitly; run `python torchlm.py --help` to see all options.

## Generate text

Training prints a generated sample at the end. You can change the prompt and decoding settings with the command-line options, for example:

```bash
python torchlm.py \\
  --data /path/to/shakespeare_corpus.txt \\
  --tokenizer nanolm_bpe_2k.json \\
  --resume nanolm_2k_12000_best.pt \\
  --steps 12000 \\
  --prompt "To be, or not to be:" \\
  --generate 120
```

Note: in the current script, `--resume` resumes training rather than providing a standalone inference-only command. The Python API also exposes `NanoTransformer.generate(...)` for sampling from a loaded model. Sampling temperature and `top_k` control token selection and variation; they cannot make the model learn patterns it did not learn during training.

## Experiment snapshot

The included log records a 12,000-step run with a 2,000-token vocabulary and 2,238,976 trainable parameters. Its best recorded validation loss was approximately **3.4607** at step **11,250**. Generated samples imitate some Shakespeare-like formatting and spelling, but sentence-level coherence is limited. This is an expected limitation of this small proof-of-concept model, not a claim that it understands Shakespeare.

The loss curve and sample-tuning CSV are included as experiment artifacts. Results depend on the exact corpus, tokenizer, checkpoint, and sampling settings.

## Limitations

- A generated passage can mix characters, plays, and unrelated dialogue.
- The model predicts tokens from training patterns; it does not verify facts or preserve plot/context reliably.
- Sampling settings affect diversity and repetition, not the underlying learned capability.
- Use only corpora you are permitted to train on, and review their license/terms before redistribution.
