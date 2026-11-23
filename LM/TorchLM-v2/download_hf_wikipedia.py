"""Stream an Indonesian Wikipedia Hugging Face dataset to Bob-compatible JSONL.

Example (Google Colab):
    python download_hf_wikipedia.py --output /content/drive/MyDrive/TorchLM-v2/data/idwiki_raw.jsonl
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit("Install dependencies with: pip install -r requirements.txt") from exc

DEFAULT_DATASET = "indonesian-nlp/wikipedia-id"


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset repository ID")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("data/idwiki_raw.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of rows")
    parser.add_argument("--log-every", type=int, default=10000)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.log_every < 1:
        parser.error("--log-every must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".part")
    print(f"Streaming {args.dataset} [{args.split}] from Hugging Face…")
    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    written = skipped = 0
    try:
        with partial.open("w", encoding="utf-8") as output:
            for row in dataset:
                if args.limit is not None and written >= args.limit:
                    break
                text = as_text(row.get("text", row.get("content", "")))
                if not text:
                    skipped += 1
                    continue
                record = {
                    "title": as_text(row.get("title", "")),
                    "text": text,
                    "docid": as_text(row.get("docid", row.get("id", ""))),
                    "source": args.dataset,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if written % args.log_every == 0:
                    output.flush()
                    print(f"Saved {written:,} articles; skipped empty rows: {skipped:,}")
        partial.replace(args.output)
    except Exception:
        print(f"Download did not finish. Partial output is at: {partial}")
        raise

    metadata = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata.write_text(json.dumps({
        "source_dataset": args.dataset,
        "split": args.split,
        "articles_written": written,
        "empty_rows_skipped": skipped,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "license_note": "Check the Hugging Face dataset card for the dataset's declared license and attribution requirements.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done: saved {written:,} articles to {args.output}")
    print(f"Metadata: {metadata}")


if __name__ == "__main__":
    main()
