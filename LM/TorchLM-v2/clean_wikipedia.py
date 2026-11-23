"""Clean JSONL article text while preserving records and provenance.

Normalizes HTML entities/tags, control characters, and whitespace; skips empty
or short documents; optionally removes exact duplicate article text.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HTML_TAGS = re.compile(r"<[^>]+>")
WHITESPACE = re.compile(r"\s+")


def clean_text(value) -> str:
    text = "" if value is None else str(value)
    text = html.unescape(text)
    text = HTML_TAGS.sub(" ", text)
    text = CONTROL_CHARS.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw JSONL from download_hf_wikipedia.py")
    parser.add_argument("--output", type=Path, default=Path("data/idwiki_clean.jsonl"))
    parser.add_argument("--min-chars", type=int, default=100, help="Discard articles shorter than this after cleaning")
    parser.add_argument("--no-dedupe", action="store_true", help="Keep exact duplicate article texts")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    if args.min_chars < 0:
        parser.error("--min-chars cannot be negative")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".part")
    seen: set[bytes] = set()
    read_count = written = empty_or_short = duplicate = malformed = 0
    try:
        with args.input.open("r", encoding="utf-8") as source, partial.open("w", encoding="utf-8") as output:
            for line_number, line in enumerate(source, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(record, dict):
                    malformed += 1
                    continue
                read_count += 1
                title = clean_text(record.get("title", ""))
                text = clean_text(record.get("text", record.get("content", "")))
                if len(text) < args.min_chars:
                    empty_or_short += 1
                    continue
                if not args.no_dedupe:
                    digest = hashlib.sha256(text.encode("utf-8")).digest()
                    if digest in seen:
                        duplicate += 1
                        continue
                    seen.add(digest)
                record["title"] = title
                record["text"] = text
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if written and written % 50000 == 0:
                    output.flush()
                    print(f"Cleaned {written:,} articles… (line {line_number:,})")
        partial.replace(args.output)
    except Exception:
        print(f"Cleaning did not finish. Partial output is at: {partial}")
        raise

    stats = {
        "input": str(args.input),
        "output": str(args.output),
        "records_read": read_count,
        "records_written": written,
        "empty_or_short_skipped": empty_or_short,
        "duplicates_skipped": duplicate,
        "malformed_lines_skipped": malformed,
        "minimum_characters": args.min_chars,
        "deduplication_enabled": not args.no_dedupe,
    }
    metadata = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done: {written:,} clean articles saved to {args.output}")
    print(f"Skipped: {empty_or_short:,} short/empty, {duplicate:,} duplicates, {malformed:,} malformed")
    print(f"Stats: {metadata}")


if __name__ == "__main__":
    main()
