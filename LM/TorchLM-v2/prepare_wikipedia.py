"""Download and extract the latest Indonesian Wikipedia article dump."""
from __future__ import annotations

import argparse
import bz2
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mwparserfromhell
import requests

DUMP_URL = "https://dumps.wikimedia.org/idwiki/latest/idwiki-latest-pages-articles.xml.bz2"
USER_AGENT = "TorchLM-v2-Bob/1.0 (educational corpus preparation)"


def lname(tag):
    return tag.rsplit("}", 1)[-1]


def download(url, destination):
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=(30, 180)) as response:
        if response.status_code == 429:
            raise RuntimeError("Wikimedia returned HTTP 429; wait before retrying")
        response.raise_for_status()
        size = int(response.headers.get("Content-Length", 0))
        free = shutil.disk_usage(destination.parent).free
        if size and size > free:
            raise OSError(f"Dump requires {size/1e9:.1f} GB; only {free/1e9:.1f} GB is free")
        with partial.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)
    partial.replace(destination)


def extract(dump_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with bz2.open(dump_path, "rb") as source, output_path.open("w", encoding="utf-8") as output:
        for _, page in ET.iterparse(source, events=("end",)):
            if lname(page.tag) != "page":
                continue
            title = namespace = ""
            redirect = False
            revision = None
            for child in page:
                name = lname(child.tag)
                if name == "title":
                    title = child.text or ""
                elif name == "ns":
                    namespace = child.text or ""
                elif name == "redirect":
                    redirect = True
                elif name == "revision":
                    revision = child
            if namespace == "0" and not redirect and revision is not None:
                revision_id = timestamp = wikitext = ""
                for item in revision:
                    name = lname(item.tag)
                    if name == "id" and not revision_id:
                        revision_id = item.text or ""
                    elif name == "timestamp":
                        timestamp = item.text or ""
                    elif name == "text":
                        wikitext = "".join(item.itertext())
                if wikitext.strip():
                    text = mwparserfromhell.parse(wikitext).strip_code(normalize=True, collapse=True)
                    text = " ".join(text.split())
                    if text:
                        output.write(json.dumps({
                            "title": title,
                            "revision_id": revision_id,
                            "timestamp": timestamp,
                            "url": "https://id.wikipedia.org/wiki/" + title.replace(" ", "_"),
                            "text": text,
                        }, ensure_ascii=False) + "\n")
                        count += 1
            page.clear()
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DUMP_URL)
    parser.add_argument("--output", type=Path, default=Path("data/idwiki.jsonl"))
    parser.add_argument("--dump", type=Path, default=None)
    parser.add_argument("--delete-dump-after-extract", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    dump = args.dump.resolve() if args.dump else output.with_suffix(".xml.bz2")
    dump.parent.mkdir(parents=True, exist_ok=True)
    if not dump.exists():
        download(args.url, dump)
    count = extract(dump, output)
    metadata = output.with_suffix(output.suffix + ".metadata.json")
    metadata.write_text(json.dumps({
        "source": args.url,
        "project": "Indonesian Wikipedia current-articles dump",
        "article_count": count,
        "license_note": "Wikipedia text is generally under CC BY-SA; retain attribution and follow applicable license terms.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {count:,} articles: {output}")
    if args.delete_dump_after_extract:
        dump.unlink()


if __name__ == "__main__":
    main()
