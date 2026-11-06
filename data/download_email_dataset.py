"""Download the public MeAJOR multi-source phishing email corpus from Zenodo."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

RECORD_ID = "18471483"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
CHUNK_SIZE = 4 * 1024 * 1024
USER_AGENT = "PBL-Phishing-Detector/1.0 (local public-dataset training)"


def download(url: str, target: Path, expected_size: int, expected_md5: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == expected_size:
        digest = hashlib.md5()
        with target.open("rb") as file:
            for block in iter(lambda: file.read(CHUNK_SIZE), b""):
                digest.update(block)
        if digest.hexdigest() == expected_md5:
            print(f"Already downloaded and checksum verified: {target}")
            return
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    digest = hashlib.md5()
    try:
        with temporary.open("wb") as output:
            for start in range(0, expected_size, CHUNK_SIZE):
                end = min(expected_size - 1, start + CHUNK_SIZE - 1)
                request = Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT})
                with urlopen(request, timeout=120) as response:
                    expected_range = f"bytes {start}-{end}/{expected_size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError(f"Range download unavailable: HTTP {response.status}, Content-Range={response.headers.get('Content-Range')!r}")
                    block = response.read(end - start + 1)
                if len(block) != end - start + 1:
                    raise RuntimeError(f"Incomplete download at byte {start}")
                output.write(block)
                digest.update(block)
                print(f"MeAJOR CSV: {end + 1:,}/{expected_size:,} bytes", flush=True)
        if temporary.stat().st_size != expected_size or digest.hexdigest() != expected_md5:
            raise RuntimeError("Downloaded file failed size or MD5 verification")
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def prepare(source: Path, output: Path) -> None:
    header = pd.read_csv(source, nrows=0)
    columns = {str(column).strip().lower(): column for column in header.columns}
    subject_col = columns.get("subject")
    body_col = columns.get("body")
    label_col = columns.get("label")
    source_col = columns.get("source")
    if not body_col or not label_col:
        raise ValueError(f"Expected body and label columns; found {list(header.columns)}")
    usecols = [column for column in (subject_col, body_col, label_col, source_col) if column]
    frame = pd.read_csv(source, usecols=usecols, low_memory=False)
    rename = {body_col: "body", label_col: "label"}
    if subject_col:
        rename[subject_col] = "subject"
    if source_col:
        rename[source_col] = "source"
    frame = frame.rename(columns=rename)
    if "subject" not in frame:
        frame["subject"] = ""
    if "source" not in frame:
        frame["source"] = "unspecified"
    frame["label"] = pd.to_numeric(frame.label, errors="coerce")
    frame = frame.dropna(subset=["body", "label"])
    if not set(frame.label.astype(int).unique()).issubset({0, 1}):
        raise ValueError("MeAJOR label column must contain only 0=benign and 1=phishing")
    frame["label"] = frame.label.astype(int)
    frame["subject"] = frame.subject.fillna("").astype(str)
    frame["body"] = frame.body.fillna("").astype(str)
    frame["source"] = frame.source.fillna("unspecified").astype(str)
    frame = frame[frame.subject.str.strip().ne("") | frame.body.str.strip().ne("")]
    frame = frame.drop_duplicates(["subject", "body"]).reset_index(drop=True)
    if frame.label.nunique() != 2:
        raise ValueError("Dataset does not contain both benign and phishing labels")
    print("Dataset inspection:")
    print(f"  rows after cleaning: {len(frame):,}")
    print(f"  label distribution (0=benign, 1=phishing): {frame.label.value_counts().sort_index().to_dict()}")
    print(f"  source distribution: {frame.source.value_counts().to_dict()}")
    print(f"  rows missing required content: {int(frame[['subject','body']].isna().any(axis=1).sum())}")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Prepared email dataset: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and validate MeAJOR, a multi-source real email phishing corpus.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/email_curated"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/phishing_emails.csv"))
    parser.add_argument("--source-file", type=Path, help="Use an already downloaded MeAJOR CSV")
    args = parser.parse_args()
    try:
        if args.source_file:
            source_path = args.source_file
        else:
            request = Request(API_URL, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=30) as response:
                record = json.load(response)
            item = next((file for file in record.get("files", []) if file["key"] == "meajor_cleaned_preprocessed.csv"), None)
            if item is None:
                raise RuntimeError(f"Zenodo record {RECORD_ID} lacks the expected CSV")
            md5_type, md5_hash = item["checksum"].split(":", 1)
            if md5_type != "md5":
                raise RuntimeError(f"Unexpected dataset checksum type: {md5_type}")
            source_path = args.output_dir / item["key"]
            download(item["links"]["self"], source_path, int(item["size"]), md5_hash)
        prepare(source_path, args.output)
    except Exception as exc:
        raise SystemExit(f"Email dataset preparation failed: {exc}\nDownload MeAJOR manually from https://zenodo.org/records/18471483 and rerun with --source-file PATH. No training data was generated.") from exc


if __name__ == "__main__":
    main()
