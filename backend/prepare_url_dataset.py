from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from domain_utils import registrable_domain
from url_features import normalize_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UCI = PROJECT_ROOT / "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
DEFAULT_HARD_NEGATIVES = PROJECT_ROOT / "data/hard_negative_urls.csv"
DEFAULT_EXTERNAL_HARD_NEGATIVES = PROJECT_ROOT / "data/hard_negative_urls_external.csv"
DEFAULT_OUT = PROJECT_ROOT / "data/processed/url_training.csv"
DEFAULT_PREPARED_HARD_NEGATIVES = PROJECT_ROOT / "data/processed/url_hard_negative_urls.csv"


def _column_map(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column).strip().lower(): column for column in frame.columns}


def _timestamp(value: object) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        return ""
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return "" if pd.isna(parsed) else parsed.isoformat()


def _load_uci(path: Path) -> tuple[list[dict], dict]:
    if not path.is_file():
        raise FileNotFoundError(f"UCI dataset not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    cols = _column_map(frame)
    url_col, label_col = cols.get("url"), cols.get("label")
    if not url_col or not label_col:
        raise ValueError("PhiUSIIL CSV must contain URL and label columns")
    records, invalid = [], 0
    for raw_url, raw_label in zip(frame[url_col], frame[label_col]):
        try:
            url = normalize_url(str(raw_url))
        except (ValueError, TypeError):
            invalid += 1
            continue
        try:
            label = int(raw_label)
        except (ValueError, TypeError):
            invalid += 1
            continue
        if label not in (0, 1):
            invalid += 1
            continue
        host = urlsplit(url).hostname or ""
        records.append({
            "url": url,
            "label": 1 - label,  # UCI: 1 legitimate, 0 phishing; project: 0/1.
            "domain": registrable_domain(host),
            "source": "UCI PhiUSIIL 967",
            "observed_at": "",
            "source_label": str(raw_label),
            "source_license": "CC BY 4.0; attribution: Prasad and Chandra, UCI PhiUSIIL Phishing URL (Website), 2024",
        })
    return records, {
        "source": "UCI PhiUSIIL Phishing URL (Website), dataset 967",
        "path": str(path),
        "rows_read": len(frame),
        "rows_accepted": len(records),
        "rows_rejected": invalid,
        "source_label_definition": "1=legitimate, 0=phishing",
        "project_label_definition": "0=legitimate, 1=phishing",
        "license": "CC BY 4.0",
        "donated_to_UCI": "2024-03-03",
        "observation_timestamps_available": False,
    }


def _load_phishtank(path: Path) -> tuple[list[dict], dict]:
    frame = pd.read_csv(path, low_memory=False)
    cols = _column_map(frame)
    url_col = cols.get("url")
    if not url_col:
        raise ValueError("PhishTank CSV must contain a url column")
    verified_col = cols.get("verified")
    if verified_col:
        frame = frame[frame[verified_col].astype(str).str.lower().eq("yes")]
    timestamp_col = cols.get("verification_time") or cols.get("submission_time")
    records, invalid = [], 0
    for _, row in frame.iterrows():
        try:
            url = normalize_url(str(row[url_col]))
        except (ValueError, TypeError):
            invalid += 1
            continue
        host = urlsplit(url).hostname or ""
        records.append({
            "url": url,
            "label": 1,
            "domain": registrable_domain(host),
            "source": "PhishTank verified feed snapshot",
            "observed_at": _timestamp(row[timestamp_col]) if timestamp_col else "",
            "source_label": "verified=yes",
            "source_license": "Use only under the user’s current PhishTank/Cisco terms and registered feed access",
        })
    return records, {
        "source": "User-supplied PhishTank verified feed snapshot",
        "path": str(path),
        "rows_read_after_verification_filter": len(frame),
        "rows_accepted": len(records),
        "rows_rejected": invalid,
        "source_label_definition": "verified online phishing URL; project label 1=phishing",
        "timestamp_column": str(timestamp_col) if timestamp_col else None,
        "license": "Use subject to current PhishTank/Cisco terms; feed access requires registration/app key",
        "observation_timestamps_available": bool(timestamp_col and any(row["observed_at"] for row in records)),
    }


def _deduplicate(records: list[dict]) -> tuple[pd.DataFrame, dict]:
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No usable URL records were supplied")
    conflicts = frame.groupby("url", sort=False).label.nunique()
    conflicting_urls = set(conflicts[conflicts > 1].index)
    conflict_count = len(conflicting_urls)
    frame = frame[~frame.url.isin(conflicting_urls)].copy()
    duplicate_count = int(frame.duplicated("url").sum())
    # Preserve provenance for same-label repeats while retaining one model row.
    frame = frame.groupby("url", sort=False, as_index=False).agg({
        "label": "first", "domain": "first",
        "source": lambda values: "; ".join(sorted(set(map(str, values)))),
        "observed_at": lambda values: min((str(v) for v in values if str(v)), default=""),
        "source_label": lambda values: "; ".join(sorted(set(map(str, values)))),
        "source_license": lambda values: "; ".join(sorted(set(map(str, values)))),
    })
    return frame, {"conflicting_url_labels_removed": conflict_count, "same_label_duplicate_rows_merged": duplicate_count}


def _hard_negative_report(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "rows": 0, "status": "not_found"}
    frame = pd.read_csv(path)
    required = {"url", "expected_label", "category"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Hard-negative CSV requires columns: {sorted(required)}")
    invalid = 0
    accepted = []
    for url, label, category in zip(frame.url, frame.expected_label, frame.category):
        try:
            url = normalize_url(str(url))
            label = int(label)
            if label not in (0, 1):
                raise ValueError
        except (ValueError, TypeError):
            invalid += 1
            continue
        accepted.append({"url": url, "label": label, "category": str(category)})
    return {
        "path": str(path), "rows_read": len(frame), "rows_accepted": len(accepted),
        "rows_rejected": invalid,
        "class_counts": {str(label): int(sum(row["label"] == label for row in accepted)) for label in (0, 1)},
        "category_counts": dict(Counter(row["category"] for row in accepted)),
        "role": "separate hard-negative source; trainer partitions by registrable domain, uses fit rows only for training augmentation, and never uses held-out rows for model/threshold selection",
    }


def _prepare_hard_negative_dataset(manual_path: Path, uci_records: list[dict], out_path: Path,
                                  external_path: Path = DEFAULT_EXTERNAL_HARD_NEGATIVES) -> dict:
    manual = pd.read_csv(manual_path)
    required = {"url", "category", "expected_label"}
    if not required.issubset(manual.columns):
        raise ValueError(f"Hard-negative CSV requires columns: {sorted(required)}")
    records = []
    token_pattern = re.compile(r"(?i)(login|log-in|signin|sign-in|verify|verification|secure|security|account|authentication|auth|payment|billing|password|support|invoice|banking|identity|portal|sso|checkout|recover|reset|credential|confirm|profile|service|user|customer|member|subscription)")
    per_domain: dict[str, int] = {}
    mined_count = 0
    for item in uci_records:
        url = item["url"]
        if item["label"] != 0:
            continue
        parsed = urlsplit(url)
        matches = sorted(set(token.lower() for token in token_pattern.findall(url)))
        query_count = len([part for part in parsed.query.split("&") if part])
        long_query = len(parsed.query) >= 40 and query_count >= 1
        multi_query = query_count >= 2
        deep_path = len([part for part in parsed.path.split("/") if part]) >= 4
        long_url = len(url) >= 110
        if not matches and not (long_query or multi_query or deep_path or long_url):
            continue
        domain = item["domain"]
        if per_domain.get(domain, 0) >= 5:
            continue
        per_domain[domain] = per_domain.get(domain, 0) + 1
        if matches:
            category = "uci_legitimate_suspicious_terms"
        elif long_query or multi_query:
            category = "uci_legitimate_query_heavy"
        elif deep_path:
            category = "uci_legitimate_deep_path"
        else:
            category = "uci_legitimate_long_url"
        records.append({
            "url": url, "category": category, "expected_label": 0,
            "provenance": f"UCI PhiUSIIL label=legitimate; mined by suspicious URL token/structure rule; terms={','.join(matches) or 'none'}; query_count={query_count}; path_depth={len([part for part in parsed.path.split('/') if part])}; URL_length={len(url)}",
            "review_date": "2026-09-27", "source": "UCI PhiUSIIL 967",
        })
        mined_count += 1
    for row in manual.to_dict("records"):
        try:
            url = normalize_url(str(row["url"]))
            label = int(row["expected_label"])
        except (ValueError, TypeError):
            continue
        records.append({
            "url": url, "category": str(row["category"]), "expected_label": label,
            "provenance": str(row.get("provenance", "manually curated official URL")),
            "review_date": str(row.get("review_date", "2026-09-27")), "source": "manual curated challenge",
        })
    external_count = 0
    if external_path.is_file():
        external = pd.read_csv(external_path)
        if not required.issubset(external.columns):
            raise ValueError(f"External challenge CSV requires columns: {sorted(required)}")
        for row in external.to_dict("records"):
            try:
                url = normalize_url(str(row["url"]))
                label = int(row["expected_label"])
            except (ValueError, TypeError):
                continue
            if label not in (0, 1):
                continue
            records.append({
                "url": url, "category": str(row["category"]), "expected_label": label,
                "provenance": str(row.get("provenance", "official external organization URL")),
                "review_date": str(row.get("review_date", "2026-09-27")), "source": "external official URL challenge",
            })
            external_count += 1
    frame = pd.DataFrame(records).drop_duplicates("url", keep="first").reset_index(drop=True)
    frame["domain"] = [registrable_domain(urlsplit(url).hostname or "") for url in frame.url]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return {
        "path": str(out_path), "manual_rows": int(len(manual)), "external_official_rows": external_count,
        "mined_UCI_legitimate_rows": mined_count,
        "total_unique_rows": int(len(frame)), "unique_registrable_domains": int(frame.domain.nunique()),
        "mining_rule": "UCI label=legitimate plus login/signin/verify/security/secure/account/auth/payment/billing/password/support/invoice/banking/identity/portal/sso/checkout/recover/reset/credential/confirm/profile/service/user/customer/member/subscription URL substring OR query length >=40 with >=1 parameter OR >=2 query parameters OR path depth >=4 OR URL length >=110; maximum 5 URLs per registrable domain",
        "mined_license": "CC BY 4.0 inherited from UCI; attribution required",
        "role": "partitioned by registrable domain; fit subset may augment training, held-out subset reports challenge performance; no threshold or model tuning on held-out subset",
    }


def prepare(uci_path: Path, out_path: Path, phishtank_path: Path | None,
            hard_negatives_path: Path,
            external_hard_negatives_path: Path = DEFAULT_EXTERNAL_HARD_NEGATIVES) -> tuple[pd.DataFrame, dict]:
    records, sources = _load_uci(uci_path)
    prepared_hard_negatives = out_path.with_name("url_hard_negative_urls.csv")
    hard_negative_stats = _prepare_hard_negative_dataset(
        hard_negatives_path, records, prepared_hard_negatives, external_hard_negatives_path,
    )
    source_reports = [sources]
    if phishtank_path:
        imported, report = _load_phishtank(phishtank_path)
        records.extend(imported)
        source_reports.append(report)
    prepared, dedupe = _deduplicate(records)
    stats = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sources": source_reports,
        "preprocessing": [
            "Normalize missing schemes as HTTPS and validate HTTP(S) syntax; never fetch URLs.",
            "Map labels to 0=legitimate, 1=phishing.",
            "Derive registrable domains with the bundled Public Suffix List snapshot for split groups.",
            "Remove malformed/over-2048-character URLs.",
            "Remove URLs with conflicting labels; merge exact same-label duplicates and preserve source provenance.",
            "Keep hard negatives in a separate CSV; the trainer assigns registrable domains to fit augmentation or held-out challenge evaluation.",
        ],
        "deduplication": dedupe,
        "rows_after_cleaning": len(prepared),
        "class_counts": {"legitimate": int((prepared.label == 0).sum()), "phishing": int((prepared.label == 1).sum())},
        "unique_registrable_domains": int(prepared.domain.nunique()),
        "timestamp_rows": int(prepared.observed_at.ne("").sum()),
        "source_counts": {str(key): int(value) for key, value in prepared.source.value_counts().items()},
        "hard_negative_challenge": hard_negative_stats,
        "temporal_evaluation_available": bool(prepared.observed_at.ne("").all()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(out_path, index=False)
    stats_path = out_path.with_name("url_dataset_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return prepared, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a provenance-preserving URL training CSV without fetching submitted URLs.")
    parser.add_argument("--uci-csv", type=Path, default=DEFAULT_UCI)
    parser.add_argument("--phishtank-csv", type=Path, help="Optional authorized PhishTank feed snapshot CSV.")
    parser.add_argument("--authorized-phishtank-use", action="store_true", help="Acknowledge that you have checked current PhishTank/Cisco terms and may use this snapshot for training.")
    parser.add_argument("--hard-negatives", type=Path, default=DEFAULT_HARD_NEGATIVES)
    parser.add_argument("--external-hard-negatives", type=Path, default=DEFAULT_EXTERNAL_HARD_NEGATIVES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.phishtank_csv and not args.authorized_phishtank_use:
        parser.error("--phishtank-csv requires --authorized-phishtank-use after checking current feed terms")
    prepared, stats = prepare(args.uci_csv, args.out, args.phishtank_csv, args.hard_negatives, args.external_hard_negatives)
    print(json.dumps({"output": str(args.out), "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
