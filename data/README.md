# Dataset preparation

Public datasets are downloaded into `data/raw/` and are not committed as model evidence unless the training metadata names them. Observe each dataset's license and terms. Do not store private or restricted email data here.

## URL

The checked-in real URL source is [UCI PhiUSIIL dataset 967](https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bdataset), licensed CC BY 4.0. UCI's project label mapping is reversed during preparation.

```bash
python backend/prepare_url_dataset.py
python backend/train_url_model.py --source prepared --csv data/processed/url_training.csv
```

Alternatively download `PhiUSIIL_Phishing_URL_Dataset.csv` from UCI and run:

```bash
python backend/prepare_url_dataset.py --uci-csv data/raw/PhiUSIIL_Phishing_URL_Dataset.csv
```

UCI labels are `1=legitimate, 0=phishing`; this project maps them to `0=legitimate, 1=phishing`. Preparation writes `data/processed/url_training.csv` and `data/processed/url_dataset_stats.json`, preserving source, label, timestamp (when present), and registrable domain. Exact duplicate URLs are merged, conflicting labels are removed, malformed URLs are excluded, and no URL is fetched. Training derives registrable-domain groups using the bundled Public Suffix List snapshot. This UCI source has no per-row observation timestamps, so no temporal performance claim is made.

An optional user-supplied PhishTank CSV may be imported only if its current terms and feed access authorize the intended use. It must be a registered/authorized snapshot; the tool does not fetch the feed or store API credentials:

```bash
python backend/prepare_url_dataset.py --phishtank-csv data/raw/online-valid.csv --authorized-phishtank-use
```

PhishTank publishes feed fields including verified status and submission/verification times in its [developer documentation](https://phishtank.org/developer_info.php); use is subject to its current [terms](https://phishtank.org/terms.php). OpenPhish data is not included because its [terms restrict use and redistribution](https://www.openphish.com/terms.html); researchers can apply to its [academic program](https://openphish.com/academic_use.html).

The URL challenge uses `data/hard_negative_urls.csv` for the earlier 47-example development set, `data/hard_negative_urls_external.csv` for 104 external organizational URLs, and URLs mined from UCI's legitimate label by token and structure pattern (1,193 rows in this run). `data/hard_negative_urls_v5.csv` holds 142 legitimate lookalikes for v5 development; its roots are split into fit, validation, and a separate development holdout, and none overlap the fixed 21-root final challenge. The final challenge was scored only after the v5 candidate and threshold were saved. It is descriptive rather than a population sample; review its false-positive rate before using a candidate.

## Email

The downloader fetches the [MeAJOR multi-source email dataset](https://zenodo.org/records/18471483) from Zenodo (about 191 MB CSV). Its `label` mapping is `0=benign, 1=phishing`; the script validates it and retains the source column for source-disjoint splits.

The downloader exclusively uses the verified MeAJOR corpus; unreferenced legacy datasets (such as `Enron.csv`) have been removed from the repository.

```bash
python data/download_email_dataset.py
python backend/train_email_model.py --csv data/raw/phishing_emails.csv
```

The trainer also accepts CSVs with subject/body or text/message columns and binary labels. It removes empty/duplicate text and requires at least two classes across enough data sources for group-disjoint train/validation/test partitions.

## Malware

The file scanner is static-only. For a real model, provide a labeled CSV with the exact feature names in `backend/malware_features.py` and `label` (`0=safe, 1=malicious`), or authorized benign and malware sample folders. The folder trainer reads bytes and does not execute samples.

```bash
python backend/train_malware_model.py --csv data/raw/malware_features.csv
# Or:
python backend/train_malware_model.py --benign data/samples/benign --malware data/samples/malware
```

EMBER's precomputed PE feature schema differs from the current inference extractor. Do not claim an EMBER-trained detector without implementing matching inference extraction and evaluating that exact schema. The bundled malware artifact is synthetic DEMO-only.
