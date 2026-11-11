# TriGuard — Quickstart Guide

A concise reference for running, testing, and training **TriGuard (Local AI Security Lab)**.

---

## 1. Quick Setup & Run (Demo Mode)

The default **DEMO mode** runs immediately without downloading external datasets.

```bash
# 1. Create and activate a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip and install backend requirements
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

# 3. Launch the API server
python backend/app.py
```

- **Web Dashboard**: Open [http://127.0.0.1:5050](http://127.0.0.1:5050)
- **API Health Check**: Open [http://127.0.0.1:5050/api/health](http://127.0.0.1:5050/api/health)

---

## 2. Train Real Machine Learning Models

### Train Real URL Model (UCI PhiUSIIL)
```bash
# Clean raw dataset into processed format
python backend/prepare_url_dataset.py

# Train candidate URL models (evaluates v5 with domain-disjoint splits)
python backend/train_url_model.py \
  --source prepared \
  --csv data/processed/url_training.csv \
  --hard-negatives data/hard_negative_urls_v5.csv \
  --candidate-version 5

# Evaluate the candidate against the frozen challenge set
python backend/evaluate_url_model_v5.py
```

### Train Real Email Model (MeAJOR)
```bash
# Download and validate MeAJOR multi-source corpus from Zenodo
python data/download_email_dataset.py

# Train calibrated TF-IDF Logistic Regression model
python backend/train_email_model.py --csv data/raw/phishing_emails.csv
```

---

## 3. Run with Real Models

Once models are trained, launch the server in **REAL mode**:

```bash
# Run with default real models (URL v6 + Email real)
APP_MODE=real python backend/app.py

# Run with an opt-in URL candidate version (v3, v4, v5, or v6)
APP_MODE=real URL_MODEL_VERSION=v5 python backend/app.py
```

> [!NOTE]
> In REAL mode, file scanning returns HTTP 503 until a real malware dataset matching `backend/malware_features.py` is trained. The system never silently falls back to demo artifacts in REAL mode.

---

## 4. Run Automated Tests

Run the full pytest suite to verify all feature extractors and API endpoints:

```bash
pytest
```

---

## 5. Install the Chrome Browser Extension

1. Ensure the Flask server is running at `http://127.0.0.1:5050`.
2. Open Chrome and navigate to `chrome://extensions`.
3. Toggle on **Developer mode** in the top right corner.
4. Click **Load unpacked** in the top left corner.
5. Select the `extension/` folder from this repository.
6. Test real-time scanning by visiting an HTTP/HTTPS site or opening an email message in Gmail.

---

## 6. Environment Variables Reference

| Variable | Default | Options | Description |
| :--- | :--- | :--- | :--- |
| `APP_MODE` | `demo` | `demo`, `real` | Controls whether synthetic demo artifacts or trained real models are loaded |
| `URL_MODEL_VERSION` | `v6` | `v3`, `v4`, `v5`, `v6` | Selects which trained URL model candidate version to serve |
| `URL_SUSPICIOUS_THRESHOLD` | *From artifact* | Float `0.0 - 1.0` | Probability threshold for SUSPICIOUS classification |
| `URL_MALICIOUS_THRESHOLD` | *From artifact* | Float `0.0 - 1.0` | Probability threshold for MALICIOUS classification |
| `ENABLE_EXTENSION_CORS` | `1` | `0`, `1` | Enables CORS origins for Chrome extensions and localhost |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` | Console logging verbosity |
