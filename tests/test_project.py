from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import app as api
from email_features import clean_email_text, email_indicators
from malware_features import extract_malware_features
from risk_engine import combine_scores, label_from_score, url_risk_decision
from url_features import (
    FEATURE_NAMES,
    extract_url_features,
    normalize_url,
    url_signal_count,
)


@pytest.fixture
def client():
    api.MODE = "demo"
    api._artifacts.clear()
    return api.app.test_client()


def test_url_feature_schema_and_normalization():
    features = extract_url_features("example.com/login?a=1")
    assert list(features) == FEATURE_NAMES
    assert features["is_https"] == 1 and features["num_question"] == 1
    assert normalize_url("http://192.0.2.1/a").startswith("http://")


def test_url_severity_guard_keeps_unsupported_high_score_at_suspicious():
    features = extract_url_features("https://www.bvuniversity.edu.in/fee-structure")
    assert url_signal_count(features) < 2
    assert url_risk_decision(
        0.9999,
        {"suspicious_probability": 0.5, "malicious_probability": 0.9},
        url_signal_count(features),
    ) == ("suspicious", 69.0)

    malicious = extract_url_features("http://192.0.2.7/login/secure/account/verify")
    assert (
        url_risk_decision(
            0.9999,
            {"suspicious_probability": 0.5, "malicious_probability": 0.9},
            url_signal_count(malicious),
        )[0]
        == "malicious"
    )


def test_url_threshold_keeps_moderate_lexical_score_in_safe_band():
    assert url_risk_decision(
        0.65,
        {"suspicious_probability": 0.70, "malicious_probability": 0.90},
        0,
    ) == ("safe", 29.0)
    assert url_risk_decision(
        0.85,
        {"suspicious_probability": 0.70, "malicious_probability": 0.90},
        3,
    ) == ("suspicious", 69.0)


def test_v3_url_candidate_is_opt_in(monkeypatch):
    monkeypatch.setattr(api, "MODE", "real")
    monkeypatch.setenv("URL_MODEL_VERSION", "v3")
    assert api.model_path("url").name == "url_model_v3.joblib"


def test_v4_url_candidate_is_opt_in(monkeypatch):
    monkeypatch.setattr(api, "MODE", "real")
    monkeypatch.setenv("URL_MODEL_VERSION", "v4")
    assert api.model_path("url").name == "url_model_v4.joblib"


def test_v5_url_candidate_is_opt_in(monkeypatch):
    monkeypatch.setattr(api, "MODE", "real")
    monkeypatch.setenv("URL_MODEL_VERSION", "v5")
    assert api.model_path("url").name == "url_model_v5.joblib"


def test_v3_url_model_loads_through_existing_api(client, monkeypatch):
    monkeypatch.setattr(api, "MODE", "real")
    monkeypatch.setenv("URL_MODEL_VERSION", "v3")
    api._artifacts.clear()
    response = client.post("/api/check-url", json={"url": "https://google.com/"})
    assert response.status_code == 200
    data = response.get_json()
    assert {
        "prediction",
        "model_probability",
        "model_scores",
        "decision_thresholds",
        "indicators",
        "model",
        "model_source",
        "model_version",
    }.issubset(data)


def test_v4_url_model_loads_through_existing_api(client, monkeypatch):
    monkeypatch.setattr(api, "MODE", "real")
    monkeypatch.setenv("URL_MODEL_VERSION", "v4")
    api._artifacts.clear()
    response = client.post("/api/check-url", json={"url": "https://login.salesforce.com/"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["model_version"] == "4.0.0-candidate"
    assert data["training_source"] == "real"


def test_v5_url_model_loads_through_existing_api(client, monkeypatch):
    monkeypatch.setattr(api, "MODE", "real")
    monkeypatch.setenv("URL_MODEL_VERSION", "v5")
    api._artifacts.clear()
    response = client.post("/api/check-url", json={"url": "https://login.salesforce.com/"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["model_version"] == "5.0.0-candidate"
    assert data["training_source"] == "real"


def test_curated_hard_negative_urls_have_provenance_and_benign_labels():
    import pandas as pd

    frame = pd.read_csv(ROOT / "data/hard_negative_urls.csv")
    assert len(frame) >= 40
    assert set(frame["expected_label"]) == {0}
    assert frame["category"].notna().all() and frame["provenance"].notna().all()
    assert any(frame["url"].str.contains("google.com", regex=False))


def test_v5_hard_negative_development_domains_are_separate_from_final_challenge():
    import pandas as pd
    from urllib.parse import urlsplit
    from domain_utils import registrable_domain

    frame = pd.read_csv(ROOT / "data/hard_negative_urls_v5.csv")
    challenge = json.loads(
        (ROOT / "backend/models/url_metrics_v4.json").read_text(encoding="utf-8")
    )["hard_negative_results"]["results"]
    development = {
        registrable_domain(urlsplit(str(url)).hostname or "")
        for url in frame["url"]
    }
    held_out = {
        registrable_domain(urlsplit(row["url"]).hostname or "")
        for row in challenge
    }
    assert len(frame) >= 100 and set(frame["expected_label"]) == {0}
    assert development.isdisjoint(held_out)


def test_bad_urls_rejected():
    with pytest.raises(ValueError):
        normalize_url("javascript:alert(1)")
    with pytest.raises(ValueError):
        normalize_url("https://example.com:bad")


def test_real_time_trusted_domain_accuracy(client):
    urls = [
        "https://github.com/login",
        "https://accounts.google.com/signin",
        "https://appleid.apple.com/sign-in",
        "https://login.microsoftonline.com/",
        "https://www.paypal.com/signin",
    ]
    for url in urls:
        response = client.post("/api/check-url", json={"url": url})
        assert response.status_code == 200
        data = response.get_json()
        assert data["prediction"] == "safe"
        assert data["risk_score"] <= 20.0


def test_brand_impersonation_detection(client):
    phish_urls = [
        "http://paypal.com.attacker-domain.xyz/login",
        "https://chase-bank-verify-account-update.net/signin",
        "http://appleid.apple.com.login-verify.top/auth",
    ]
    for url in phish_urls:
        response = client.post("/api/check-url", json={"url": url})
        assert response.status_code == 200
        data = response.get_json()
        assert data["prediction"] == "malicious"
        assert data["risk_score"] >= 70.0


def test_email_html_cleaning_and_indicators():
    text = clean_email_text(
        "Hello", "<b>Verify your password now</b> https://example.test"
    )
    assert "verify your password" in text and "url" in text and "<b>" not in text
    assert "Contains one or more links" in email_indicators("", "https://example.test")


def test_malware_static_features(tmp_path):
    sample = tmp_path / "sample.exe"
    sample.write_bytes(b"MZ" + b"hello powershell" * 10)
    features = extract_malware_features(sample)
    assert features["pe_magic"] == 1 and features["suspicious_token_count"] >= 1


def test_risk_engine_thresholds_and_combination():
    assert [label_from_score(x) for x in (10, 45, 90)] == [
        "safe",
        "suspicious",
        "malicious",
    ]
    assert combine_scores([50]) == 50
    assert combine_scores([100, 50]) == 90


def test_health_model_metadata(client):
    response = client.get("/api/health")
    body = response.get_json()
    assert response.status_code == 200 and body["port"] == 5050 and body["mode"] == "demo"
    assert all(body["models"][name]["available"] for name in ("url", "email", "malware"))


def test_url_inference_named_features(client):
    response = client.post("/api/check-url", json={"url": "http://192.0.2.8/verify?x=5"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["prediction"] in {"safe", "suspicious", "malicious"}
    assert 0 <= data["risk_score"] <= 100
    assert data["reasons"]
    assert data["model_scores"]["url"] >= data["risk_score"]
    assert {
        "model_probability",
        "model_scores",
        "decision_thresholds",
        "indicators",
        "model",
        "model_source",
        "model_version",
    }.issubset(data)


def test_api_validation(client):
    assert client.post("/api/check-url", json={}).status_code == 400
    assert client.post("/api/check-url", json={"url": "javascript:alert(1)"}).status_code == 400
    assert client.post("/api/check-email", json={"subject": " ", "body": " "}).status_code == 400
    assert client.post("/api/check-email", json={"subject": 123, "body": "abc"}).status_code == 400


def test_email_inference(client):
    response = client.post(
        "/api/check-email",
        json={
            "subject": "URGENT account suspended",
            "body": "Verify your password now at https://example.test",
        },
    )
    data = response.get_json()
    assert response.status_code == 200 and data["reasons"] and data["model"]
    assert "email" in data["model_scores"] and "embedded_urls" in data["model_scores"]
    assert "password" not in json.dumps(data).lower() or "credentials" in json.dumps(data).lower()


def test_file_validation_and_inference(client):
    unsupported = client.post(
        "/api/check-file",
        data={"file": (io.BytesIO(b"x"), "bad.py")},
        content_type="multipart/form-data",
    )
    assert unsupported.status_code == 415

    response = client.post(
        "/api/check-file",
        data={"file": (io.BytesIO(b"plain benign text sample"), "sample.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200 and response.get_json()["filename"] == "sample.txt"


def test_oversized_file_rejected(client):
    too_big = b"x" * (10 * 1024 * 1024 + 1)
    response = client.post(
        "/api/check-file",
        data={"file": (io.BytesIO(too_big), "huge.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 413


def test_frontend_assets_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/style.css").status_code == 200
    assert client.get("/script.js").status_code == 200


def test_extension_targets_port_5050():
    manifest = json.loads((ROOT / "extension/manifest.json").read_text(encoding="utf-8"))
    background = (ROOT / "extension/background.js").read_text(encoding="utf-8")
    assert "webNavigation" in manifest["permissions"]
    assert "http://127.0.0.1:5050" in background
    assert "AUTO_CHECK_EMAIL" in background
