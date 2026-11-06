from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import parse_qsl, urlsplit, urlunsplit

FEATURE_NAMES = [
    "url_length", "domain_length", "path_length", "query_length",
    "num_dots", "num_hyphens", "num_underscores", "num_slashes",
    "num_question", "num_equals", "num_at", "num_ampersands", "num_percent",
    "num_digits", "num_letters", "num_subdomains", "is_https", "is_ip_host",
    "has_punycode", "has_port", "has_fragment", "suspicious_word_count",
    "digit_ratio", "letter_ratio", "special_ratio", "host_label_count",
    "host_max_label_length", "host_digit_count", "host_hyphen_count",
    "path_segment_count", "query_param_count", "encoded_byte_count",
    "suspicious_host_word_count", "suspicious_path_word_count", "url_char_entropy",
    "host_char_entropy", "path_char_entropy", "query_char_entropy", "has_unusual_port",
]
SUSPICIOUS_WORDS = {
    "login", "signin", "verify", "verification", "secure", "account", "update",
    "confirm", "bank", "password", "wallet", "payment", "invoice", "free",
    "bonus", "urgent", "alert", "credential", "recover", "suspend", "unlock",
}
_WORD_RE = re.compile(r"[a-z0-9]+", re.I)


def normalize_url(value: str) -> str:
    """Normalize scheme-less HTTP(S) input; never fetch or resolve it."""
    value = value.strip()
    if not value:
        raise ValueError("url is required")
    if len(value) > 2048:
        raise ValueError("url exceeds the 2048 character limit")
    if any(ord(char) < 32 for char in value):
        raise ValueError("url contains control characters")
    candidate = value if "://" in value else "https://" + value
    try:
        parsed = urlsplit(candidate)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("url is malformed") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("url must be a valid HTTP or HTTPS URL")
    if any(char.isspace() for char in parsed.hostname):
        raise ValueError("url hostname is malformed")
    return candidate


def canonicalize_root_path_for_model(value: str) -> str:
    """Treat an empty HTTP(S) path and '/' as the same model input.

    Browsers serialize a bare host as a URL with a trailing slash. Keeping that
    serialization detail caused the current numeric URL model to treat the
    third slash in ``https://host/`` as strong phishing evidence.
    """
    normalized = normalize_url(value)
    parsed = urlsplit(normalized)
    if parsed.path != "/":
        return normalized
    return urlunsplit((parsed.scheme, parsed.netloc, "", parsed.query, parsed.fragment))


def _words(value: str) -> set[str]:
    return set(_WORD_RE.findall(value.lower()))


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    size = len(value)
    return -sum((count / size) * math.log2(count / size) for count in counts.values())


def extract_url_features(url: str) -> dict[str, float]:
    """Ordered URL-only lexical/static features shared by training and inference."""
    raw = normalize_url(url)
    parsed = urlsplit(raw)
    host = (parsed.hostname or "").lower().rstrip(".")
    labels = [part for part in host.split(".") if part]
    path = parsed.path or ""
    query = parsed.query or ""
    try:
        ip_host = int(ipaddress.ip_address(host) is not None)
    except ValueError:
        ip_host = 0
    port = parsed.port
    all_words = _words(raw)
    host_words = _words(host)
    path_words = _words(path)
    digit_count = sum(char.isdigit() for char in raw)
    letter_count = sum(char.isalpha() for char in raw)
    total = max(1, len(raw))
    features = {
        "url_length": len(raw),
        "domain_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": raw.count("."),
        "num_hyphens": raw.count("-"),
        "num_underscores": raw.count("_"),
        "num_slashes": raw.count("/"),
        "num_question": raw.count("?"),
        "num_equals": raw.count("="),
        "num_at": raw.count("@"),
        "num_ampersands": raw.count("&"),
        "num_percent": raw.count("%"),
        "num_digits": digit_count,
        "num_letters": letter_count,
        "num_subdomains": max(0, len(labels) - 2),
        "is_https": int(parsed.scheme.lower() == "https"),
        "is_ip_host": ip_host,
        "has_punycode": int("xn--" in host),
        "has_port": int(parsed.port is not None),
        "has_fragment": int(bool(parsed.fragment)),
        "suspicious_word_count": len(all_words & SUSPICIOUS_WORDS),
        "digit_ratio": digit_count / total,
        "letter_ratio": letter_count / total,
        "special_ratio": (total - digit_count - letter_count) / total,
        "host_label_count": len(labels),
        "host_max_label_length": max(map(len, labels), default=0),
        "host_digit_count": sum(char.isdigit() for char in host),
        "host_hyphen_count": host.count("-"),
        "path_segment_count": sum(bool(segment) for segment in path.split("/")),
        "query_param_count": len(parse_qsl(query, keep_blank_values=True)),
        "encoded_byte_count": len(re.findall(r"%[0-9a-f]{2}", raw, re.I)),
        "suspicious_host_word_count": len(host_words & SUSPICIOUS_WORDS),
        "suspicious_path_word_count": len((path_words | _words(query)) & SUSPICIOUS_WORDS),
        "url_char_entropy": _entropy(raw),
        "host_char_entropy": _entropy(host),
        "path_char_entropy": _entropy(path),
        "query_char_entropy": _entropy(query),
        "has_unusual_port": int(port is not None and port not in {80, 443}),
    }
    return {name: float(features[name]) for name in FEATURE_NAMES}


def url_indicators(features: dict[str, float], url: str) -> list[str]:
    """Describe observed features only; this function never derives a verdict."""
    parsed = urlsplit(normalize_url(url))
    indicators = []
    if features["is_ip_host"]:
        indicators.append("Hostname is an IP address.")
    if features["has_punycode"]:
        indicators.append("Hostname contains punycode encoding.")
    if features["num_at"]:
        indicators.append("URL contains an @ character in its authority or path.")
    if features["num_subdomains"] >= 3:
        indicators.append(f"Hostname has {int(features['num_subdomains'])} subdomain labels.")
    terms = sorted(_words(url) & SUSPICIOUS_WORDS)
    if terms:
        indicators.append("URL contains security/account terms: " + ", ".join(terms) + ".")
    if parsed.scheme.lower() == "http":
        indicators.append("URL uses HTTP without transport encryption.")
    return indicators


def url_signal_count(features: dict[str, float]) -> int:
    """Count independent structural risk signals used only to guard MALICIOUS severity."""
    checks = (
        features["is_ip_host"] == 1,
        features["has_punycode"] == 1,
        features["num_at"] > 0,
        features["is_https"] == 0,
        features["num_subdomains"] >= 3,
        features["suspicious_word_count"] >= 2,
        features["url_length"] >= 120,
        features["host_digit_count"] >= 3,
        features["num_percent"] >= 3,
    )
    return sum(checks)
