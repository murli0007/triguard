from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser


URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.I)


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
    def handle_data(self, data):
        self.parts.append(data)
    def handle_entityref(self, name):
        self.parts.append(unescape(f"&{name};"))
    def handle_charref(self, name):
        self.parts.append(unescape(f"&#{name};"))


def _strip_html(value: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(value)
        parser.close()
        return " ".join(parser.parts)
    except Exception:
        return re.sub(r"<[^>]{0,2048}>", " ", value)


def clean_email_text(subject: str = "", body: str = "") -> str:
    """Remove HTML markup while retaining wording and URL presence for TF-IDF."""
    subject = str(subject or "")
    body = str(body or "")
    combined = f"{subject}\n{body}"
    combined = unescape(combined)
    combined = _strip_html(combined)
    combined = URL_PATTERN.sub(" URL ", combined)
    combined = re.sub(r"\b[\w.+%-]+@[\w.-]+\.[A-Za-z]{2,}\b", " EMAIL ", combined)
    combined = re.sub(r"[^\w$!?@'\s-]", " ", combined, flags=re.UNICODE)
    return re.sub(r"\s+", " ", combined).strip().lower()


def email_indicators(subject: str = "", body: str = "") -> list[str]:
    text = f"{subject}\n{body}"
    low = text.lower()
    reasons = []
    patterns = [
        (r"\b(urgent|immediately|within \d+ hours?|act now|final warning|expires today)\b", "Urgent or time-pressure wording"),
        (r"\b(password|credentials?|one[- ]time (?:password|code)|otp|card number|bank details|recovery phrase)\b", "Requests credentials or sensitive payment information"),
        (r"https?://|www\.", "Contains one or more links"),
        (r"\b(verify|confirm|validate|restore|unlock) (?:your )?(?:account|identity|payment|password)\b", "Account verification or recovery wording"),
        (r"\b(ceo|administrator|security team|support team|microsoft|apple|paypal|amazon)\b", "Uses an organization or authority identity"),
    ]
    for pattern, description in patterns:
        if re.search(pattern, low, re.I):
            reasons.append(description)
    if text.count("!") >= 3:
        reasons.append("Excessive exclamation marks")
    return reasons
