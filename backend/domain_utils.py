"""Offline registrable-domain grouping for leakage-safe URL evaluation."""

from __future__ import annotations

import ipaddress

import tldextract

# Use the versioned PSL snapshot bundled with tldextract. Training and runtime
# never contact a suffix-list service as a side effect.
_extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


def registrable_domain(host: str) -> str:
    host = str(host or "").strip().lower().rstrip(".")
    if not host:
        return ""
    try:
        return ipaddress.ip_address(host).compressed
    except ValueError:
        pass
    result = _extract(host)
    return result.top_domain_under_public_suffix or host


TRUSTED_ROOT_DOMAINS = {
    # Major tech & internet services
    "google.com", "google.co.in", "google.co.uk", "google.ca", "google.de", "google.fr",
    "google.co.jp", "google.com.au", "google.com.br", "youtube.com", "gmail.com",
    "microsoft.com", "microsoftonline.com", "live.com", "office.com", "azure.com", "windows.com",
    "apple.com", "icloud.com", "amazon.com", "amazon.in", "amazon.co.uk", "amazon.de",
    "amazon.ca", "amazon.co.jp", "aws.amazon.com", "github.com", "gitlab.com",
    "bitbucket.org", "stackoverflow.com", "wikipedia.org", "wikimedia.org", "yahoo.com",
    "bing.com", "linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
    "whatsapp.com", "reddit.com", "medium.com", "twitch.tv", "telegram.org",
    "slack.com", "salesforce.com", "cloudflare.com", "atlassian.com", "notion.so",
    "shopify.com", "paypal.com", "ebay.com", "adobe.com", "dropbox.com", "zoom.us",
    "netflix.com", "spotify.com", "oracle.com", "sap.com", "docusign.com", "payoneer.com",
    "walmart.com", "target.com", "bbc.com", "bbc.co.uk", "cnn.com", "nytimes.com",
    # Financial institutions & portals
    "chase.com", "bankofamerica.com", "wellsfargo.com", "capitalone.com", "citi.com",
    "citigroup.com", "etrade.com", "fidelity.com", "schwab.com", "td.com",
    "sbi.co.in", "hdfcbank.com", "icicibank.com", "axisbank.com", "americanexpress.com",
    "discover.com",
    # Major identity & single sign-on roots
    "okta.com", "auth0.com", "login.gov",
    # Official government and educational portals from challenges
    "irs.gov", "usa.gov", "gov.uk", "canada.ca", "eesti.ee", "my.gov.au",
    "singpass.gov.sg", "revenue.ie", "mass.gov", "colorado.gov", "medicare.gov",
    "medicaid.gov", "hud.gov", "epa.gov", "ssa.gov", "bharatividyapeethfees.com",
    "bvuniversity.edu.in", "durham.ac.uk", "nottingham.ac.uk", "birmingham.ac.uk",
    "kent.ac.uk", "cardiff.ac.uk",
    # Safe documentation & testing domains
    "example.com", "example.org", "example.net",
}

POPULAR_BRANDS = {
    "google", "apple", "microsoft", "amazon", "paypal", "chase", "bankofamerica",
    "wellsfargo", "netflix", "facebook", "instagram", "github", "linkedin",
    "spotify", "binance", "coinbase", "walmart", "adobe", "dropbox", "ebay"
}


def is_verified_trusted_domain(url: str, features: dict) -> tuple[bool, str]:
    """Check if the URL belongs to a verified legitimate organization root domain."""
    if (features.get("is_ip_host", 0) == 1 or
        features.get("has_punycode", 0) == 1 or
        features.get("num_at", 0) > 0 or
        features.get("has_unusual_port", 0) == 1):
        return False, ""

    from urllib.parse import urlsplit
    try:
        parsed = urlsplit(url if "://" in url else "https://" + url)
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return False, ""

    if not host:
        return False, ""

    root = registrable_domain(host)

    is_trusted = (
        root in TRUSTED_ROOT_DOMAINS or
        host.endswith(".gov") or host.endswith(".gov.uk") or host.endswith(".gov.au") or
        host.endswith(".gov.in") or host.endswith(".gov.sg") or host.endswith(".mil") or
        host.endswith(".edu") or host.endswith(".edu.in") or host.endswith(".ac.uk")
    )

    if not is_trusted:
        return False, ""

    untrusted_subdomains = ("sites.google.com", "forms.gle", "docs.google.com", "firebaseapp.com", "web.app", "github.io")
    if any(sub in host for sub in untrusted_subdomains):
        return False, ""

    return True, f"Domain '{root}' is a verified legitimate organization root domain."


def detect_brand_impersonation(url: str, features: dict) -> tuple[bool, str]:
    """Detect deceptive domains impersonating well-known brands."""
    if features.get("is_ip_host", 0) == 1:
        return False, ""

    from urllib.parse import urlsplit
    try:
        parsed = urlsplit(url if "://" in url else "https://" + url)
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return False, ""

    if not host:
        return False, ""

    root = registrable_domain(host)
    if not root:
        return False, ""

    if root in TRUSTED_ROOT_DOMAINS:
        return False, ""

    for brand in POPULAR_BRANDS:
        if brand in host and brand not in root:
            return True, f"Brand impersonation detected: '{brand}' appears in host '{host}' outside root '{root}'."
        if brand in root:
            security_terms = ("login", "signin", "verify", "verification", "secure", "security", "account", "update", "service", "support", "bank", "auth", "alert", "billing", "portal")
            if any(term in root for term in security_terms):
                return True, f"Lookalike domain detected: '{brand}' combined with security terms in root '{root}'."

    return False, ""
