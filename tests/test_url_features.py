import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from domain_utils import registrable_domain
from url_features import extract_url_features, normalize_url


def test_https_feature():
    data = extract_url_features("https://example.com/login")
    assert data["is_https"] == 1


def test_ip_feature():
    data = extract_url_features("http://192.168.1.4/login")
    assert data["is_ip_host"] == 1


def test_counts():
    data = extract_url_features("http://example.com/a-b?x=1")
    assert data["num_hyphens"] == 1
    assert data["num_question"] == 1
    assert data["num_equals"] == 1


def test_unusual_port_entropy_punycode_and_query_features():
    data = extract_url_features("http://xn--e1afmkfd.example:8443/verify?account=123&next=%2Fhome#login")
    assert data["has_punycode"] == 1
    assert data["has_unusual_port"] == 1
    assert data["query_param_count"] == 2
    assert data["encoded_byte_count"] == 1
    assert data["host_char_entropy"] > 0
    assert data["path_char_entropy"] > 0


def test_registrable_domain_groups_subdomains_and_compound_suffixes():
    assert registrable_domain("accounts.google.com") == "google.com"
    assert registrable_domain("www.example.co.uk") == "example.co.uk"
    assert registrable_domain("192.0.2.1") == "192.0.2.1"


def test_url_limits_and_empty_inputs():
    for value in ("", "javascript:alert(1)", "https://example.com:bad", "https://example.com/" + "a" * 2050):
        try:
            normalize_url(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid URL rejection: {value[:30]}")
