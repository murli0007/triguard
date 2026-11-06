# v3 hard-negative false-positive analysis

Analyzed all 18 legitimate challenge URLs that v3 flagged at its selected 0.70 threshold.

## Pattern counts

Counts are overlapping properties, not separate explanations; the model uses only character TF-IDF n-grams plus a logistic classifier. It does not receive the numeric feature columns listed below as separate inputs.

| Pattern | Flagged URLs |
|---|---:|
| security/account terms | 15/18 |
| URL length >= 100 | 1/18 |
| query parameters present | 4/18 |
| one or more subdomain labels | 15/18 |
| hyphen present | 6/18 |
| HTTPS | 18/18 |
| URLs on legitimate organization/service roots | 18/18 |

TLDs: {'com': 15, 'in': 2, 'gov': 1}. These 18 URLs span 11 distinct registrable roots. URL length range/median: 18/39.5/107 characters.

## Per-URL evidence

The positive n-grams below are the largest direct contributions to the raw logistic score before isotonic calibration. Negative n-grams push in the opposite direction. The isotonic calibrator maps all 18 out-of-domain scores to probability 1.0, so the displayed probability is saturated and does not distinguish their raw scores.

| URL | Root | Length | Queries | Subdomains | Hyphens | TLD | Terms | Top positive character n-grams (contribution) |
|---|---|---:|---:|---:|---:|---|---|---|
| `https://accounts.google.com/signin/v2/identifier?service=mail&flowName=GlifWebSignIn&flowEntry=ServiceLogin` | `google.com` | 107 | 3 | 1 | 0 | `com` | account, login, service, signin | `m/` (+0.296), `om/` (+0.291), `.com/` (+0.290), `com/` (+0.290), `n/` (+0.129) |
| `https://support.google.com/accounts/answer/32040?hl=en` | `google.com` | 54 | 1 | 1 | 0 | `com` | account, support | `m/` (+0.494), `om/` (+0.485), `.com/` (+0.485), `com/` (+0.485), `/a` (+0.360) |
| `https://www.amazon.in/gp/help/customer/display.html?nodeId=201910460` | `amazon.in` | 68 | 1 | 1 | 0 | `in` | none | `p/` (+0.516), `r/` (+0.209), `n/` (+0.185), `/d` (+0.179), `/c` (+0.170) |
| `https://google.com/` | `google.com` | 19 | 0 | 0 | 0 | `com` | none | `m/` (+1.018), `om/` (+1.000), `.com/` (+0.999), `com/` (+0.999), `/g` (+0.250) |
| `https://myaccount.google.com/security` | `google.com` | 37 | 0 | 1 | 0 | `com` | account, security | `m/` (+0.569), `om/` (+0.559), `.com/` (+0.559), `com/` (+0.559), `/m` (+0.229) |
| `https://github.com/login?return_to=%2Fsettings` | `github.com` | 46 | 1 | 0 | 0 | `com` | login | `m/` (+0.534), `om/` (+0.524), `.com/` (+0.524), `com/` (+0.524), `/g` (+0.131) |
| `https://docs.github.com/en/authentication` | `github.com` | 41 | 0 | 1 | 0 | `com` | authentication | `m/` (+0.567), `om/` (+0.557), `.com/` (+0.556), `com/` (+0.556), `n/` (+0.248) |
| `https://www.microsoft.com/en-us/microsoft-365` | `microsoft.com` | 45 | 0 | 1 | 2 | `com` | none | `m/` (+0.502), `om/` (+0.493), `.com/` (+0.492), `com/` (+0.492), `365` (+0.296) |
| `https://account.microsoft.com/account` | `microsoft.com` | 37 | 0 | 1 | 0 | `com` | account | `m/` (+0.561), `om/` (+0.551), `.com/` (+0.551), `com/` (+0.551), `/a` (+0.409) |
| `https://login.microsoftonline.com/` | `microsoftonline.com` | 34 | 0 | 1 | 0 | `com` | login | `m/` (+0.630), `om/` (+0.619), `.com/` (+0.619), `com/` (+0.619), `s://l` (+0.170) |
| `https://appleid.apple.com/sign-in` | `apple.com` | 33 | 0 | 1 | 1 | `com` | sign-in | `m/` (+0.585), `om/` (+0.575), `.com/` (+0.575), `com/` (+0.575), `/a` (+0.252) |
| `https://support.apple.com/en-us/102656` | `apple.com` | 38 | 0 | 1 | 1 | `com` | support | `m/` (+0.585), `om/` (+0.575), `.com/` (+0.574), `com/` (+0.574), `65` (+0.284) |
| `https://www.paypal.com/signin` | `paypal.com` | 29 | 0 | 1 | 0 | `com` | signin | `m/` (+0.736), `om/` (+0.723), `.com/` (+0.722), `com/` (+0.722), `/s` (+0.276) |
| `https://www.paypal.com/myaccount/home` | `paypal.com` | 37 | 0 | 1 | 0 | `com` | account | `m/` (+0.585), `om/` (+0.575), `.com/` (+0.575), `com/` (+0.575), `t/` (+0.458) |
| `https://www.bankofamerica.com/online-banking/sign-in/` | `bankofamerica.com` | 53 | 0 | 1 | 2 | `com` | banking, sign-in | `m/` (+0.451), `om/` (+0.443), `.com/` (+0.443), `com/` (+0.443), `n/` (+0.197) |
| `https://www.icicibank.com/personal-banking/online-services` | `icicibank.com` | 58 | 0 | 1 | 2 | `com` | banking, service | `m/` (+0.457), `om/` (+0.449), `.com/` (+0.449), `com/` (+0.449), `/p` (+0.178) |
| `https://www.sbi.co.in/web/personal-banking` | `sbi.co.in` | 42 | 0 | 1 | 1 | `in` | banking | `n/` (+0.233), `/p` (+0.207), `.in/` (+0.132), `web` (+0.117), `in/` (+0.098) |
| `https://login.gov/` | `login.gov` | 18 | 0 | 0 | 0 | `gov` | login | `s://l` (+0.274), `//l` (+0.266), `://l` (+0.264), `ogin.` (+0.259), `gin.` (+0.255) |

## What drove the errors

- **Character n-grams:** Common high-positive fragments are `m/`, `om/`, `.com/`, `com/`, `/a`, `n/`, `/s`, `65`, `p/`, `/m`. In particular, `.com/` and `com/` fragments appear across benign brand URLs and receive positive phishing weight. The model has learned dataset-specific URL character correlations rather than reputation or page behavior.
- **Login/account wording:** Present in many examples, but it is not the only cause: help pages, a home page, and product-content URLs with no such term were also flagged.
- **Length, queries, subdomains, hyphens, and TLD:** These are descriptive correlates only for the selected v3 model. They are not explicit model features; several short URLs without query parameters were also assigned probability 1.0.
- **HTTPS:** All 18 use HTTPS, but HTTPS is not a direct input to this character-only model. HTTPS is not evidence of legitimacy by itself.
- **Brand names:** The challenge URLs belong to legitimate service roots, but v3 has no brand/reputation field. The model does not know that these roots are legitimate; adding per-domain allow rules would hide the underlying generalization failure.
- **Calibration:** Isotonic calibration selected for a slightly lower Brier score on broad validation extrapolates these domain-shifted scores to 1.0. For the PayPal sign-in URL, the raw logistic margin is negative but the calibrated probability is 1.0; the calibrator therefore pushes that case across the warning threshold and destroys useful confidence ranking. This is a calibration failure in addition to the underlying character-model bias.

## Individual raw-score explanation

- `https://accounts.google.com/signin/v2/identifier?service=mail&flowName=GlifWebSignIn&flowEntry=ServiceLogin`: raw margin 3.6099; calibrated probability 1.0000; positive fragments: `m/` +0.2959, `om/` +0.2907, `.com/` +0.2904, `com/` +0.2904, `n/` +0.1293, `/a` +0.1275, `/i` +0.1215; strongest opposing fragments: `er` -0.1116, `/acco` -0.0832, `.go` -0.0823, `ps://` -0.0674.
- `https://support.google.com/accounts/answer/32040?hl=en`: raw margin 4.3336; calibrated probability 1.0000; positive fragments: `m/` +0.4939, `om/` +0.4852, `.com/` +0.4847, `com/` +0.4847, `/a` +0.3603, `r/` +0.2437, `s://s` +0.1909; strongest opposing fragments: `/acco` -0.1389, `.go` -0.1374, `ps://` -0.1124, `s://` -0.1124.
- `https://www.amazon.in/gp/help/customer/display.html?nodeId=201910460`: raw margin 4.2617; calibrated probability 1.0000; positive fragments: `p/` +0.5159, `r/` +0.2094, `n/` +0.1854, `/d` +0.1790, `/c` +0.1698, `ml` +0.1485, `.ht` +0.1256; strongest opposing fragments: `ww` -0.3287, `s://w` -0.2738, `/www.` -0.1240, `www.` -0.1238.
- `https://google.com/`: raw margin 5.2681; calibrated probability 1.0000; positive fragments: `m/` +1.0177, `om/` +0.9996, `.com/` +0.9988, `com/` +0.9986, `/g` +0.2500, `//g` +0.2001, `s://g` +0.1905; strongest opposing fragments: `ps://` -0.2316, `s://` -0.2316, `tps:` -0.2315, `tps:/` -0.2315.
- `https://myaccount.google.com/security`: raw margin 2.1757; calibrated probability 1.0000; positive fragments: `m/` +0.5695, `om/` +0.5594, `.com/` +0.5589, `com/` +0.5589, `/m` +0.2287, `/s` +0.2136, `s://m` +0.1641; strongest opposing fragments: `ount.` -0.1924, `unt.` -0.1857, `.go` -0.1584, `m/sec` -0.1509.
- `https://github.com/login?return_to=%2Fsettings`: raw margin 2.7434; calibrated probability 1.0000; positive fragments: `m/` +0.5337, `om/` +0.5243, `.com/` +0.5238, `com/` +0.5237, `/g` +0.1311, `//g` +0.1049, `s://g` +0.0999; strongest opposing fragments: `m/log` -0.2806, `om/lo` -0.2789, `m/lo` -0.2757, `com/l` -0.2078.
- `https://docs.github.com/en/authentication`: raw margin 6.0094; calibrated probability 1.0000; positive fragments: `m/` +0.5666, `om/` +0.5566, `.com/` +0.5561, `com/` +0.5561, `n/` +0.2475, `/a` +0.2441, `/d` +0.2389; strongest opposing fragments: `th` -0.1357, `ti` -0.1339, `ps://` -0.1290, `s://` -0.1290.
- `https://www.microsoft.com/en-us/microsoft-365`: raw margin 3.0287; calibrated probability 1.0000; positive fragments: `m/` +0.5018, `om/` +0.4929, `.com/` +0.4925, `com/` +0.4924, `365` +0.2958, `65` +0.2436, `36` +0.2321; strongest opposing fragments: `ww` -0.3886, `s://w` -0.3237, `/www.` -0.1465, `www.` -0.1464.
- `https://account.microsoft.com/account`: raw margin 1.8870; calibrated probability 1.0000; positive fragments: `m/` +0.5610, `om/` +0.5511, `.com/` +0.5506, `com/` +0.5506, `/a` +0.4093, `cc` +0.1388, `//a` +0.1250; strongest opposing fragments: `/acco` -0.2671, `/acc` -0.1935, `ount.` -0.1895, `unt.` -0.1829.
- `https://login.microsoftonline.com/`: raw margin 4.7871; calibrated probability 1.0000; positive fragments: `m/` +0.6304, `om/` +0.6193, `.com/` +0.6187, `com/` +0.6187, `s://l` +0.1703, `//l` +0.1650, `://l` +0.1641; strongest opposing fragments: `ps://` -0.1435, `s://` -0.1435, `tps:` -0.1434, `tps:/` -0.1434.
- `https://appleid.apple.com/sign-in`: raw margin 4.2681; calibrated probability 1.0000; positive fragments: `m/` +0.5855, `om/` +0.5751, `.com/` +0.5746, `com/` +0.5745, `/a` +0.2522, `/s` +0.2196, `app` +0.1528; strongest opposing fragments: `le` -0.2124, `id.a` -0.1744, `ps://` -0.1333, `s://` -0.1333.
- `https://support.apple.com/en-us/102656`: raw margin 6.5776; calibrated probability 1.0000; positive fragments: `m/` +0.5849, `om/` +0.5745, `.com/` +0.5740, `com/` +0.5739, `65` +0.2839, `/e` +0.2325, `s://s` +0.2260; strongest opposing fragments: `t.a` -0.1722, `ps://` -0.1331, `s://` -0.1331, `tps:` -0.1330.
- `https://www.paypal.com/signin`: raw margin -0.4292; calibrated probability 1.0000; positive fragments: `m/` +0.7358, `om/` +0.7228, `.com/` +0.7222, `com/` +0.7221, `/s` +0.2760, `.p` +0.1174, `pay` +0.1074; strongest opposing fragments: `ww` -0.5698, `s://w` -0.4747, `ww.p` -0.2151, `www.p` -0.2151.
- `https://www.paypal.com/myaccount/home`: raw margin 1.2786; calibrated probability 1.0000; positive fragments: `m/` +0.5855, `om/` +0.5751, `.com/` +0.5746, `com/` +0.5745, `t/` +0.4575, `/m` +0.2351, `/h` +0.1552; strongest opposing fragments: `ww` -0.4534, `s://w` -0.3777, `ww.p` -0.1712, `www.p` -0.1712.
- `https://www.bankofamerica.com/online-banking/sign-in/`: raw margin -0.1960; calibrated probability 1.0000; positive fragments: `m/` +0.4513, `om/` +0.4433, `.com/` +0.4430, `com/` +0.4429, `n/` +0.1972, `/s` +0.1693, `/o` +0.1459; strongest opposing fragments: `ww` -0.3495, `s://w` -0.2911, `banki` -0.1564, `ankin` -0.1525.
- `https://www.icicibank.com/personal-banking/online-services`: raw margin 1.2321; calibrated probability 1.0000; positive fragments: `m/` +0.4573, `om/` +0.4492, `.com/` +0.4488, `com/` +0.4487, `/p` +0.1776, `/o` +0.1478, `g/` +0.1465; strongest opposing fragments: `ww` -0.3541, `s://w` -0.2950, `banki` -0.1584, `ankin` -0.1545.
- `https://www.sbi.co.in/web/personal-banking`: raw margin -0.4442; calibrated probability 1.0000; positive fragments: `n/` +0.2326, `/p` +0.2068, `.in/` +0.1324, `web` +0.1167, `in/` +0.0979, `/we` +0.0975, `nk` +0.0753; strongest opposing fragments: `ww` -0.4123, `s://w` -0.3435, `/w` -0.2109, `banki` -0.1845.
- `https://login.gov/`: raw margin 0.7142; calibrated probability 1.0000; positive fragments: `s://l` +0.2744, `//l` +0.2660, `://l` +0.2645, `ogin.` +0.2587, `gin.` +0.2546, `v/` +0.2220, `/l` +0.1289; strongest opposing fragments: `.gov` -0.3378, `.go` -0.2827, `gov` -0.2574, `ps://` -0.2313.
