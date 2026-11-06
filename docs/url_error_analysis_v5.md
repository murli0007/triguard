# URL error analysis v5

v4's selected threshold was 0.80. v5's selected threshold was 0.79; v5 was frozen before evaluating this challenge. The 21 challenge URLs are all legitimate.

## Why v4 flagged the three URLs

The v4 candidate uses character TF-IDF plus 39 scaled numeric URL features, logistic regression, and sigmoid calibration. Its 0.80 probability threshold corresponds to a raw logistic margin of about 0.152. Each URL's raw margin exceeded that cut. The learned weights reflect generic URL-shape correlations; they do not establish that the named domains are risky.

### `https://login.salesforce.com/`

- v4 probability/margin: 0.992634 / 2.727183; v4 threshold: 0.80.
- v5 probability/prediction: 0.443939 / not flagged at 0.79.
- Explicit lexical indicators: URL contains security/account terms: login.
- Relevant shape values: URL length 29, hostname length 20, path length 1, dots 2, subdomains 1, HTTPS True, query parameters 0.
- Largest active positive character n-gram contributions: `s://l` (+0.050), `://l` (+0.047), `//l` (+0.047), `.s` (+0.046), `ogin.` (+0.034), `.sale` (+0.032), `ale` (+0.030), `gin.` (+0.030).
- Largest active negative character n-gram contributions: `/logi` (-0.161), `/log` (-0.152), `/lo` (-0.141), `//log` (-0.137).
- Largest positive numeric-feature contributions: `num_slashes` (+2.578), `host_max_label_length` (+1.348), `suspicious_host_word_count` (+1.025), `path_segment_count` (+0.980), `num_subdomains` (+0.591), `host_label_count` (+0.591).
- Sigmoid calibration parameters: a=-1.365699, b=-1.178916; it maps the raw margin monotonically to the reported probability. The high scores were not caused by a flat isotonic calibration plateau.

### `https://us.etrade.com/`

- v4 probability/margin: 0.997284 / 3.461329; v4 threshold: 0.80.
- v5 probability/prediction: 0.999120 / flagged at 0.79.
- Explicit lexical indicators: no explicit suspicious-word indicator.
- Relevant shape values: URL length 22, hostname length 13, path length 1, dots 2, subdomains 1, HTTPS True, query parameters 0.
- Largest active positive character n-gram contributions: `s://u` (+0.192), `://u` (+0.176), `//u` (+0.176), `/u` (+0.158), `ad` (+0.098), `/us` (+0.049), `et` (+0.044), `://us` (+0.041).
- Largest active negative character n-gram contributions: `us` (-0.129), `.e` (-0.098), `.com` (-0.092), `e.co` (-0.087).
- Largest positive numeric-feature contributions: `host_max_label_length` (+3.492), `num_slashes` (+2.578), `path_segment_count` (+0.980), `num_subdomains` (+0.591), `host_label_count` (+0.591), `letter_ratio` (+0.391).
- Sigmoid calibration parameters: a=-1.365699, b=-1.178916; it maps the raw margin monotonically to the reported probability. The high scores were not caused by a flat isotonic calibration plateau.

### `https://www.simplii.com/`

- v4 probability/margin: 0.925566 / 0.982341; v4 threshold: 0.80.
- v5 probability/prediction: 0.005268 / not flagged at 0.79.
- Explicit lexical indicators: no explicit suspicious-word indicator.
- Relevant shape values: URL length 24, hostname length 15, path length 1, dots 2, subdomains 1, HTTPS True, query parameters 0.
- Largest active positive character n-gram contributions: `pl` (+0.133), `i.` (+0.113), `i.com` (+0.105), `i.c` (+0.094), `li` (+0.088), `mpli` (+0.084), `i.co` (+0.081), `impli` (+0.079).
- Largest active negative character n-gram contributions: `ww` (-0.582), `s://w` (-0.273), `/www.` (-0.212), `www.` (-0.212).
- Largest positive numeric-feature contributions: `host_max_label_length` (+2.956), `num_slashes` (+2.578), `path_segment_count` (+0.980), `num_subdomains` (+0.591), `host_label_count` (+0.591), `path_length` (+0.208).
- Sigmoid calibration parameters: a=-1.365699, b=-1.178916; it maps the raw margin monotonically to the reported probability. The high scores were not caused by a flat isotonic calibration plateau.

## Comparison with correctly classified legitimate lookalikes

These are correctly classified v4 hard-negative validation examples, separate from the final challenge. They show that the false positives did not come from one simple word alone:

- `https://slack.com/signin` — v4 p=0.6392, correctly below 0.80 (enterprise_login).
- `https://www.oracle.com/account/sign-in/` — v4 p=0.0891, correctly below 0.80 (enterprise_account).
- `https://www.natwest.com/` — v4 p=0.5641, correctly below 0.80 (banking_home).
- `https://www.sofi.com/login/` — v4 p=0.5242, correctly below 0.80 (banking_login).
- `https://www.meridiancu.ca/` — v4 p=0.5208, correctly below 0.80 (banking_home).
- `https://www.absa.co.za/` — v4 p=0.4148, correctly below 0.80 (banking_home).

## Why the remaining v5 false positive was flagged

### `https://us.etrade.com/`

v5 probability/margin: 0.999120 / 4.550480; threshold 0.79.
No explicit suspicious-word indicator. URL length 22, hostname length 13, one subdomain label, HTTPS, and no query parameters.
Top positive word n-grams: `com` (+1.336), `https` (+0.306), `https us` (+0.107).
Top positive character n-grams: `m/` (+0.503), `om/` (+0.489), `com/` (+0.489), `.com/` (+0.489), `et` (+0.161), `/u` (+0.102), `s://u` (+0.093), `//u` (+0.088).
Sigmoid parameters: a=-1.325505, b=-1.003159. The score is driven by generic URL tokens and fragments, especially `com`, `https`, and `.com/`, rather than an explicit suspicious-word rule.


## v5 result on the fixed challenge

v5 flagged 1 of 21 legitimate URLs (4.76%). The three v4 errors are reported individually above; these scores were measured after v5's threshold was fixed. No domain-specific allow rule was introduced.

## UCI false positives and false negatives

On the v5 UCI final partition, confusion matrix TN/FP/FN/TP is [[27181, 2], [153, 16399]]. This is an unseen-domain result on static data, not a live-web performance claim.
