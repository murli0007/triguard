# URL error analysis

Selected threshold: 0.70. Errors are from the registrable-domain-disjoint UCI test partition, after model selection.

False positives: 1.

- `https://www.2020projectmanagement.com` — p=0.7296; features: no listed indicator

False negatives: 164.

- `https://www.id292468292.ru` — p=0.0897; features: no listed indicator
- `https://www.ak3.duckdns.org/` — p=0.0057; features: no listed indicator
- `https://www.coalco.ru/alegrospolish/4/` — p=0.2121; features: no listed indicator
- `https://www.ridm.in` — p=0.0000; features: no listed indicator
- `https://www.ahsansupermarket.com` — p=0.0056; features: no listed indicator
- `https://www.coalco.ru/alegrospolish/6` — p=0.2121; features: no listed indicator
- `https://www.shapeshinft.click/` — p=0.2121; features: no listed indicator
- `https://www.mygov3-au.top` — p=0.2121; features: no listed indicator
- `https://www.ekaterinajeans.ru/alegrospolish/1` — p=0.6923; features: no listed indicator
- `https://www.dentalbolivia.com/acs` — p=0.6923; features: no listed indicator
- `https://www.gyvd.xyz/` — p=0.6923; features: no listed indicator
- `https://www.yy24b.cc/` — p=0.0098; features: no listed indicator
- `https://www.coalco.ru/alegrospolish/6/` — p=0.2121; features: no listed indicator
- `https://www.atoclaim.org` — p=0.0013; features: no listed indicator
- `https://www.grst.com/` — p=0.0056; features: no listed indicator

Common false-positive feature presence: No repeated listed feature was shared by false positives..

Hard-negative challenge false positives: 18 of 29 (62.1%).

## Hard-negative examples flagged

- `https://accounts.google.com/signin/v2/identifier?service=mail&flowName=GlifWebSignIn&flowEntry=ServiceLogin` — official_login; p=1.0000
- `https://support.google.com/accounts/answer/32040?hl=en` — official_help; p=1.0000
- `https://www.amazon.in/gp/help/customer/display.html?nodeId=201910460` — official_help; p=1.0000
- `https://google.com/` — official_home; p=1.0000
- `https://myaccount.google.com/security` — official_account; p=1.0000
- `https://github.com/login?return_to=%2Fsettings` — official_login; p=1.0000
- `https://docs.github.com/en/authentication` — official_help; p=1.0000
- `https://www.microsoft.com/en-us/microsoft-365` — official_content; p=1.0000
- `https://account.microsoft.com/account` — official_account; p=1.0000
- `https://login.microsoftonline.com/` — official_login; p=1.0000
- `https://appleid.apple.com/sign-in` — official_login; p=1.0000
- `https://support.apple.com/en-us/102656` — official_help; p=1.0000
- `https://www.paypal.com/signin` — official_login; p=1.0000
- `https://www.paypal.com/myaccount/home` — official_account; p=1.0000
- `https://www.bankofamerica.com/online-banking/sign-in/` — official_login; p=1.0000
- `https://www.icicibank.com/personal-banking/online-services` — official_payment; p=1.0000
- `https://www.sbi.co.in/web/personal-banking` — official_payment; p=1.0000
- `https://login.gov/` — official_login; p=1.0000

## Weaknesses and next steps

This URL-only model cannot observe page content, destination reputation, redirects, or when a URL is newly registered. Current UCI data has no timestamps, so this is not a temporal evaluation. The hard-negative set is small and mixes manually curated URLs with URLs mined from the UCI legitimate class; report it separately and expand it with independently verified examples. Check the listed false positives and false negatives before changing any feature or threshold. Do not add domain-specific allow rules.
