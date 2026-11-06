# URL error analysis

Selected threshold: 0.80. Errors are from the registrable-domain-disjoint UCI test partition, after model selection.

False positives: 1.

- `https://www.travelmatters.stephenandandie.com` — p=0.8155; features: no listed indicator

False negatives: 161.

- `https://www.solutionfun.info` — p=0.0069; features: no listed indicator
- `https://zurbrioerf.com` — p=0.6845; features: no listed indicator
- `https://www.bendiqo-log-au.com` — p=0.2342; features: no listed indicator
- `https://nftracking.com` — p=0.4954; features: no listed indicator
- `https://amzon.co.jp.fotosvalladolid.com` — p=0.5252; features: Hostname has 3 subdomain labels.
- `https://ing.esp-clientes.co` — p=0.7747; features: no listed indicator
- `https://www.alhashimia.net` — p=0.0016; features: no listed indicator
- `https://www.coinbesaprime-logi.com` — p=0.0636; features: no listed indicator
- `https://www.somahe-sarl.com` — p=0.0183; features: no listed indicator
- `https://name-sessrvices.mefound.com` — p=0.7649; features: no listed indicator
- `https://www.microsoft-sicherheitsupdate.com` — p=0.0070; features: no listed indicator
- `https://www.cheatdiamond.my.id` — p=0.0003; features: no listed indicator
- `https://www.direlandgun.com` — p=0.0040; features: no listed indicator
- `https://www.bee-wicked.com/` — p=0.7791; features: no listed indicator
- `https://www.2ww.me/w399y` — p=0.4225; features: no listed indicator

Common false-positive feature presence: No repeated listed feature was shared by false positives.

Hard-negative challenge false positives: 3 of 21 (14.3%).

## Hard-negative examples flagged

- `https://login.salesforce.com/` — enterprise_login; p=0.9926
- `https://us.etrade.com/` — banking_home; p=0.9973
- `https://www.simplii.com/` — banking_home; p=0.9256

## Weaknesses and next steps

This URL-only model cannot observe page content, destination reputation, redirects, or when a URL is newly registered. Current UCI data has no timestamps, so this is not a temporal evaluation. The hard-negative set is small and mixes manually curated URLs with URLs mined from the UCI legitimate class; report it separately and expand it with independently verified examples. Check the listed false positives and false negatives before changing any feature or threshold. Do not add domain-specific allow rules.
