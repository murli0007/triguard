# URL error analysis

Selected threshold: 0.71. Errors are from the registrable-domain-disjoint UCI test partition, after model selection.

False positives: 5.

- `https://www.iswc2017.semanticweb.org` — p=0.8985; features: no listed indicator
- `https://www.apocalypse2008-2015.com` — p=0.9816; features: no listed indicator
- `https://www.entreprises.cci-paris-idf.fr` — p=0.9325; features: no listed indicator
- `https://www.weboldal-webaruhaz-keszites-budapest.com` — p=0.7538; features: no listed indicator
- `https://www.dynamicleadership.management` — p=0.9556; features: no listed indicator

False negatives: 207.

- `https://www.mutfakci.az` — p=0.0013; features: no listed indicator
- `https://www.ioginunlcovlabcp.com` — p=0.0040; features: no listed indicator
- `https://www.apple-us.info` — p=0.2098; features: no listed indicator
- `https://www.ahsansupermarket.com` — p=0.0130; features: no listed indicator
- `https://www.mofcomx.com/jp` — p=0.6764; features: no listed indicator
- `https://gyt.italkolmascotas.repl.co` — p=0.4357; features: no listed indicator
- `https://www.aflacxvmlyr.com` — p=0.0007; features: no listed indicator
- `https://wed.bancaebisaplus.repl.co` — p=0.4699; features: no listed indicator
- `https://www.shopwebstudio.com` — p=0.0275; features: no listed indicator
- `https://www.microsoft-sicherheitsupdate.com` — p=0.2173; features: no listed indicator
- `https://wvwcajalsullana.com` — p=0.6944; features: no listed indicator
- `https://onl.tufinansie.repl.co` — p=0.4799; features: no listed indicator
- `https://www.coinbasewalletones.com` — p=0.3182; features: no listed indicator
- `https://www.chibabamk.fytqm.com` — p=0.0267; features: no listed indicator
- `https://www.goto.otopplace.com` — p=0.0107; features: no listed indicator

Common false-positive feature presence: No repeated listed feature was shared by false positives.

Hard-negative challenge false positives: 1 of 23 (4.3%).

## Hard-negative examples flagged

- `https://accounts.spotify.com/en/login` — account_management; p=0.7678

## Weaknesses and next steps

This URL-only model cannot observe page content, destination reputation, redirects, or when a URL is newly registered. Current UCI data has no timestamps, so this is not a temporal evaluation. The hard-negative set is small and mixes manually curated URLs with URLs mined from the UCI legitimate class; report it separately and expand it with independently verified examples. Check the listed false positives and false negatives before changing any feature or threshold. Do not add domain-specific allow rules.
