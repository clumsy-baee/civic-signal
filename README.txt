# Civic Signal

A prototype of a multilingual, federated **Digital Public Good** that turns citizen development requests
into a ranked, explainable shortlist of infrastructure projects for policymakers.

**Live demo:** enable GitHub Pages (Settings → Pages → branch `main`, folder `/docs`) and open the site.
No server needed; `docs/index.html` is self-contained.

## What the prototype does

1. **Ingest** synthetic requests in 6 languages (English, Hindi, Portuguese, Russian, Chinese, Zulu).
   Districts with less internet access report less, on purpose.
2. **Understand** each request: sector and urgency (keyword stub, replace with ASR + LLM ensemble).
3. **Deduplicate** repeat submissions from the same person.
4. **Correct representation** by up-weighting low-connectivity districts, then apply empirical-Bayes
   shrinkage so sparse districts are not over-read.
5. **Rank** with a transparent MCDA score:
   `P = w1·demand + w2·urgency + w3·gap + w4·vulnerability + w5·cost-effectiveness − w6·already-funded`
6. **Stress-test** the ranking: 500 random weight settings show how often each project stays in the top 10.
7. **Explore** in the dashboard: change weights live and watch the map and ranking update.

## Run it

```bash
pip install -r requirements.txt
python -m civic_signal.pipeline    # writes docs/index.html and data/ranked_projects.csv
python -m pytest                   # tests
```

## Layout

```
civic_signal/pipeline.py             data generation, NLU stub, scoring, sensitivity
civic_signal/dashboard_template.html dashboard (data injected at build time)
docs/index.html                      generated dashboard (GitHub Pages)
tests/                               unit tests
```

## From prototype to platform

| Prototype | Production direction |
|---|---|
| Synthetic requests | WhatsApp, IVR, SMS/USSD, assisted kiosks |
| Keyword classifier | Whisper-class ASR + multilingual LLM with schema-constrained extraction |
| Random district grid | Real admin boundaries, H3 indexing, PostGIS |
| Single node | One sovereign node per BRICS country, federated learning, aggregated cross-country benchmarks |
| Ranking only | Closed-loop status updates and difference-in-differences impact tracking |

All data here is synthetic. Do not read the rankings as findings about any real place.

## License

Apache-2.0. Add the license text via GitHub (Add file → Create new file → name it `LICENSE` → Choose a license template).
