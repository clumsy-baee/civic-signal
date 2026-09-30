"""End-to-end prototype: synthetic multilingual requests -> NLU -> dedupe ->
representation correction -> Bayesian smoothing -> MCDA ranking -> sensitivity."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

SECTORS = ["water", "roads", "health", "power", "broadband"]
LANGS = ["en", "hi", "pt", "ru", "zh", "zu"]
WORDS = {  # sector vocabulary per language (stand-in for a real NLU model)
    "en": ["water", "road", "clinic", "electricity", "internet"],
    "hi": ["पानी", "सड़क", "अस्पताल", "बिजली", "इंटरनेट"],
    "pt": ["água", "estrada", "hospital", "energia", "internet"],
    "ru": ["вода", "дорога", "больница", "электричество", "интернет"],
    "zh": ["水", "道路", "医院", "电力", "网络"],
    "zu": ["amanzi", "umgwaqo", "umtholampilo", "ugesi", "i-inthanethi"],
}
TEMPLATES = {"en": "We have no {w} in our village, please help", "hi": "हमारे गाँव में {w} की समस्या है",
             "pt": "Nossa comunidade precisa de {w}", "ru": "В нашей деревне проблема: {w}",
             "zh": "我们村的{w}有问题", "zu": "Sidinga {w} emzini wethu"}
URGENT = {"en": "urgent", "hi": "तुरंत", "pt": "urgente", "ru": "срочно", "zh": "紧急", "zu": "ngokushesha"}
COST = dict(zip(SECTORS, [1.0, 3.0, 2.0, 1.5, 0.8]))  # relative cost per project unit
W_DEFAULT = dict(demand=.30, severity=.15, gap=.25, vulnerability=.15, cost_eff=.15, funded=.20)


def make_districts(n=24, seed=7):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"district": [f"D{i+1:02d}" for i in range(n)], "x": r.random(n), "y": r.random(n),
                      "pop": r.integers(80_000, 900_000, n), "rural": r.beta(3, 2, n),
                      "digital": r.beta(2, 3, n).clip(.05, .95), "poverty": r.beta(2, 4, n),
                      "lang": r.choice(LANGS, n)})
    for s in SECTORS:
        d[f"gap_{s}"] = (r.beta(2, 3, n) * (.5 + d.rural)).clip(0, 1)
        d[f"funded_{s}"] = np.where(r.random(n) < .25, r.uniform(.4, 1, n), 0)
    return d


def make_requests(d, seed=7):
    """Volume follows need x digital access, so digitally poor areas are under-reported."""
    r = np.random.default_rng(seed + 1); rows = []
    for _, a in d.iterrows():
        for si, s in enumerate(SECTORS):
            k = r.poisson(a["pop"] / 1000 * a["digital"] * a[f"gap_{s}"] * .35)
            users = r.integers(0, max(k, 1), k)  # repeat users create duplicates
            for u in users:
                lang = a["lang"] if r.random() < .8 else r.choice(LANGS)
                text = TEMPLATES[lang].format(w=WORDS[lang][si])
                if r.random() < a[f"gap_{s}"] * .6: text += " " + URGENT[lang]
                rows.append((a["district"], f"{a['district']}-u{u}", lang, text))
    return pd.DataFrame(rows, columns=["district", "user", "lang", "text"])


def classify(text):
    """Keyword NLU stub. Swap for an ASR + LLM/classifier ensemble in production."""
    sector = next((s for lg in LANGS for s, w in zip(SECTORS, WORDS[lg]) if w in text), None)
    return sector, any(u in text for u in URGENT.values())


def process(req):
    req = req.copy()
    req[["sector", "urgent"]] = req["text"].apply(lambda t: pd.Series(classify(t)))
    req = req.dropna(subset=["sector"])
    dedup = req.drop_duplicates(["user", "sector"])  # one voice per person per issue
    return req, dedup


def score(d, dedup):
    c = dedup.groupby(["district", "sector"]).agg(n=("user", "size"), urgent=("urgent", "mean")).reset_index()
    rows = d.merge(pd.DataFrame([(x, s) for x in d.district for s in SECTORS], columns=["district", "sector"]))
    rows = rows.merge(c, how="left", on=["district", "sector"]).fillna({"n": 0, "urgent": 0})
    rows["gap"] = [rows.loc[i, f"gap_{s}"] for i, s in zip(rows.index, rows.sector)]
    rows["funded"] = [rows.loc[i, f"funded_{s}"] for i, s in zip(rows.index, rows.sector)]
    # representation correction: up-weight low-connectivity districts
    rows["n_w"] = rows.n / (.25 + rows.digital)
    # empirical-Bayes shrinkage toward the national rate for sparse areas
    kpop = rows["pop"] / 1000; g = rows.n_w.sum() / kpop.sum()
    rows["demand"] = (rows.n_w + 5 * g) / (kpop + 5)
    rows["severity"] = rows.urgent
    rows["vulnerability"] = rows["pop"] * rows.poverty * (.5 + rows.rural)
    rows["cost_eff"] = rows["pop"] * rows.gap / rows.sector.map(COST)
    mm = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)
    for k in ["demand", "severity", "gap", "vulnerability", "cost_eff", "funded"]:
        rows[k + "_n"] = mm(rows[k])
    return rows


def priority(rows, w):
    return sum(w[k] * rows[k + "_n"] * (-1 if k == "funded" else 1) for k in w)


def sensitivity(rows, draws=500, top=10, seed=3):
    r = np.random.default_rng(seed); keys = list(W_DEFAULT); base = np.array([W_DEFAULT[k] for k in keys])
    hits = np.zeros(len(rows))
    for _ in range(draws):
        w = dict(zip(keys, r.dirichlet(base * 40)))
        hits[np.argsort(-priority(rows, w).values)[:top]] += 1
    return hits / draws


def run(out="docs"):
    d = make_districts(); raw, dedup = process(make_requests(d)); rows = score(d, dedup)
    rows["priority"] = priority(rows, W_DEFAULT); rows["stability"] = sensitivity(rows)
    rows = rows.sort_values("priority", ascending=False)
    Path("data").mkdir(exist_ok=True); rows.to_csv("data/ranked_projects.csv", index=False)
    cols = ["district", "sector", "x", "y", "pop", "n", "urgent", "lang", "stability"] + [k + "_n" for k in W_DEFAULT]
    payload = {"weights": W_DEFAULT, "rows": json.loads(rows[cols].round(4).to_json(orient="records")),
               "stats": {"raw": len(raw), "dedup": len(dedup)}}
    tpl = (Path(__file__).parent / "dashboard_template.html").read_text(encoding="utf-8")
    Path(out).mkdir(exist_ok=True)
    (Path(out) / "index.html").write_text(tpl.replace("/*DATA*/", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")
    return rows


if __name__ == "__main__":
    r = run(); print(r[["district", "sector", "priority", "stability"]].head(10).to_string(index=False))
