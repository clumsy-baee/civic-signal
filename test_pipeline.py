from civic_signal.pipeline import *

def test_classify_multilingual():
    for lg in LANGS:
        for s, w in zip(SECTORS, WORDS[lg]):
            assert classify(TEMPLATES[lg].format(w=w))[0] == s

def test_dedup_and_ranking():
    d = make_districts(); raw, dd = process(make_requests(d))
    assert len(dd) < len(raw)
    rows = score(d, dd); rows["p"] = priority(rows, W_DEFAULT)
    assert rows.p.notna().all() and len(rows) == len(d) * len(SECTORS)

def test_funded_lowers_score():
    d = make_districts(); _, dd = process(make_requests(d)); rows = score(d, dd)
    m = rows.funded_n > .5
    assert (priority(rows, dict(W_DEFAULT, funded=1.0))[m] < priority(rows, dict(W_DEFAULT, funded=0.0))[m]).all()
