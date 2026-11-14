"""Performance matrix P[c,d] and transfer gap G for text only (Qwen3.5-9B, 1000 prompts, Readout Filtering).
c = fitting corpus (EN Wikipedia, FR Wikipedia), d = evaluation domain: EN items = multihop/typo/association/poetry (English prompts),
FR items = the scored multilingual items whose prompt is French (ids french-*, fr-*).
P = recall@10 (EN: macro of the 4 tasks; FR: item mean) of per-item recall@10 (item = fraction of intermediates whose best rank over layers <= 10), mean over pools A and B.
G[c,d] = (P[d,d] - P[c,d]) / P[d,d]."""
import json, pathlib
import pandas as pd
R = pathlib.Path(__file__).resolve().parents[1] / "qwen3.5-9b"
cor = pd.read_csv(R / "correctness.csv"); cor.columns = ["eval", "item", "ok"]
ok = set(map(tuple, cor[cor.ok.astype(str) == "True"][["eval", "item"]].values))
def domain(ev, it):
    if ev == "multilingual": return "FR" if it.startswith(("french-", "fr-")) else None
    return "EN"
def per_item(path, lenses):
    d = pd.read_csv(path); d = d[d.lens.isin(lenses)]
    m = d.groupby(["lens", "eval", "item", "intermediate"])["rank"].min().reset_index()
    m["hit"] = (m["rank"] <= 10).astype(float)
    return m.groupby(["lens", "eval", "item"]).hit.mean().reset_index()
out = {}
for c in ["en", "fr"]:
    vals = []
    for pool in "ab":
        run = R / "runs" / f"{c}-{pool}" / "n1000"
        x = pd.concat([per_item(run / "baselines_filtered.csv", ["logit", "j", "r"]), per_item(run / "jpp_scores.csv", ["jpp"])])
        x["dom"] = [domain(e, i) for e, i in zip(x["eval"], x["item"])]
        x = x.dropna(subset=["dom"]); x["pool"] = pool; vals.append(x)
    x = pd.concat(vals)
    for (l, d), g in x.groupby(["lens", "dom"]):
        # EN: macro of the 4 English-prompt tasks (item mean per task, then task mean); FR: item mean
        pools = g.groupby(["pool", "eval"]).hit.mean().groupby("pool").mean()
        out.setdefault(l, {}).setdefault(d, {})[c] = float(pools.mean() * 100)
        out[l][d]["n_items"] = int(g[g.pool == "a"].shape[0])
for l in out:
    for d, row in out[l].items():
        dd = {"EN": "en", "FR": "fr"}[d]
        print(l, d, {k: round(v, 1) for k, v in row.items()})
json.dump(out, open(pathlib.Path(__file__).parent / "P_text.json", "w"), indent=1)
