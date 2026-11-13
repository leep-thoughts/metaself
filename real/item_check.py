"""
What drives equal opportunity's real slope with ACCURACY (grid median ~1.1)? It survives
removing the generator-pair lean (real/pair_check.py), so it lives below the unit level.

Hypothesis: an item-level lean shared across judges (eta_x: one answer looks better than the
oracle says, e.g. correct but messy vs wrong but polished). The grid moves accuracy by
reweighting on `confidence` = other judges' mean |l| on the same pair item, which selects
items with large shared leans.

Per row: residual e = l - (z L_u + b_u)  (unit fit from real/pair_check.units; units without
both outcomes: judge x task x z mean). Versions, each through real/slope_gap's pipeline:
    real            as released
    shuffle_items   e shuffled within (judge, unit, z): no item-level structure left
    no_shared_item  e minus the mean residual of the OTHER judges on the same item
                    (leave-one-judge-out estimate of eta_x)
If no_shared_item kills the accuracy slope, the shared item lean is the cause.

Run with: python -m real.item_check          writes real/item_check_results.json
"""

import json
import os

import numpy as np
import pandas as pd

from real import null_injection
from real.pair_check import UNIT, pipeline, units, with_l

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "item_check_results.json")
ITEM = ["task", "opponent", "proxy", "problem_idx"]


def main():
    d = null_injection.load()
    u, _ = units(d)
    ui = u.set_index(UNIT)
    key = pd.MultiIndex.from_frame(d[UNIT])
    z = np.where(d.a_correct, 1.0, -1.0)
    fit = z * ui.L.reindex(key).to_numpy() + ui.b.reindex(key).to_numpy()
    fallback = d.groupby(["judge", "task", "a_correct"]).l.transform("mean").to_numpy()
    fit = np.where(np.isnan(fit), fallback, fit)
    e = d.l.to_numpy() - fit

    rng = np.random.default_rng(0)
    grp = d.groupby(UNIT + ["a_correct"]).indices
    e_shuf = e.copy()
    for idx in grp.values():
        e_shuf[idx] = e[rng.permutation(idx)]

    es = pd.Series(e)
    tot = es.groupby([d[c] for c in ITEM]).transform("sum").to_numpy()
    cnt = es.groupby([d[c] for c in ITEM]).transform("count").to_numpy()
    eta = np.where(cnt > 1, (tot - e) / np.maximum(cnt - 1, 1), 0.0)
    shared_corr = float(np.corrcoef(e[cnt > 1], eta[cnt > 1])[0, 1])
    print(f"corr(own residual, other judges' residual on the same item) = {shared_corr:.3f}")

    versions = {"real": d.l.to_numpy(), "shuffle_items": fit + e_shuf, "no_shared_item": fit + e - eta}
    out = {"residual_corr_across_judges": shared_corr}
    for name, l in versions.items():
        out[name] = pipeline(with_l(d, l))
        g = out[name]["grid"]["median"]
        print(f"{name:15s} grid median: " + ", ".join(f"{k} {v:.2f}" for k, v in g.items()), flush=True)
        json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
