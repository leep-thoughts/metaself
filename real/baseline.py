"""
Baselines on real judges (Chen et al. 2025 release, see real/chen.py): before turning
any knob, what do natural p, actual accuracy and the three published measures look
like for each judge -- overall, per task, and per opponent?

Only usable pairs (exactly one answer correct). "A" in the measure code = the judge's
own answer, so oracle_truth = own answer is the correct one, judge_decisions = voted own.

Vote rules (the judge saw each pair in both orders, with a Tie option):
    A  average both orders: vote own if mean(p_own) > mean(p_other); exact ties dropped
    C  both orders must agree: own (or other) only if it is the argmax over
       [own, tie, other] in BOTH orders; anything else dropped

Roytburg et al.'s EQB comes from their own judge-swap subset (real/roytburg.py) and is
attached to the direct-setting rows only.

Error bars: 95% percentile bootstrap, resampling problems (all of a problem's rows
move together, since one problem appears once per opponent).

Run with: python -m real.baseline
"""

import os

import numpy as np
import pandas as pd

from measures import panickssery, equal_opportunity, hspp, roytburg
from real import chen
from real import roytburg as roytburg_data

N_BOOT = 500
SEED = 0
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_results.csv")


def votes(df: pd.DataFrame, rule: str = "A") -> pd.Series:
    """1.0 = voted own, 0.0 = voted other, NaN = tie (dropped)."""
    if rule == "A":
        own = (df.p_own_1 + df.p_own_2) / 2
        other = (df.p_other_1 + df.p_other_2) / 2
        return pd.Series(np.where(own > other, 1.0, np.where(own < other, 0.0, np.nan)), index=df.index)
    if rule == "C":
        def verdict(g):  # 0 = own, 1 = tie, 2 = other
            return np.argmax(df[[f"p_own_{g}", f"p_tie_{g}", f"p_other_{g}"]].to_numpy(), axis=1)
        v1, v2 = verdict(1), verdict(2)
        return pd.Series(np.where((v1 == 0) & (v2 == 0), 1.0,
                                  np.where((v1 == 2) & (v2 == 2), 0.0, np.nan)), index=df.index)
    raise ValueError(rule)


def stats(truth, vote) -> dict:
    """Knob readings + the three measures, on one sample. Measures run unchanged."""
    truth, vote = list(truth), list(vote)
    s = {"p": float(np.mean(truth)),
         "actual_accuracy": float(np.mean([t == v for t, v in zip(truth, vote)])),
         "panickssery": panickssery.estimate_bias(vote)}
    has_both = 0 < sum(truth) < len(truth)
    s["equal_opportunity"] = equal_opportunity.estimate_bias(truth, vote) if has_both else np.nan
    s["hspp"] = hspp.estimate_bias(truth, vote) if not all(truth) else np.nan
    return s


def bootstrap(group: pd.DataFrame, rng) -> dict:
    """Point estimates plus 95% CIs, resampling problem_id clusters."""
    truth = group.own_wins.to_numpy()
    vote = group.vote.to_numpy().astype(bool)
    point = stats(truth, vote)
    idx_by_problem = list(group.groupby(["task", "problem_id"]).indices.values())
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(len(idx_by_problem), size=len(idx_by_problem))
        idx = np.concatenate([idx_by_problem[i] for i in pick])
        boots.append(stats(truth[idx], vote[idx]))
    boots = pd.DataFrame(boots)
    out = {"n_pairs": len(group), "n_problems": len(idx_by_problem)}
    for k, v in point.items():
        out[k] = v
        out[k + "_lo"], out[k + "_hi"] = np.nanpercentile(boots[k], [2.5, 97.5])
    return out


def roytburg_table(rng) -> pd.DataFrame:
    """
    Roytburg et al.'s EQB per judge (all / per task / per opponent), on their judge-swap
    subset (real/roytburg.py): items where the judge's own answer is wrong and at least
    one outcome-matched proxy exists. Uses the judge's probabilities, as the paper does,
    so it doesn't depend on the vote rule. 95% bootstrap CI over problems.
    """
    df = roytburg_data.load()
    df = df[~df.own_correct].dropna(subset=["s_proxy"]).rename(columns={"problem_idx": "problem_id"})
    rows = []
    for split in ("all", "task", "opponent"):
        keys = ["judge"] if split == "all" else ["judge", split]
        for key, g in df.groupby(keys):
            key = key if isinstance(key, tuple) else (key,)
            g = g.reset_index(drop=True)
            truth, s_self, s_proxy = g.own_correct.to_numpy(), g.s_self.to_numpy(), g.s_proxy.to_numpy()
            point = roytburg.estimate_bias(truth, s_self, s_proxy)
            idx_by_problem = list(g.groupby(["task", "problem_id"]).indices.values())
            boots = []
            for _ in range(N_BOOT):
                pick = rng.integers(len(idx_by_problem), size=len(idx_by_problem))
                idx = np.concatenate([idx_by_problem[i] for i in pick])
                boots.append(roytburg.estimate_bias(truth[idx], s_self[idx], s_proxy[idx]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"split": split, "judge": key[0], "slice": key[1] if len(key) > 1 else "all",
                         "roytburg": point, "roytburg_lo": lo, "roytburg_hi": hi, "roytburg_n": len(g)})
    return pd.DataFrame(rows)


def add_roytburg(res: pd.DataFrame, rng) -> pd.DataFrame:
    """Attach EQB to the direct-setting rows (same judges, tasks and opponents)."""
    rb = roytburg_table(rng).assign(setting="direct")
    return res.merge(rb, on=["setting", "split", "judge", "slice"], how="left")


def run() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for setting in chen.SETTING_GLOB:
        u = chen.usable(chen.load(setting))
        for rule in ("A", "C"):
            u["vote"] = votes(u, rule)
            kept = u.dropna(subset=["vote"])
            dropped = 1 - len(kept) / len(u)
            for split in ("all", "task", "opponent"):
                keys = ["judge"] if split == "all" else ["judge", split]
                for key, g in kept.groupby(keys):
                    key = key if isinstance(key, tuple) else (key,)
                    rows.append({"setting": setting, "rule": rule, "split": split,
                                 "judge": key[0], "slice": key[1] if len(key) > 1 else "all",
                                 "dropped_frac_setting": dropped,
                                 **bootstrap(g.reset_index(drop=True), rng)})
    return add_roytburg(pd.DataFrame(rows), rng)


if __name__ == "__main__":
    res = run()
    res.to_csv(OUT_PATH, index=False)
    cols = ["n_pairs", "p", "actual_accuracy", "panickssery", "equal_opportunity", "hspp", "roytburg"]
    for (setting, rule), g in res[res.split == "all"].groupby(["setting", "rule"], sort=False):
        print(f"\n== {setting}, rule {rule} (dropped {g.dropped_frac_setting.iloc[0]:.0%} of usable pairs)")
        print(g.set_index("judge")[cols].sort_values("actual_accuracy").round(3).to_string())
    print(f"\nwrote {OUT_PATH}")
