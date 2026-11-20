"""
Data points behind the chapter-1 slopes of the "Self-Preference Confounds and Fixes" page.

Reruns the reweighting grid (real/slope_gap.py, as in real/pair_check.py) for the three versions of
the votes (real, lean removed, lean from the rule) and saves every reachable grid point: per judge,
the achieved generation quality p, the achieved judge accuracy and each measure.

Run with: python -m real.plot_data      writes real/plot_grid_points.csv
"""

import os

import numpy as np
import pandas as pd

from real import null_injection
from real import slope_gap as sg
from real.pair_check import UNIT, units, with_l

sg.WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", sg.WORKERS))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_grid_points.csv")


def main():
    d = null_injection.load()
    u, _ = units(d)
    key = pd.MultiIndex.from_frame(d[UNIT])
    ui = u.set_index(UNIT)
    b = ui.b.reindex(key).to_numpy()
    versions = {"real": d.l.to_numpy(),
                "lean removed": d.l.to_numpy() - np.nan_to_num(b - ui.b_bar.reindex(key).to_numpy()),
                "lean from rule": d.l.to_numpy() - np.nan_to_num(b - ui.b_model.reindex(key).to_numpy())}
    out = []
    for name, l in versions.items():
        dd = with_l(d, l)
        sg.load = lambda dd=dd: dd
        df = sg.prepare()
        r = sg.run_grid(df, n_reps=4)
        r["version"] = name
        out.append(r)
        print(name, len(r), flush=True)
    pd.concat(out).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
