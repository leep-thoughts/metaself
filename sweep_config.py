"""
Grid and run parameters for meta_eval_sweep.py, pulled into their own file so the
knobs can be edited without touching the sweep/measurement logic.

NOTE on range: p and accuracy both get converted to log-odds via logit(x) =
log(x / (1-x)) elsewhere in the pipeline (see synth/oracle.py, synth/judge.py).
logit(0) and logit(1) are mathematically undefined (infinite log-odds) -- and in
Python, 1.0/0.0 raises ZeroDivisionError before even reaching log(), so including
either endpoint literally would crash the sweep. P_VALUES and ACCURACIES therefore
stop just short of the true endpoints (0.02 / 0.98) instead.
"""

P_VALUES = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
ACCURACIES = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
SELF_LEANS = [-2.0, -1.5, -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]

N_ITEMS = 2000
N_REPEATS = 10
MASTER_SEED = 12345

# Representative fixed values used by the single-knob charts (response-to-self_lean,
# quality-sweep, accuracy-sweep) -- must each be a member of the lists above.
REP_P = 0.8
REP_ACCURACY = 0.8
REP_SELF_LEAN = 0.2

# Fine 1-D slices for the dashboard's line charts: one knob swept at the same step
# size as the dashboard sliders, the other two held at the REP_* anchors. Refining
# the full 3-D grid to this step instead would be ~760k points.
FINE_P_VALUES = [round(0.02 + 0.01 * i, 2) for i in range(97)]
FINE_ACCURACIES = [round(0.02 + 0.01 * i, 2) for i in range(97)]
FINE_SELF_LEANS = [round(-2.0 + 0.05 * i, 2) for i in range(81)]
