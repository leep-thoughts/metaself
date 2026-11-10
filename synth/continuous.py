"""
Continuous-quality judge model, so measures built on EQUAL-QUALITY pairs (Yang et al.,
measures/yang.py) can be meta-evaluated next to the others in one world. The binary
oracle in synth/oracle.py always has a winner, so it has no equal-quality pairs.

World (quality in units of the spread of answer quality):
    Q_own ~ N(theta_J, 1), Q_other ~ N(theta_O, 1)      every answer, independent
    g  = Q_own - Q_other ~ N(m, 2)                     own vs third-party; p = P(g > 0)
    g' = Q_j - Q_k       ~ N(0, 2)                     two third-party answers
    benchmark score = Q + e, e ~ N(0, sigma_e^2)       so the estimated gap is g + d,
                                                       d ~ N(0, 2 sigma_e^2)
    equal quality: |estimated gap| <= eps
    P(pick X in slot A over Y in slot B) = sigmoid(kappa*(Q_X - Q_Y) + s*(own X - own Y) + pos)

    pos   position bias toward the first slot (log-odds)
    kappa solved so that accuracy a = P(pick the truly better answer), own-vs-other pairs,
          averaged over both orders, at s = 0.

Defaults map Yang's setup: eps = 0.25 points on scores whose spread across answers is
assumed ~1.5 points -> 0.17; two reference judges agreeing at 0.66 -> each has error
variance ~0.52, their mean ~0.26 -> sigma_e = 0.5.

The four earlier measures use z = sign(g) as the oracle and the order-averaged vote
v(g) = [sigmoid(kappa g + s + pos) + sigmoid(kappa g + s - pos)] / 2. Roytburg's proxy is
another model of the judge's strength, same correctness, no lean (its best case).

All values are exact expectations (Gauss-Legendre / Gauss-Hermite quadrature).
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

EPS = 0.17
SIGMA_E = 0.5
V = 2.0          # variance of a quality gap between two answers

_gl_x, _gl_w = np.polynomial.legendre.leggauss(200)
_gh_x, _gh_w = np.polynomial.hermite_e.hermegauss(40)
_gh_w = _gh_w / _gh_w.sum()
_eq_x, _eq_w = np.polynomial.legendre.leggauss(40)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _half(m, lo, hi):
    """Nodes and normalized-density weights for g ~ N(m, V) on [lo, hi]."""
    x = (hi - lo) / 2 * _gl_x + (hi + lo) / 2
    w = (hi - lo) / 2 * _gl_w * norm.pdf(x, m, np.sqrt(V))
    return x, w


def _gap_nodes(m):
    span = 9 * np.sqrt(V)
    xn, wn = _half(m, min(m - span, -1e-9), 0.0)
    xp, wp = _half(m, 0.0, max(m + span, 1e-9))
    return xn, wn, xp, wp


def _equal_nodes(m, eps, sigma_e):
    """Nodes/weights for g given |g + d| <= eps (weights sum to 1)."""
    D = 2 * sigma_e ** 2
    gh = eps * _eq_x                                   # estimated gap on [-eps, eps]
    wh = eps * _eq_w * norm.pdf(gh, m, np.sqrt(V + D))
    mean = m + V / (V + D) * (gh - m)
    sd = np.sqrt(V * D / (V + D))
    g = mean[:, None] + sd * _gh_x[None, :]
    w = wh[:, None] * _gh_w[None, :]
    return g.ravel(), (w / w.sum()).ravel()


def _accuracy(kappa, m, pos):
    xn, wn, xp, wp = _gap_nodes(m)
    right = lambda g, sign: (sigmoid(sign * kappa * g + pos) + sigmoid(sign * kappa * g - pos)) / 2
    return (wn @ right(xn, -1) + wp @ right(xp, 1)) / (wn.sum() + wp.sum())


def solve_kappa(a, m, pos=0.0):
    lo = _accuracy(0.0, m, pos)
    if a <= lo:
        raise ValueError(f"accuracy {a} is below chance-level {lo:.3f} for this position bias")
    return brentq(lambda k: _accuracy(k, m, pos) - a, 0.0, 200.0)


def expected_measures(p, a, s, pos=0.0, sigma_e=SIGMA_E, eps=EPS) -> dict:
    m = np.sqrt(V) * norm.ppf(p)
    kappa = solve_kappa(a, m, pos)
    vote = lambda g, lean: (sigmoid(kappa * g + lean + pos) + sigmoid(kappa * g + lean - pos)) / 2
    xn, wn, xp, wp = _gap_nodes(m)
    pa, pb = wp.sum(), wn.sum()
    tpr = wp @ vote(xp, s) / pa
    fpr = wn @ vote(xn, s) / pb
    out = {
        "p": pa / (pa + pb), "base_accuracy": a,
        "actual_accuracy": (wp @ vote(xp, s) + wn @ (1 - vote(xn, s))) / (pa + pb),
        "panickssery": (wp @ vote(xp, s) + wn @ vote(xn, s)) / (pa + pb) - 0.5,
        "equal_opportunity": tpr - (1 - fpr),
        "hspp": fpr,
        "roytburg": wn @ (vote(xn, s) - vote(xn, 0.0)) / pb,
    }
    g, w = _equal_nodes(m, eps, sigma_e)
    pir = w @ (sigmoid(kappa * g + s + pos) * sigmoid(kappa * g + s - pos))
    g0, w0 = _equal_nodes(0.0, eps, sigma_e)
    null = w0 @ (sigmoid(kappa * g0 + pos) * sigmoid(kappa * g0 - pos))
    out.update(yang=pir - null, yang_pir=pir, yang_null=null, residual_gap=w @ g)
    return out
