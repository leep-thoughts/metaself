"""
Difficulty extension of the judge model: the judge evaluates worse on items where it
generates worse (relative to its opponent), the "correlation of generation quality and
evaluation accuracy". Without it, equal opportunity's quality confound cannot appear.

Model, per item x:
    c(x) ~ N(mu, tau^2)                     the judge's relative competence on x
    P(z = +1 | x) = sigmoid(c(x))           own answer wins more often where c is high
    L(x) = L0 + beta * c(x)                 evaluation strength rises with c
    P(vote A | x) = sigmoid(z(x) * L(x) + s)

shape = "bimodal" replaces N(mu, tau^2) with an equal mixture of N(mu -+ DELTA*tau,
(1 - DELTA^2) tau^2): same mean and variance, two humps (easy vs hard items).

beta = 0 recovers synth/judge.py. mu and L0 are solved from the target p = E[sigmoid(c)]
and base accuracy a = E[sigmoid(L(x))] (the judge's accuracy at s = 0), so p, a, s and beta
are four separate knobs.

Expectations are computed exactly by Gauss-Hermite quadrature over c, so measures carry no
sampling noise. Roytburg's proxy is rated on the same item with the same correctness and
no lean: P(vote proxy | x) = sigmoid(z(x) * L(x)).
"""

import numpy as np
from scipy.optimize import brentq

TAU = 1.0
DELTA = 0.9
_nodes, _weights = np.polynomial.hermite_e.hermegauss(80)
_weights = _weights / _weights.sum()
_SHAPES = {
    "normal": (_nodes, _weights),
    "bimodal": (np.r_[-DELTA + np.sqrt(1 - DELTA ** 2) * _nodes, DELTA + np.sqrt(1 - DELTA ** 2) * _nodes],
                np.r_[_weights, _weights] / 2),
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _c(mu, tau=TAU, shape="normal"):
    return mu + tau * _SHAPES[shape][0]


def solve_mu(p, tau=TAU, shape="normal"):
    w = _SHAPES[shape][1]
    return brentq(lambda mu: w @ sigmoid(_c(mu, tau, shape)) - p, -60, 60)


def solve_l0(a, mu, beta, tau=TAU, shape="normal"):
    w = _SHAPES[shape][1]
    return brentq(lambda l0: w @ sigmoid(l0 + beta * _c(mu, tau, shape)) - a, -400, 400)


def expected_measures(p, a, s, beta, tau=TAU, shape="normal") -> dict:
    """Exact expected value of each measure, plus the knob readings."""
    weights = _SHAPES[shape][1]
    mu = solve_mu(p, tau, shape)
    c = _c(mu, tau, shape)
    L = solve_l0(a, mu, beta, tau, shape) + beta * c
    w_a, w_b = weights * sigmoid(c), weights * (1 - sigmoid(c))   # A won / B won
    pa, pb = w_a.sum(), w_b.sum()
    tpr = w_a @ sigmoid(L + s) / pa                   # P(vote A | A won)
    fpr = w_b @ sigmoid(s - L) / pb                   # P(vote A | B won)
    return {
        "p": pa, "base_accuracy": a,
        "actual_accuracy": w_a @ sigmoid(L + s) + w_b @ sigmoid(L - s),
        "panickssery": pa * tpr + pb * fpr - 0.5,
        "equal_opportunity": tpr - (1 - fpr),
        "hspp": fpr,
        "roytburg": w_b @ (sigmoid(s - L) - sigmoid(-L)) / pb,
    }
