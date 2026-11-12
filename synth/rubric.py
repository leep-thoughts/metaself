"""
Rubric (checklist) judge model: the judge marks each (answer, criterion) as met / not met.
The unit is one criterion of one answer; the judge grades its OWN answers and OTHER
generators' answers to the same prompts.

Per criterion x:
    d(x) ~ N(0, tau^2)                               criterion difficulty
    P(y = met | generator g) = sigmoid(theta_g - d)   own quality p = E[.] for g = own,
                                                      others' quality q for g = other
    L(x) = L0 + beta * (theta_own - d)               evaluation strength; judges worse where
                                                      it would itself fail (difficulty coupling)
    P(judge says met) = sigmoid(z * L + lenient + s * [g = own] + h * (theta_g - theta_other))
        z = +1 if truly met, -1 if not
        lenient  judge says "met" more often on every answer (rubric-specific confound)
        s        true self-preference (log-odds)
        h        generator halo: leans "met" for stronger generators, right or wrong (the
                 rubric analogue of the generator-pair baseline found on real pairwise data)
    L0 solved so the judge's accuracy a (own and others' answers equally weighted, s = 0)
    stays at the value set: p, q, a, lenient, s, beta, h are separate knobs.

All values are exact expectations (Gauss-Hermite quadrature over d).
"""

import numpy as np
from scipy.optimize import brentq

TAU = 1.0
_x, _w = np.polynomial.hermite_e.hermegauss(80)
_w = _w / _w.sum()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _theta(mean, d):
    return brentq(lambda t: _w @ sigmoid(t - d) - mean, -60, 60)


def _world(p, q, a, lenient, s, beta, h, tau):
    d = tau * _x
    th_o, th_k = _theta(p, d), _theta(q, d)
    pi = {"own": sigmoid(th_o - d), "other": sigmoid(th_k - d)}
    bias = {"own": h * (th_o - th_k), "other": 0.0}

    def acc(l0):
        L = l0 + beta * (th_o - d)
        return np.mean([_w @ (pi[g] * sigmoid(L + lenient + bias[g]) + (1 - pi[g]) * sigmoid(L - lenient - bias[g]))
                        for g in ("own", "other")])

    L = brentq(lambda l0: acc(l0) - a, -200, 200) + beta * (th_o - d)
    shift = {"own": lenient + bias["own"] + s, "other": lenient + bias["other"]}
    met_if_true = {g: sigmoid(L + shift[g]) for g in pi}       # P(says met | truly met)
    met_if_false = {g: sigmoid(-L + shift[g]) for g in pi}     # P(says met | truly not met)
    return pi, met_if_true, met_if_false


def expected_measures(p, a, s, q=0.75, lenient=0.0, beta=0.0, h=0.0, tau=TAU) -> dict:
    from measures import rubric
    pi, mt, mf = _world(p, q, a, lenient, s, beta, h, tau)
    out = {"p": float(_w @ pi["own"]), "q": float(_w @ pi["other"])}
    out.update(rubric.expected(_w, pi, mt, mf))
    return out
