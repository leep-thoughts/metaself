"""
Panickssery et al. -- the naive self-preference measure.

Bias = mean over items of P(vote=A) - 0.5.

It only needs the judge's votes -- no oracle data at all. That's exactly its weakness:
it can't tell "the judge likes A because A's outputs are genuinely better" apart from
"the judge likes A because it's biased toward A".

Derivation (see synth/judge.py for the generative model; L = logit(accuracy)):
    E[Bias] = p * sigmoid(L + self_lean) + (1-p) * sigmoid(-L + self_lean) - 0.5
  where p = P(oracle says A wins).

At self_lean = 0, sigmoid(L) = accuracy and sigmoid(-L) = 1 - accuracy, so this reduces
to:
    E[Bias] = p*accuracy + (1-p)*(1-accuracy) - 0.5

which is nonzero whenever quality isn't exactly tied (p != 0.5) or the judge isn't an
exact coin flip (accuracy != 0.5) -- a quality/accuracy confound leaking straight into
the "bias" estimate, with zero actual self-preference involved. This is the headline
demonstration of why this measure is suspect: run it with self_lean=0 and a real
quality gap (p != 0.5) and watch it report a nonzero bias anyway.
"""


def estimate_bias(judge_decisions) -> float:
    return sum(judge_decisions) / len(judge_decisions) - 0.5
