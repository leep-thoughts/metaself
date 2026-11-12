# metaself: context for Claude Code

Research project: meta-evaluating self-preference-bias (SPB) measures for LLM judges. A synthetic
simulator with known knobs (generation quality p, judge accuracy a, true self-lean s, difficulty
coupling β, halo h) tests whether each published measure responds only to s; real public data
(Chen et al., Roytburg et al., Yang et al., Pombal et al.) checks the simulator.

## How the user wants to work
- Replies short; tables and lists over prose; no long paragraphs.
- **GPU jobs: write the submit script, the user submits.** CPU-only jobs (Aleph calls, downloads) I may submit.
- **Free models only**: Aleph (Vulcan's in-house gateway) or self-hosted open weights via vLLM. No paid APIs.
  Aleph key: `~/.aleph_tyk.env`, never print or commit it.
- "Follow previous works": match the original papers' settings (e.g. Pombal: model defaults, only max tokens set).
- Report any new dependency (module or Python package).
- Commit only when asked. Don't overwrite published artifacts without consent.
- Artifact page sources go in `data/pages/` (git-ignored), not the scratchpad.

## Layout
| Path | What |
|---|---|
| `synth/`, `measures/` | simulator and measures (pairwise: panickssery, equal_opportunity, hspp, roytburg, yang; rubric: `measures/rubric.py`) |
| `*_sweep.py` (root) | simulator sweeps → `*_results.json` |
| `real/` | measures on real data; `real/*_results.csv` |
| `infer/` | vLLM + Aleph inference; run orders in `infer/README.md` |
| `data/`, `infer/data/` | git-ignored: cloned datasets, generated answers/verdicts. Copy between clusters with rsync |

## Clusters
`infer/clusters/<name>.sh` per cluster (picked by `$CC_CLUSTER`), `infer/clusters/local.sh` gitignored
overrides (account). `source infer/env.sh` before anything. Details: `infer/README.md` → Clusters.
- **Vulcan**: 4× L40S 48 GB; Aleph lives here; labelling and analysis run here.
- **Narval**: 4× A100 40 GB; no internet on compute nodes; 70B/72B models don't fit (stay on Vulcan).
- Don't run the same job on both clusters (two output files to merge). Sync outputs back to Vulcan.

## Findings so far
- Pairwise real data: generator pairs (not difficulty) drive the real slopes of EO/HSPP; the judge leans to
  slot A more against weaker opponents. Proposed sim fix: P(vote A) = σ(z·L + s + b_pair).
- Rubric sim: score gap always confounded by p; β confounds HSPP-Rub but not the matched measures;
  halo h confounds all, incl. matched → needs a second judge (difference-in-differences).
- IFEval rubric, Aleph judges (`real/rubric_real.py`): matched_fail > 0 for Qwen (0.31–0.33) and gpt-oss
  (0.11–0.16); Gemma's HSPP-Rub likely a p effect (matched CI includes 0); Command-R/Aya ≈ 0 (Aya says "met" to all).

## Status (update when it changes) — 2026-11-12
| Experiment | State | Next |
|---|---|---|
| IFEval rubric, Aleph (8 models) | answers, labels, verdicts complete | — |
| IFEval rubric, 6 GPU models | jobs queued on Vulcan (`ifeval_gpu_submit.sh`) | then Aleph judges rerun on new answers → `python -m real.rubric_real` |
| β calibration (`infer/beta_data.py`, `real/beta_fit.py`) | jobs queued on Vulcan (`beta_judges_submit.sh`) | `python -m real.beta_fit`; check sim slopes vs real EO 0.69 (p) / 1.31 (a) |
| Yang on Chen's data | gpt-oss-120b + OLMo scores done (OLMo weak); Command-R+ queued on Vulcan | `python -m infer.yang_data pairs` → `yang_judges_submit.sh` → `python -m real.yang_real`; decide OLMo weighting |
| Later | sim pair baseline b_pair; HealthBench rubric; reasoning on/off comparison (`infer/data/ifeval_nothink/`) | |
