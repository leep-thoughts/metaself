# scripts/: run the pending work

Run from the repo root on Vulcan. Each script submits its follow-up analysis with a Slurm
dependency, so you run it once. Step-by-step details: `infer/README.md`.

| Script | Who | What | Time | Output |
|---|---|---|---|---|
| `bash scripts/yang.sh` | **you (GPU)** | Yang pairs (CPU, built) → 11 GPU judge jobs → analysis | 1–6 h per judge | `real/yang_real_results.csv` |
| `bash scripts/healthbench_gpu.sh` | **you (GPU)** | HealthBench on 6 GPU models: generate → judge all → Aleph judges the new answers → analysis | ~1 h GPU, then Aleph ~1 day | `real/rubric_real_healthbench.csv` |
| `sbatch scripts/analyses.sbatch [groups]` | anyone (CPU) | rerun analyses: `rubric` (IFEval + halo DiD), `healthbench`, `beta`, `pairs` (pair lean, consensus errors, sim slopes), `yang` | minutes to ~1 h | `real/*_results.*`, `logs/analyses_<id>.out` |

Options
- Yang without the weak OLMo rater: `rm infer/data/yang_judge_*.jsonl; YANG_RATERS=gpt-oss-120b,command-r-plus bash scripts/yang.sh`
- Only some judges: `bash scripts/yang.sh qwen-2.5-7b gemma-2-9b`
- On Narval: the same scripts work; 70B/72B judges are skipped there (run them on Vulcan). Copy outputs back to Vulcan for the analyses.

## Moving the stuck 70B jobs to Fir (4x H100 80 GB)
| # | Where | Command |
|---|---|---|
| 1 | Vulcan | `git push` (code incl. `infer/clusters/fir.sh`) |
| 2 | Fir login | `cd $SCRATCH && git clone <repo> metaself && cd metaself`; `cp infer/clusters/local.sh.example infer/clusters/local.sh` (set ACCOUNT); copy `~/.cache/huggingface/token` |
| 3 | Fir login | `bash scripts/fir_70b.sh setup` (venv + 3 × 70B weights, hours; use tmux) |
| 3b | Vulcan | once: `bash scripts/sync_fir.sh globus-setup` (Globus CLI + `globus login`), then `export VULCAN_EP=<id>`; activate both collections once in the Globus web app |
| 4 | Vulcan | `bash scripts/sync_fir.sh push` (Pombal data, Yang pairs; Globus by default, `SYNC=rsync` for ssh) |
| 5 | Fir | `bash scripts/fir_70b.sh start` (3 Yang judges + Llama-70B HealthBench answers) |
| 6 | Vulcan | `bash scripts/vulcan_reroute.sh cancel` (stuck 70B jobs + their dependants) |
| 7 | Vulcan | after Fir's answers job: `bash scripts/sync_fir.sh pull`, then `bash scripts/vulcan_reroute.sh hb` |
| 8 | Vulcan → Fir | `bash scripts/sync_fir.sh push-hb`, then on Fir `bash scripts/fir_70b.sh hb-judge` |
| 9 | Vulcan | after Fir's jobs: `bash scripts/sync_fir.sh pull`; `sbatch scripts/analyses.sbatch yang healthbench` |
