---
name: survey-results
description: Survey recently completed jobs across all servers, extract metrics, update the relevant reports, then report findings + interpretation + suggested future work in the conversation. Use when the user asks for a status update with results pulled and reports written, e.g. "check the recent runs and update the reports".
---

## Inputs
Parse from: `$ARGUMENTS`
- `--server` (default: `all`): `dgx` | `apollo` | `narval` | `trillium` | `killarney` | `all`
- `--ended-since` (default: 24h ago): ISO timestamp; only jobs whose End falls on or after this are reported.

Use the host and VPN rules from `.claude/CLAUDE.md`. Skip Apollo from the SLURM survey — its training runs in tmux/docker, not SLURM. For Apollo, capture the tmux pane and read the latest `Iter` log.

## Step 1 — Survey what completed

Filter by **End time**, not start time. Some training runs go up to ~36 h (resumes especially), so a `-S 24h` start filter would silently drop them. Use a wide `-S` window (7 days, safely covers any active run length) and post-filter the End column with awk against `--ended-since`.

For each SLURM server:

```bash
ssh trail_dgx "source ~/.bashrc && squeue -u spapais --format='%.10i %.20j %.8T %.10M %.10l %.20b'; echo === ; \
  sacct -u spapais -S \$(date -d '7 days ago' +%Y-%m-%dT%H:%M:%S) \
    --format=JobID,JobName,State,Elapsed,End -X --parsable2 -n 2>/dev/null \
  | awk -F'|' -v cutoff='<ended_since>' '\$5 != \"Unknown\" && \$5 >= cutoff'"
```

Same pattern for `narval`, `trillium_gpu`, and `killarney` (Killarney needs `source /etc/profile.d/modules.sh && module load slurm/killarney/25.05.6`).

The awk filter keeps any row whose `End` is on or after `<ended_since>`, regardless of when the job started — so a 36 h run that started 2 days ago and ended 6 h ago will show up under a 24 h end-time lookback.

Apollo:
```bash
ssh apollo "tmux ls; tmux capture-pane -t <session> -p | tail -3"
```

## Step 2 — Identify new completed jobs

Compare COMPLETED jobs (state=COMPLETED, elapsed > a few minutes — skip the trivial CANCELLED/sub-1-min jobs) against what's already mentioned in `reports/2026_04_28_paper_plan.md` and `reports/2026_04_28_results_summary.md`. Anything not yet logged is new.

For pending or short eval batches that match an existing report's experiment family, group them as "in-flight" or "follow-up" rather than treating each as a new headline result.

## Step 3 — Extract config + metrics for each new completed job

```bash
ssh <server> "grep -m1 -oE 'projects/configs/[^ ]+\\.py' /scratch/spapais/ForeSight/logs/foresight-<JOBID>.log"
ssh <server> "grep -E 'val/L2|val/obj_box_col|val/car_EPA|val/car_min_ade_err|val/pedestrian_EPA|val/pedestrian_min_ade_err|val/img_bbox_NuScenes/NDS|val/img_bbox_NuScenes/mAP$|val/mAP_normal' /scratch/spapais/ForeSight/logs/foresight-<JOBID>.log | grep -v '▁' | tail -10"
```

Report only the metrics actually present in the log. Missing entries should be `-`, not made-up. Cross-reference DGX evals using `scontrol show job <id>` (the Command field includes the eval ckpt path).

## Step 4 — Update reports

Pick the right report per result:

| Result family | Report |
|---|---|
| New stage-2 training variant on a tracked baseline | `reports/2026_04_28_results_summary.md` (add row to the matching subsection) + `reports/2026_04_28_paper_plan.md` (mark TODO done with one-line metric summary) |
| Rescore / scoring variants | `reports/2026_04_26_plan_scoring.md` (add an Exp section) + paper_plan TODO |
| Stage-1 training | `reports/2026_04_28_paper_plan.md` |
| K/V-channel diagnostics | `reports/2026_04_26_plan_gaps.md` |
| Decoder scaling | `reports/2026_04_23_decoder_scaling.md` |

Rules:
- Edit existing tables in place. Don't append override blocks at the end.
- Add a "Confirmed Findings" line in `results_summary.md` when the result is decisive (clear win, clean negative, or surfaces a new mechanism). Skip findings updates for runs that just confirm an existing claim.
- Reference job IDs in parentheses so the reader can dig into the log.
- Don't bundle unrelated user WIP into your edits — if the user's working tree has unrelated pending changes (decoder.py, plan_research.md, etc.), restrict edits to the specific files you're touching for this survey.

## Step 5 — Report in conversation

Format:

```
## Status

| Server | Job | Config | State | Headline metric |
|---|---|---|---|---|
...

## Reports updated

- file_a.md — ...
- file_b.md — ...

## Interpretation

(2-4 bullets, mechanism-focused. What did each new result tell us about the working hypothesis? Where does it stack with prior findings? What does it rule in / rule out?)

## Future work

(2-3 concrete next steps. Be specific: which config to write, which baseline to compare against, which knob to sweep. Don't list everything that *could* be done — list what *should* be done given the new evidence.)
```

## Things to avoid

- Don't trigger pulls of `work_dirs/` or large rsyncs from inside this skill — `sync-results` exists for that. This skill is for *reading* results from the remote logs.
- Don't commit or push report updates without explicit user confirmation. Stage edits, show the diff, ask.
- Don't overstate noise-floor effects. Single-run deltas under ~0.007 L2 / ~0.015 pp CR / ~0.012 cross-cluster are noise (see Run-to-Run Noise table in `results_summary.md`).
- Don't speculate beyond what's in the logs. If a metric is missing or a job is still running, say so.
