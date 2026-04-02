---
name: run-experiments
description: One-shot ML experiment batch for ForeSight — survey prior work, propose a batch of experiments, create all configs, queue all jobs on SLURM, then log all results. Two mandatory confirmation checkpoints before any compute is used.
allowed-tools: Read, Write, Edit, Bash(git:*), Bash(ssh:*), Bash(cp:*), Bash(date:*), Bash(cat:*), Bash(head:*), Bash(tail:*), Bash(ls:*)
---

Arguments: $ARGUMENTS

| Arg | Required | Default |
|-----|----------|---------|
| `--goal` | yes | — |
| `--base-config` | no | `projects/configs/sparsedrive_r50_stage2_4gpu.py` |
| `--server` | no | `narval` |

**Objective:** All experiments must serve `--goal` and improve the primary metrics. See CLAUDE.md for full metric definitions.

---

## Phase 1 — Survey

### 1. Read the base config

```bash
cat <base-config>
```

Record current values of: `total_batch_size`, `num_epochs`, `with_map`, `queue_length`, `num_det`, `num_decoder`, all loss weights, backbone `type`, and LR schedule.

### 2. Read the model architecture

If `--goal` involves a specific component (detection, map, motion, or planning), read the relevant source file. See CLAUDE.md for the key source files table.

### 3. Read prior research findings

**Always** read the baseline performance report:
```bash
cat reports/2026_02_16_baselines.md
```

Extract the specific metric values for each experiment (use these as your baseline reference).

Then list and read all reports relevant to `--goal`:

```bash
ls -1 reports/
```

From each report, extract: which changes worked and which did not, and any failure modes documented.

---

## Phase 2 — Propose

### 1. State the problem

> **Problem:** `<one sentence — what is underperforming or missing functionality>`
> **Approach:** `<one sentence — the improvements these experiments will test>`

### 2. Propose all experiments

Use the $ARGUMENTS to design 1–3 experiments. Prefer fewer — one experiment is enough when the hypothesis is sharp. Add more only when sweeping a parameter range or comparing competing approaches. Each experiment must test one hypothesis with config changes, new components, or new loss formulations:

| # | Config suffix | Change | Mechanism | Expected effect |
|---|--------------|--------|-----------|-----------------|
| 1 | `<suffix_1>` | `<change>: <old> → <new>` | `<why this helps>` | `<metric direction>` |
| 2 | `<suffix_2>` | ... | ... | ... |

Rules:
- Experiments are independent — no experiment depends on the result of another in this batch
- Do not propose ideas already in `reports/0000_00_00_research_review.md` Confirmed Findings or Avoid lists

---

**CHECKPOINT 1:** Print the problem statement, planned solution, and the full experiment table. Ask: "Does this look correct? Reply 'yes' to proceed or give feedback to revise." Do not proceed until confirmed.

---

## Phase 3 — Create

### 1. Create the session branch

Derive `<feature>` from `--goal` as a short snake_case label (e.g. `planning_loss`, `temporal_queue`).

```bash
git checkout -b sd_<feature>
```

### 2. Create the session report

```bash
date +%Y_%m_%d
```

Create `reports/<YYYY_MM_DD>_<feature>.md` (referred to as `<report>` throughout):

```markdown
# <feature> — <YYYY-MM-DD>

## Intro
Describe the problem and the experiments.

## Method
Describe the method used to test the hypotheses.

## Results

| # | Config | L2 | obj_box_col | car_ade | NDS | Status | Notes | Job ID |
|---|--------|----|-------------|---------|-----|--------|-------|--------|

## Discussion
_(filled at the end)_

## Future Work
_(filled at the end)_
```

### 3. Create all configs and any code changes needed

`<config_stem>` = `<base_config_stem>_<suffix>` where `<base_config_stem>` is the filename stem of `<base-config>` and `<suffix>` matches the suffix column from the proposal table.

For **each** experiment:
1. Copy the base config: `cp <base-config> projects/configs/<config_stem>.py`
2. Use the **Edit tool** to make each targeted change — edit only the lines that differ from the base config
3. Update the WandB run name: find the `name=` line inside `log_config` and replace it with `'<config_stem>'`

After editing all configs, re-read each one to confirm every change is correct and no unintended lines were modified.

---

**CHECKPOINT 2:** Summarize the code changes. Ask: "Ready to commit and push? Reply 'yes' to proceed or give feedback to revise." Do not commit or push until confirmed.

---

### 4. Commit, push, and pull on server

```bash
git add projects/configs/<config_stem_1>.py projects/configs/<config_stem_2>.py ... <any changed source files>
git commit -m "sd_<feature>: add experiment configs

<brief description of what is being tested>"
git push -u origin sd_<feature>
ssh <server> "source ~/.bashrc && cd /home/spapais/ForeSight && git fetch origin sd_<feature> && git checkout sd_<feature>"
```

---

## Phase 4 — Submit

Submit all jobs using the `/submit-job` skill:

```
/submit-job --server <server> --config projects/configs/<config_stem_1>.py
/submit-job --server <server> --config projects/configs/<config_stem_2>.py
```

Record all returned job IDs in the **Results** table in `<report>`.

---

## Phase 5 — Poll & Triage

Use the `/loop` skill to poll at a 1h interval:

```
/loop 1h Check SLURM statuses for jobs <JOB_IDS> on <server>: run squeue and sacct, triage any newly finished or failed jobs per the Phase 5 triage table, then if all jobs are resolved exit the loop and proceed to Phase 6 of the autoresearch2 skill for branch sd_<feature>.
```

**Triage table:**

| `sacct` State | Action |
|---------------|--------|
| `COMPLETED` | Mark done. |
| `FAILED` | Diagnose and fix (see below). |
| `TIMEOUT` | Treat as FAILED — tail log, fix if possible, resubmit once. |
| `CANCELLED` | Mark as `crash`, no resubmit. |

**For every `FAILED` or `TIMEOUT` job**, tail the log to find the cause:

```bash
ssh <server> "tail -80 /home/spapais/ForeSight/logs/foresight-<JOB_ID>.log"
```

Fix the root cause, then resubmit:

```
/submit-job --server <server> --config projects/configs/<config_stem>.py
```

Each job may be resubmitted **at most once**. If a resubmitted job also fails, mark it as `crash` and move on.

Update the Results table in `<report>` with the new job ID. If the fix requires a non-trivial code change, commit it before resubmitting:

```bash
git add <changed files>
git commit -m "sd_<feature>: fix <config_stem> (<one-line error>)"
git push && ssh <server> "source ~/.bashrc && cd /home/spapais/ForeSight && git pull"
```

---

## Phase 6 — Log Results

Process **all** experiments — do not skip any, even if they crashed.

### 1. Parse metrics for each experiment

```
/parse-metrics --server <server> --job <JOB_ID> --config <config_stem>
```

If a job was marked `crash` during Phase 5, skip metric parsing and use the error summary already recorded in `<report>`.

### 2. Update `<report>` for each experiment

Fill in the metrics for the experiment in the Results table.

### 3. Fill `## Discussion` and `## Future Work` in `<report>`

Write a cross-experiment summary: which hypotheses were confirmed, which were not, and recommended next steps.

### 4. Update `reports/0000_00_00_research_review.md`

- Add new entries to **Confirmed Findings** for results that extend what is known.
- Update **Promising Directions** "Highest priority" block if a new best was found.
- Add any newly confirmed failure modes to **Avoid / treat carefully**.
- Update the `Last updated:` date line.

### 5. Commit all logs

```bash
git add <report> reports/0000_00_00_research_review.md
git commit -m "sd_<feature>: log results"
git push
```

### 6. Print summary

Print a concise table of all experiments with their key metrics and status. State the overall conclusion in one sentence.

---

## Rules
- Never ask for confirmation outside of the two designated checkpoints
- Never git reset after a bad result — keep all branches regardless of outcome
- All session logging goes into `<report>` (`reports/<YYYY_MM_DD>_<feature>.md`)
- Each experiment must test a single independent hypothesis
