---
name: run-experiments
description: Plans, creates, submits, monitors, and logs a small ForeSight experiment batch after surveying prior results and waiting at two confirmation checkpoints.
allowed-tools: Read, Write, Edit, Bash(git:*), Bash(ssh:*), Bash(cp:*), Bash(date:*), Bash(cat:*), Bash(head:*), Bash(tail:*), Bash(ls:*)
---

Goal: `$ARGUMENTS`

Default assumptions unless the goal says otherwise:
- server: `narval`
- base config: `projects/configs/sparsedrive_r50_stage2_4gpu_bs24.py`
- batch size: 1 to 3 independent experiments

All experiments must serve the stated goal and target the primary planning metrics first.

## Phase 1: Survey
1. Read only the architecture files relevant to the subsystem touched by the goal. Use the project architecture notes to find the right file.
2. Read:
```bash
cat reports/2026_02_16_baselines.md
cat reports/0000_00_00_research_findings.md
cat reports/0000_00_00_research_directions.md
ls -1 reports/
```
3. Read only the report files that are relevant to the goal.
4. Extract:
   - baseline metric values to compare against
   - ideas already ruled out
   - promising directions already supported by evidence

## Phase 2: Propose
Write:

```text
Problem: <one sentence>
Approach: <one sentence>
```

Then design 1 to 3 experiments in this format:

| # | Config suffix | Change | Mechanism | Expected effect |
|---|---------------|--------|-----------|-----------------|
| 1 | `<suffix>` | `<old> -> <new>` | `<why>` | `<metric direction>` |

Rules:
- Each experiment tests one hypothesis.
- Experiments are independent.
- Do not re-run ideas already ruled out in `reports/0000_00_00_research_findings.md`.
- Prefer directions already motivated in `reports/0000_00_00_research_directions.md` unless the goal explicitly asks for something novel.

### Checkpoint 1
Print the problem statement, approach, and full experiment table. Ask exactly:
`Does this look correct? Reply 'yes' to proceed or give feedback to revise.`

Do not continue until the user confirms.

## Phase 3: Create
1. Derive `<feature>` as a short snake_case label and work on branch `sd_<feature>`, unless the goal specifies a branch name.
2. Create `reports/<YYYY_MM_DD>_<feature>.md` with:

```markdown
# <feature> - <YYYY-MM-DD>

## Intro

## Method

## Results

| # | Config | L2 | obj_box_col | car_ade | NDS | Status | Notes | Job ID |
|---|--------|----|-------------|---------|-----|--------|-------|--------|

## Discussion

## Future Work
```

3. For each experiment:
   - copy the base config to `projects/configs/<config_stem>.py`
   - edit only the lines that must differ
   - set the WandB run name to `<config_stem>`
4. Re-read every new config and confirm the intended changes only.

### Checkpoint 2
Summarize the code and config changes. Ask exactly:
`Ready to commit and push? Reply 'yes' to proceed or give feedback to revise.`

Do not commit or push until the user confirms.

5. Commit, push, and sync the target server:
```bash
git add projects/configs/<config_stem_1>.py projects/configs/<config_stem_2>.py ... <changed source files>
git commit -m "sd_<feature>: add experiment configs

<brief description>"
git push -u origin sd_<feature>
ssh <server> "source ~/.bashrc && cd /home/spapais/ForeSight && git fetch origin sd_<feature> && git checkout sd_<feature>"
```

## Phase 4: Submit
Submit every config with `/submit-job --server <server> --config projects/configs/<config_stem>.py`.

Record each returned job ID in the report table immediately.

## Phase 5: Poll and Triage
Use `/loop` to poll roughly hourly until every job is resolved.

State handling:
- `COMPLETED`: mark done
- `FAILED`: inspect the log, fix the root cause, resubmit once
- `TIMEOUT`: treat like `FAILED`
- `CANCELLED`: mark as crash and do not resubmit

For failed jobs, inspect:
```bash
ssh <server> "tail -80 /home/spapais/ForeSight/logs/foresight-<JOB_ID>.log"
```

If a non-trivial fix is required, commit and push the fix before resubmitting. Never resubmit the same experiment more than once.

## Phase 6: Log Results
1. Run `/parse-metrics --server <server> --job <JOB_ID> --config <config_stem>` for every completed experiment.
2. Update the report table for all experiments, including crashes.
3. Write `## Discussion` and `## Future Work`.
4. Update both summary files:
   - `reports/0000_00_00_research_findings.md`
   - `reports/0000_00_00_research_directions.md`
5. Commit the logging updates:
```bash
git add <report> reports/0000_00_00_research_findings.md reports/0000_00_00_research_directions.md
git commit -m "sd_<feature>: log results"
git push
```
6. Print a concise experiment summary table and one-sentence conclusion.

## Rules
- Ask for confirmation only at the two checkpoints above.
- Never discard branches because results are poor.
- Keep all session logging in `reports/<YYYY_MM_DD>_<feature>.md`.
