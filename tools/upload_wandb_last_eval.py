"""
Backfill missing final eval metrics for Apollo wandb runs.

Apollo's Docker --rm flag kills the container before wandb can flush the final
eval. This script parses the actual final eval from the work_dirs log files and
logs them to wandb at summary_step+2 (summary_step+1 was already taken by a
prior failed recovery attempt that logged stale data).
"""
import json
import os
import re
import glob
import math
import wandb

WANDB_DIR = "/workspace/ForeSight/wandb"
WORK_DIRS = "/workspace/ForeSight/work_dirs"
ENTITY = "trailab"
PROJECT = "ForeSight"

# Maps wandb run ID -> config stem (work_dir name)
RUN_CONFIG_MAP = {
    "xcy3j16r": "sparsedrive_r50_stage1_8gpu_noflash_aux2d",
    "mm9iirk4": "sparsedrive_r50_stage1_8gpu_noflash_aux2p5d",
    "ft2v6y1e": "sparsedrive_r50_stage1_8gpu_noflash_dn_rot3d_aux2p5d",
    "aug7dy1d": "sparsedrive_r50_stage1_8gpu_noflash_joint",
}


def parse_final_eval_from_log(log_path):
    """Parse the last Iter(val) line from a mmdet log and return metric dict."""
    last_val_line = None
    with open(log_path) as f:
        for line in f:
            if "Iter(val)" in line:
                last_val_line = line

    if last_val_line is None:
        return None

    metrics = {}
    # Parse "key: value" pairs from the log line
    pairs = re.findall(r"([\w/]+):\s*([-\d.]+|nan)", last_val_line)
    for key, val in pairs:
        try:
            v = float(val)
            if not math.isnan(v):
                metrics[f"val/{key}"] = v
        except ValueError:
            pass

    # Also parse the summary lines for mAP/NDS/mAP_normal/ped_crossing etc.
    # These appear just before Iter(val) as plain "key= value" lines
    with open(log_path) as f:
        content = f.read()
    for key in ["mAP", "NDS", "mAP_normal", "ped_crossing", "divider", "boundary"]:
        matches = re.findall(rf"{key}[=:]\s*([0-9.]+)", content)
        if matches:
            metrics[f"val/{key}"] = float(matches[-1])

    return metrics if metrics else None


def get_remote_val_steps(api, run_id):
    """Return list of steps that have val metrics in wandb history."""
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        history = run.history(keys=["val/img_bbox_NuScenes/NDS"], pandas=False)
        return [r["_step"] for r in history if r.get("val/img_bbox_NuScenes/NDS") is not None]
    except Exception as e:
        if "not found" in str(e).lower():
            return None  # run doesn't exist
        print(f"  [WARN] {run_id}: {e}")
        return []


def get_summary_step(run_id):
    """Get _step from the run's wandb-summary.json."""
    for summary_path in sorted(glob.glob(f"{WANDB_DIR}/run-*-{run_id}/files/wandb-summary.json")):
        with open(summary_path) as f:
            return json.load(f).get("_step", -1)
    return -1


def main():
    api = wandb.Api()
    failed = []

    for run_id, stem in RUN_CONFIG_MAP.items():
        print(f"\n--- {run_id} ({stem}) ---")

        val_steps = get_remote_val_steps(api, run_id)
        if val_steps is None:
            print(f"  [SKIP] run not found in wandb")
            continue

        summary_step = get_summary_step(run_id)
        # step summary_step+1 has stale data from prior attempt; use +2
        target_step = summary_step + 2

        if target_step in val_steps or (summary_step + 1) in val_steps:
            # Check if the step+2 value differs from step+1 (i.e. already correct)
            correct_steps = [s for s in val_steps if s >= summary_step]
            if len(correct_steps) >= 2:
                print(f"  [OK] already has correct eval at steps {correct_steps}")
                continue

        # Find log file
        log_files = sorted(glob.glob(f"{WORK_DIRS}/{stem}/*.log"))
        if not log_files:
            print(f"  [SKIP] no log file found in work_dirs/{stem}/")
            continue
        log_path = log_files[-1]

        # Parse metrics
        metrics = parse_final_eval_from_log(log_path)
        if not metrics:
            print(f"  [SKIP] no Iter(val) line found in {log_path}")
            continue

        nds = metrics.get("val/img_bbox_NuScenes/NDS") or metrics.get("val/NDS", "?")
        print(f"  Parsed NDS={nds} from log, logging at step {target_step}...")

        try:
            run = wandb.init(
                project=PROJECT,
                entity=ENTITY,
                id=run_id,
                resume="must",
                settings=wandb.Settings(silent=False),
            )
            wandb.log(metrics, step=target_step, commit=True)
            wandb.finish()
            print(f"  [DONE]")
        except Exception as e:
            print(f"  [FAIL]: {e}")
            failed.append(run_id)
            try:
                wandb.finish()
            except Exception:
                pass

    print(f"\nDone. Failed: {failed if failed else 'none'}")


if __name__ == "__main__":
    main()
