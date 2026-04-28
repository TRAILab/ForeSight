#!/usr/bin/env python3
"""GPU queue table, plot, and probe submission for Narval/Trillium/Killarney.

Default: refresh the unified cluster table + queue-time plot from live SLURM
accounting, print a recommendation, then submit 4-GPU no-op probe jobs (fire
and forget; results show up on the next run via sacct).
  --report   only refresh the table/plot/recommendation (skip probe submission)
  --probe    only submit probes (skip table/plot/recommendation)
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CLUSTERS = ("narval", "trillium", "killarney")
SECONDS_PER_DAY = 24 * 60 * 60
SACCT_FIELDS = (
    "JobIDRaw",
    "Submit",
    "Start",
    "Timelimit",
    "ReqTRES",
    "AllocTRES",
)
CLUSTER_INFO: dict[str, dict[str, str]] = {
    "narval": {
        "host": "narval",
        "init": "source ~/.bashrc",
        "node_type": "4xA100-80GB",
        "partition": "gpubase_bygpu_b1",
        "account": "rrg-swasland_gpu",
        "sbatch_opts": "--account=rrg-swasland --ntasks=1 --cpus-per-task=12 --mem=120gb --time=11:59:00 --gres=gpu:a100:4",
        "probe_cwd": "",
    },
    "trillium": {
        "host": "trillium_gpu",
        "init": "source ~/.bashrc",
        "node_type": "4xH100-80GB",
        "partition": "compute",
        "account": "rrg-swasland",
        "sbatch_opts": "--account=rrg-swasland --ntasks=1 --cpus-per-task=24 --time=11:59:00 --gpus-per-node=4",
        "probe_cwd": "",
    },
    "killarney": {
        "host": "killarney",
        "init": "source /etc/profile.d/modules.sh && module load slurm/killarney/24.05.7",
        "node_type": "4xL40S-48GB",
        "partition": "gpubase_l40s_b1",
        "account": "",
        "sbatch_opts": "--account=aip-swasland --ntasks=1 --cpus-per-task=16 --mem=120gb --time=11:59:00 --gres=gpu:l40s:4",
        "probe_cwd": "/scratch",
    },
}
SNAPSHOT_TIMEOUT = 45
SUBMIT_TIMEOUT = 15
GPU_TRES_RE = re.compile(r"^gres/gpu(?::[^=,]+)?=(\d+)$")
PERCENTILE_LOW = 25.0
PERCENTILE_HIGH = 75.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh GPU queue table/plot and submit probe jobs.",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=7,
        help="display window in days for plot/CSV (default: 7)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="print the full daily-median series instead of only the latest value",
    )
    parser.add_argument(
        "--start-time",
        default=os.environ.get("START_TIME", "1970-01-01"),
        help="SLURM accounting start time (default: START_TIME env or 1970-01-01)",
    )
    parser.add_argument(
        "--end-time",
        default=os.environ.get("END_TIME", "now"),
        help="SLURM accounting end time (default: END_TIME env or now)",
    )
    parser.add_argument(
        "--slurm-user",
        default=os.environ.get("SLURM_USER", "spapais"),
        help="SLURM user to query (default: SLURM_USER env or spapais)",
    )
    parser.add_argument(
        "--time-limit",
        default=os.environ.get("TIME_LIMIT", "11:59:00"),
        help="filter to this SLURM time limit (default: TIME_LIMIT env or 11:59:00)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=int(os.environ.get("GPUS", "4")),
        help="filter to jobs requesting this GPU count (default: GPUS env or 4)",
    )
    parser.add_argument(
        "--query-timeout",
        type=int,
        default=int(os.environ.get("QUERY_TIMEOUT", "120")),
        help="per-cluster SLURM accounting timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=REPO / "reports" / "queue_time_plot.png",
        help="path for the generated queue-time plot (default: reports/queue_time_plot.png)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip plot generation",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--report",
        action="store_true",
        help="only refresh table/plot/recommendation; skip probe submission",
    )
    mode.add_argument(
        "--probe",
        action="store_true",
        help="only submit probes; skip table/plot/recommendation",
    )
    return parser.parse_args()


def format_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "N/A"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def parse_slurm_time(value: str) -> int | None:
    if value in ("", "Unknown", "None"):
        return None
    normalized = value.replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(normalized, fmt).timestamp())
        except ValueError:
            continue
    try:
        return int(datetime.fromisoformat(value).timestamp())
    except ValueError:
        return None


def has_gpu_count(req_tres: str, alloc_tres: str, gpus: int) -> bool:
    for tres in (req_tres, alloc_tres):
        for item in tres.split(","):
            match = GPU_TRES_RE.match(item)
            if match and int(match.group(1)) == gpus:
                return True
    return False


def query_cluster(
    cluster: str,
    host: str,
    init: str,
    start_time: str,
    end_time: str,
    slurm_user: str,
    time_limit: str,
    gpus: int,
    timeout: int,
) -> list[dict[str, object]]:
    sacct_format = ",".join(SACCT_FIELDS)
    remote_cmd = (
        f"{init} && sacct -u {shlex.quote(slurm_user)} -X --parsable2 --noheader "
        f"--starttime {shlex.quote(start_time)} --endtime {shlex.quote(end_time)} "
        f"--format={shlex.quote(sacct_format)}"
    )
    result = subprocess.run(
        ["ssh", host, remote_cmd],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{cluster}: query failed: {stderr}")

    rows: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) != len(SACCT_FIELDS):
            continue
        job_id, submit, start, row_time_limit, req_tres, alloc_tres = fields
        if row_time_limit != time_limit:
            continue
        if not has_gpu_count(req_tres, alloc_tres, gpus):
            continue

        submit_epoch = parse_slurm_time(submit)
        start_epoch = parse_slurm_time(start)
        if submit_epoch is None or start_epoch is None:
            continue

        rows.append(
            {
                "cluster": cluster,
                "job_id": job_id,
                "submit_epoch": submit_epoch,
                "queue_seconds": max(0, start_epoch - submit_epoch),
            }
        )
    return rows


def query_all_clusters(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(CLUSTER_INFO)) as executor:
        futures = {
            executor.submit(
                query_cluster,
                cluster,
                info["host"],
                info["init"],
                args.start_time,
                args.end_time,
                args.slurm_user,
                args.time_limit,
                args.gpus,
                args.query_timeout,
            ): cluster
            for cluster, info in CLUSTER_INFO.items()
        }
        for future in as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception as exc:
                failures.append(str(exc))
    rows.sort(key=lambda row: (str(row["cluster"]), int(row["submit_epoch"]), str(row["job_id"])))
    return rows, failures


def query_snapshot(cluster: str, info: dict[str, str], timeout: int) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "idle": None, "mix": None, "alloc": None, "down": None,
        "pending": None, "levelfs": None,
    }
    partition = shlex.quote(info["partition"])
    parts = [
        info["init"],
        f'sinfo -p {partition} --noheader -o "%n %t" 2>/dev/null | sort -k1,1 -u | '
        'awk \'{st=$2; gsub(/[^a-z]/,"",st); '
        'if(st=="idle") i++; else if(st=="mix") m++; else if(st=="alloc") a++; else d++} '
        'END{printf "STATES %d %d %d %d\\n", i+0, m+0, a+0, d+0}\'',
        f'echo "PENDING $(squeue -p {partition} -t PENDING --noheader 2>/dev/null | wc -l)"',
    ]
    if info["account"]:
        account = shlex.quote(info["account"])
        parts.append(
            f'LFS=$(sshare -l -A {account} --parsable2 --noheader 2>/dev/null | '
            'awk -F"|" \'$2==""{print $9; exit}\'); '
            'echo "LEVELFS ${LFS:-N/A}"'
        )
    else:
        parts.append('echo "LEVELFS N/A"')
    remote_cmd = "; ".join(parts)
    try:
        result = subprocess.run(
            ["ssh", info["host"], remote_cmd],
            check=False, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return snapshot
    if result.returncode != 0:
        return snapshot
    for line in result.stdout.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "STATES" and len(tokens) == 5:
            try:
                snapshot["idle"] = int(tokens[1])
                snapshot["mix"] = int(tokens[2])
                snapshot["alloc"] = int(tokens[3])
                snapshot["down"] = int(tokens[4])
            except ValueError:
                pass
        elif tokens[0] == "PENDING" and len(tokens) == 2:
            try:
                snapshot["pending"] = int(tokens[1])
            except ValueError:
                pass
        elif tokens[0] == "LEVELFS" and len(tokens) == 2 and tokens[1] != "N/A":
            try:
                snapshot["levelfs"] = float(tokens[1])
            except ValueError:
                pass
    return snapshot


def query_all_snapshots(timeout: int) -> dict[str, dict[str, object]]:
    snapshots: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=len(CLUSTER_INFO)) as executor:
        futures = {
            executor.submit(query_snapshot, cluster, info, timeout): cluster
            for cluster, info in CLUSTER_INFO.items()
        }
        for future in as_completed(futures):
            cluster = futures[future]
            try:
                snapshots[cluster] = future.result()
            except Exception:
                snapshots[cluster] = {
                    "idle": None, "mix": None, "alloc": None, "down": None,
                    "pending": None, "levelfs": None,
                }
    return snapshots


def percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(sorted_values[lo])
    frac = rank - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def latest_epoch(rows: list[dict[str, object]]) -> int | None:
    if not rows:
        return None
    return max(int(row["submit_epoch"]) for row in rows)


def day_start(epoch: int) -> int:
    dt = datetime.fromtimestamp(epoch)
    return int(datetime(dt.year, dt.month, dt.day).timestamp())


def day_label(epoch: int) -> datetime:
    return datetime.fromtimestamp(epoch)


def build_daily_series(
    rows: list[dict[str, object]], window_days: int
) -> dict[str, list[dict[str, object]]]:
    if not rows:
        return {cluster: [] for cluster in DEFAULT_CLUSTERS}

    earliest_day = day_start(min(int(row["submit_epoch"]) for row in rows))
    max_day = day_start(max(int(row["submit_epoch"]) for row in rows))
    plot_start_day = max(earliest_day, max_day - (window_days - 1) * SECONDS_PER_DAY)
    values_by_cluster_day: dict[str, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        values_by_cluster_day[str(row["cluster"])][day_start(int(row["submit_epoch"]))].append(
            int(row["queue_seconds"])
        )

    series: dict[str, list[dict[str, object]]] = {}
    for cluster in DEFAULT_CLUSTERS:
        per_day = values_by_cluster_day.get(cluster, {})
        cluster_series: list[dict[str, object]] = []
        day = plot_start_day
        while day <= max_day:
            day_values = per_day.get(day, [])
            cluster_series.append(
                {
                    "day": day,
                    "daily_median": percentile(day_values, 50.0),
                    "daily_low": percentile(day_values, PERCENTILE_LOW),
                    "daily_high": percentile(day_values, PERCENTILE_HIGH),
                    "samples": len(day_values),
                }
            )
            day += SECONDS_PER_DAY
        series[cluster] = cluster_series
    return series


def _latest_per_cluster(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    by_cluster: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_cluster[str(row["cluster"])].append(row)
    latest: dict[str, dict[str, object]] = {}
    for cluster, cluster_rows in by_cluster.items():
        if cluster_rows:
            latest[cluster] = max(cluster_rows, key=lambda r: int(r["submit_epoch"]))
    return latest


def print_unified_table(
    rows: list[dict[str, object]],
    snapshots: dict[str, dict[str, object]],
) -> None:
    now = int(datetime.now().timestamp())
    latest = _latest_per_cluster(rows)
    headers = [
        "Cluster", "Node Type", "Nodes", "Pending",
        "Jobs/Node", "LevelFS", "Queue Time", "Last Probed",
    ]
    widths = [11, 14, 7, 9, 11, 9, 14, 13]

    def fmt_row(cells: list[str]) -> str:
        return "".join(f"{cell:<{widths[i]}}" for i, cell in enumerate(cells))

    print(fmt_row(headers))
    for cluster in DEFAULT_CLUSTERS:
        info = CLUSTER_INFO[cluster]
        snap = snapshots.get(cluster, {})
        idle, mix, alloc = snap.get("idle"), snap.get("mix"), snap.get("alloc")
        if None not in (idle, mix, alloc):
            active: int | None = int(idle) + int(mix) + int(alloc)  # type: ignore[arg-type]
            active_str = str(active)
        else:
            active = None
            active_str = "N/A"
        pending = snap.get("pending")
        pending_str = "N/A" if pending is None else str(pending)
        if active is not None and active > 0 and pending is not None:
            ratio_str = f"{int(pending) / active:.2f}"
        else:
            ratio_str = "N/A"
        lfs = snap.get("levelfs")
        if lfs is None:
            lfs_str = "1.000" if cluster == "killarney" else "N/A"
        else:
            lfs_str = f"{float(lfs):.3f}"
        if cluster in latest:
            row = latest[cluster]
            queue_str = format_seconds(int(row["queue_seconds"]))
            hrs_ago = (now - int(row["submit_epoch"])) / 3600
            probed_str = f"{hrs_ago:.1f}h ago"
        else:
            queue_str = "-"
            probed_str = "-"
        print(fmt_row([
            cluster, info["node_type"], active_str, pending_str,
            ratio_str, lfs_str, queue_str, probed_str,
        ]))


def print_recommendation(rows: list[dict[str, object]]) -> None:
    latest = _latest_per_cluster(rows)
    best: str | None = None
    best_qs: int | None = None
    for cluster in DEFAULT_CLUSTERS:
        if cluster not in latest:
            continue
        qs = int(latest[cluster]["queue_seconds"])
        if best_qs is None or qs < best_qs:
            best, best_qs = cluster, qs
    if best is None or best_qs is None:
        print("Recommendation: no probe history yet.")
    else:
        print(f"Recommendation: submit to {best} (last queue {format_seconds(best_qs)})")


def print_all(series: dict[str, list[dict[str, object]]], window_days: int) -> None:
    low_label = f"daily_p{int(PERCENTILE_LOW)}_seconds"
    high_label = f"daily_p{int(PERCENTILE_HIGH)}_seconds"
    print(f"cluster,day,daily_median_seconds,{low_label},{high_label},samples")

    def fmt(value: object) -> str:
        return "" if value is None else str(int(round(float(value))))

    for cluster in DEFAULT_CLUSTERS:
        for row in series.get(cluster, []):
            print(
                f"{cluster},{day_label(int(row['day'])).date()},"
                f"{fmt(row['daily_median'])},"
                f"{fmt(row['daily_low'])},{fmt(row['daily_high'])},"
                f"{row['samples']}"
            )


def _to_floats(values: list[object]) -> list[float]:
    return [float("nan") if v is None else float(v) for v in values]


def _hours_label(time_limit: str) -> str:
    try:
        hms = time_limit
        day_hours = 0
        if "-" in hms:
            days, hms = hms.split("-", 1)
            day_hours = int(days) * 24
        parts = hms.split(":")
        h = int(parts[0]) if len(parts) > 0 and parts[0] else 0
        m = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        s = int(parts[2]) if len(parts) > 2 and parts[2] else 0
        total_hours = day_hours + h + m / 60 + s / 3600
        return f"{round(total_hours)}-hr"
    except (ValueError, IndexError):
        return time_limit


def plot_series(
    series: dict[str, list[dict[str, object]]],
    rows: list[dict[str, object]],
    plot_path: Path,
    window_days: int,
    time_limit: str,
    gpus: int,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    colors = {
        "narval": "#1f77b4",
        "trillium": "#2ca02c",
        "killarney": "#d62728",
    }
    cluster_labels = {
        "narval": f"Narval {gpus}xA100",
        "trillium": f"Trillium {gpus}xH100",
        "killarney": f"Killarney {gpus}xL40S",
    }
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_trend, ax_dist) = plt.subplots(
        1,
        2,
        figsize=(18, 7),
        sharey=True,
        gridspec_kw={"width_ratios": [3, 1]},
    )

    latest = latest_epoch(rows)
    cutoff = None if latest is None else latest - window_days * SECONDS_PER_DAY
    cluster_window_values: dict[str, list[int]] = {}
    for cluster in DEFAULT_CLUSTERS:
        cluster_window_values[cluster] = [
            int(row["queue_seconds"])
            for row in rows
            if str(row["cluster"]) == cluster
            and (cutoff is None or int(row["submit_epoch"]) >= cutoff)
        ]

    cap_hours = 5
    cap = cap_hours * 3600

    for cluster in DEFAULT_CLUSTERS:
        cluster_series = series.get(cluster, [])
        if not cluster_series:
            continue
        days = [day_label(int(row["day"])) for row in cluster_series]
        daily_median = _to_floats([row["daily_median"] for row in cluster_series])
        daily_low = _to_floats([row["daily_low"] for row in cluster_series])
        daily_high = _to_floats([row["daily_high"] for row in cluster_series])
        color = colors[cluster]
        display_name = cluster_labels[cluster]
        short_name = display_name.split()[0]

        ax_trend.fill_between(
            days,
            daily_low,
            daily_high,
            color=color,
            alpha=0.18,
            linewidth=0,
            label=f"{short_name} IQR",
        )
        ax_trend.plot(
            days,
            daily_low,
            color=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.85,
        )
        ax_trend.plot(
            days,
            daily_high,
            color=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.85,
        )
        ax_trend.plot(
            days,
            daily_median,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=display_name,
        )

    ax_trend.set_title(
        f"Daily Median Queue Times - "
        f"{_hours_label(time_limit)} Time Limit, {gpus} GPUs",
        fontsize=17,
    )
    ax_trend.set_xlabel("Day", fontsize=15)
    ax_trend.set_ylabel("Queue time (hours)", fontsize=15)
    ax_trend.set_ylim(0, cap)
    ax_trend.yaxis.set_major_locator(mticker.MultipleLocator(3600))
    ax_trend.yaxis.set_major_formatter(lambda value, _: f"{int(round(value / 3600))}")
    ax_trend.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_trend.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax_trend.xaxis.get_major_locator())
    )
    ax_trend.tick_params(axis="both", labelsize=14)
    ax_trend.grid(True, linewidth=0.5, alpha=0.35)
    handles, labels = ax_trend.get_legend_handles_labels()
    handle_for_label = dict(zip(labels, handles))
    ordered: list[tuple[object, str]] = []
    for cluster in DEFAULT_CLUSTERS:
        line_label = cluster_labels[cluster]
        band_label = f"{line_label.split()[0]} IQR"
        if line_label in handle_for_label:
            ordered.append((handle_for_label[line_label], line_label))
        if band_label in handle_for_label:
            ordered.append((handle_for_label[band_label], band_label))
    ax_trend.legend(
        [h for h, _ in ordered],
        [l for _, l in ordered],
        ncol=3,
        fontsize=13,
        loc="upper right",
    )

    bin_edges = np.linspace(0, cap, 21)
    for cluster in DEFAULT_CLUSTERS:
        values = cluster_window_values.get(cluster, [])
        if not values:
            continue
        over = sum(1 for v in values if v > cap)
        color = colors[cluster]
        display_name = cluster_labels[cluster]
        label = display_name
        if over:
            label += f" (>{cap_hours}h: {over})"
        ax_dist.hist(
            values,
            bins=bin_edges,
            orientation="horizontal",
            color=color,
            alpha=0.35,
            edgecolor=color,
            linewidth=1.5,
            density=True,
            histtype="stepfilled",
            label=label,
        )

    ax_dist.set_title(f"{window_days}-Day Distribution", fontsize=17)
    ax_dist.set_xlabel("Density", fontsize=15)
    ax_dist.tick_params(axis="both", labelsize=14)
    ax_dist.grid(True, linewidth=0.5, alpha=0.35)
    ax_dist.legend(fontsize=12, loc="upper right")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def submit_probe(cluster: str, info: dict[str, str], timeout: int) -> str:
    parts = [info["init"]]
    if info.get("probe_cwd"):
        parts.append(f"cd {shlex.quote(info['probe_cwd'])}")
    parts.append(
        f"sbatch {info['sbatch_opts']} --job-name=queue_probe "
        "--output=/dev/null --wrap='true'"
    )
    remote_cmd = " && ".join(parts)
    result = subprocess.run(
        ["ssh", info["host"], remote_cmd],
        check=False, capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "<no output>"
        raise RuntimeError(f"submit failed (exit={result.returncode}): {msg}")
    for line in result.stdout.splitlines():
        if line.startswith("Submitted batch job"):
            return line.split()[-1]
    raise RuntimeError(f"no job ID in output: {result.stdout.strip()}")


def submit_all_probes(timeout: int) -> int:
    failures = 0
    with ThreadPoolExecutor(max_workers=len(CLUSTER_INFO)) as pool:
        futures = {
            pool.submit(submit_probe, cluster, info, timeout): cluster
            for cluster, info in CLUSTER_INFO.items()
        }
        for future in as_completed(futures):
            cluster = futures[future]
            try:
                job_id = future.result()
                print(f"[probe:{cluster}] submitted {job_id}")
            except Exception as exc:
                failures += 1
                print(f"[probe:{cluster}] {exc}", file=sys.stderr)
    print("Probes submitted; queue times will appear on the next run.")
    return failures


def run_report(args: argparse.Namespace) -> int:
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows_future = pool.submit(query_all_clusters, args)
        snapshots_future = pool.submit(query_all_snapshots, SNAPSHOT_TIMEOUT)
        rows, failures = rows_future.result()
        snapshots = snapshots_future.result()
    for failure in failures:
        print(failure)

    series = build_daily_series(rows, args.window)
    print_unified_table(rows, snapshots)
    print_recommendation(rows)
    if args.all:
        print_all(series, args.window)
    if not args.no_plot:
        plot_series(series, rows, args.plot, args.window, args.time_limit, args.gpus)
        print(f"Plot written to {args.plot}")
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    if args.window <= 0:
        raise SystemExit("--window must be positive")

    if args.probe:
        return 1 if submit_all_probes(SUBMIT_TIMEOUT) else 0

    rc = run_report(args)
    if not args.report:
        print()
        submit_all_probes(SUBMIT_TIMEOUT)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
