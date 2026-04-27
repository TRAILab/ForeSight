#!/usr/bin/env python3
"""Query SLURM accounting live and summarize queue times by cluster."""

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import subprocess
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
CLUSTER_QUERIES = (
    ("narval", "narval", "source ~/.bashrc"),
    ("trillium", "trillium_gpu", "source ~/.bashrc"),
    (
        "killarney",
        "killarney",
        "source /etc/profile.d/modules.sh && module load slurm/killarney/24.05.7",
    ),
)
GPU_TRES_RE = re.compile(r"^gres/gpu(?::[^=,]+)?=(\d+)$")
PERCENTILE_LOW = 25.0
PERCENTILE_HIGH = 75.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query SLURM accounting live and summarize queue times."
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
    with ThreadPoolExecutor(max_workers=len(CLUSTER_QUERIES)) as executor:
        futures = {
            executor.submit(
                query_cluster,
                cluster,
                host,
                init,
                args.start_time,
                args.end_time,
                args.slurm_user,
                args.time_limit,
                args.gpus,
                args.query_timeout,
            ): cluster
            for cluster, host, init in CLUSTER_QUERIES
        }
        for future in as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception as exc:
                failures.append(str(exc))
    rows.sort(key=lambda row: (str(row["cluster"]), int(row["submit_epoch"]), str(row["job_id"])))
    return rows, failures


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


def main() -> int:
    args = parse_args()
    if args.window <= 0:
        raise SystemExit("--window must be positive")

    rows, failures = query_all_clusters(args)
    for failure in failures:
        print(failure)

    series = build_daily_series(rows, args.window)
    if args.all:
        print_all(series, args.window)
    if not args.no_plot:
        plot_series(series, rows, args.plot, args.window, args.time_limit, args.gpus)
        print(f"Plot written to {args.plot}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
