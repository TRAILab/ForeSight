#!/usr/bin/env python3
"""GPU queue dashboard for Narval/Trillium/Killarney with bucketed time/gpu views.

Default: refresh per-bucket cluster tables + queue-time plot from live SLURM
accounting (across all TRAIL members), then submit 4-GPU no-op probe jobs (fire and
forget; results show up on the next run via sacct).
  --report   only refresh tables/plot (skip probe submission)
  --probe    only submit probes (skip tables/plot)
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import os
import re
import shlex
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(
    os.environ.get(
        "QUEUE_TIME_CACHE_DIR",
        str(Path.home() / ".cache" / "trail-queue-time"),
    )
)
DEFAULT_CLUSTERS = ("narval", "trillium", "killarney", "killarney_h100")
ADDITIONAL_QUEUE_HEATMAP_CLUSTERS = ("tamia", "rorqual", "fir")
MAP_CLUSTERS = tuple(dict.fromkeys((*DEFAULT_CLUSTERS, *ADDITIONAL_QUEUE_HEATMAP_CLUSTERS, "vulcan", "nibi", "utias", "apollo")))
SACCT_CLUSTERS = tuple(dict.fromkeys((*DEFAULT_CLUSTERS, *ADDITIONAL_QUEUE_HEATMAP_CLUSTERS)))
PENDING_HEATMAP_CLUSTERS = tuple(dict.fromkeys((*DEFAULT_CLUSTERS, *ADDITIONAL_QUEUE_HEATMAP_CLUSTERS)))
QUEUE_HEATMAP_CLUSTERS = PENDING_HEATMAP_CLUSTERS
TERMINAL_CLUSTERS = ("narval", "trillium", "killarney", "killarney_h100")
DRAC_ALLOCATED_SUMMARY_CLUSTERS = ("narval", "trillium")
OTHER_SUMMARY_CLUSTERS = ("killarney", "killarney_h100", "tamia", "fir", "rorqual")
MEMBER_ALLOCATED_USAGE_CLUSTERS = DRAC_ALLOCATED_SUMMARY_CLUSTERS
MEMBER_OTHER_USAGE_CLUSTERS = OTHER_SUMMARY_CLUSTERS
GPU_UTIL_USAGE_CLUSTERS = ("trillium", "fir")
CLUSTER_LOCATION = {
    "killarney": "Vector, Toronto",
    "killarney_h100": "Vector, Toronto",
    "tamia": "Mila, Montreal",
    "fir": "SFU, Vancouver",
    "rorqual": "ETS, Montreal",
}
CLUSTER_GEO = {
    "narval": {"site": "DRAC, Montreal", "lat": 45.5019, "lon": -73.5674},
    "trillium": {"site": "SciNet, Toronto", "lat": 43.6532, "lon": -79.3832},
    "killarney": {"site": "Vector, Toronto", "lat": 43.6616, "lon": -79.3910},
    "killarney_h100": {"site": "Vector, Toronto", "lat": 43.6616, "lon": -79.3910},
    "tamia": {"site": "Mila, Montreal", "lat": 45.5088, "lon": -73.5878},
    "fir": {"site": "SFU, Vancouver", "lat": 49.2781, "lon": -122.9199},
    "rorqual": {"site": "ETS, Montreal", "lat": 45.4948, "lon": -73.5620},
    "vulcan": {"site": "Amii, Edmonton", "lat": 53.5462, "lon": -113.4937},
    "nibi": {"site": "Waterloo", "lat": 43.4723, "lon": -80.5449},
    "apollo": {"site": "UWaterloo", "lat": 43.07, "lon": -80.96},
    "utias": {"site": "UTIAS, Toronto", "lat": 44.1800, "lon": -79.2600},
}
TRAIL_CLUSTERS = ("dgx", "apollo", "turing", "lovelace", "ums", "um1", "um2", "um3")
DASHBOARD_STALE_CLUSTERS = tuple(dict.fromkeys((*SACCT_CLUSTERS, *TRAIL_CLUSTERS)))
SECONDS_PER_DAY = 24 * 60 * 60
SACCT_FIELDS = (
    "JobIDRaw",
    "User",
    "Submit",
    "Start",
    "End",
    "State",
    "Timelimit",
    "AllocCPUS",
    "ReqMem",
    "ReqTRES",
    "AllocTRES",
    "AveRSS",
    "MaxDiskRead",
    "MaxDiskWrite",
    "TRESUsageInAve",
    "TRESUsageInTot",
)
CLUSTER_INFO: dict[str, dict[str, str]] = {
    "narval": {
        "kind": "slurm",
        "group": "external",
        "host": "narval",
        "ip": "narval.alliancecan.ca",
        "init": "source ~/.bashrc",
        "node_type": "4xA100-80GB",
        "partition": "gpubase_bygpu_b1,gpubase_bygpu_b2,gpubase_bygpu_b3,gpubase_bygpu_b4,gpubase_bygpu_b5,gpubase_bynode_b1,gpubase_bynode_b2,gpubase_bynode_b3,gpubase_bynode_b4,gpubase_bynode_b5",
        "account": "rrg-swasland_gpu",
        "sbatch_opts": "--account=rrg-swasland --ntasks=1 --cpus-per-task=12 --mem=120gb --time=11:59:00 --gres=gpu:a100:4",
        "sbatch_opts_short": "--account=rrg-swasland --ntasks=1 --cpus-per-task=12 --mem=120gb --time=2:59:00 --gres=gpu:a100:4",
        "sbatch_opts_24h": "--account=rrg-swasland --ntasks=1 --cpus-per-task=12 --mem=120gb --time=23:59:00 --gres=gpu:a100:4",
        "sbatch_opts_24h_1gpu": "--account=rrg-swasland --ntasks=1 --cpus-per-task=3 --mem=30gb --time=23:59:00 --gres=gpu:a100:1",
        "sacct_account": "rrg-swasland",
        "probe_cwd": "",
        "lab_users_dir": "$HOME/projects/rrg-swasland",
        "weekly_gpu_hrs_target": "172",
    },
    "trillium": {
        "kind": "slurm",
        "group": "external",
        "host": "trillium_gpu",
        "ip": "trillium-gpu.scinet.utoronto.ca",
        "init": "source ~/.bashrc",
        "node_type": "4xH100-80GB",
        "partition": "compute,compute_full_node,compute_h200,compute_h200_full_node",
        "account": "rrg-swasland",
        "sbatch_opts": "--account=rrg-swasland --ntasks=1 --cpus-per-task=24 --time=11:59:00 --gpus-per-node=4",
        "sbatch_opts_short": "--account=rrg-swasland --ntasks=1 --cpus-per-task=24 --time=2:59:00 --gpus-per-node=4",
        "sbatch_opts_24h": "--account=rrg-swasland --ntasks=1 --cpus-per-task=24 --time=23:59:00 --gpus-per-node=4",
        "sacct_account": "rrg-swasland",
        "probe_cwd": "",
        "lab_users_dir": "$HOME/links/projects/rrg-swasland",
        "weekly_gpu_hrs_target": "103",
        "pending_gpu_groups": (4, 1),
    },
    "killarney": {
        "kind": "slurm",
        "group": "external",
        "host": "killarney",
        "ip": "killarney.alliancecan.ca",
        "init": "source /etc/profile.d/modules.sh && module load slurm/killarney/24.05.7",
        "node_type": "4xL40S-48GB",
        "partition": "gpubase_l40s_b1,gpubase_l40s_b2,gpubase_l40s_b3,gpubase_l40s_b4,gpubase_l40s_b5",
        "account": "",
        "sbatch_opts": "--account=aip-swasland --ntasks=1 --cpus-per-task=16 --mem=120gb --time=11:59:00 --gres=gpu:l40s:4",
        "sbatch_opts_short": "--account=aip-swasland --ntasks=1 --cpus-per-task=16 --mem=120gb --time=2:59:00 --gres=gpu:l40s:4",
        "sbatch_opts_24h": "--account=aip-swasland --ntasks=1 --cpus-per-task=16 --mem=120gb --time=23:59:00 --gres=gpu:l40s:4",
        "sacct_account": "aip-swasland",
        "probe_cwd": "/scratch",
        "lab_users_dir": "$HOME/projects/aip-swasland",
        "gres_filter": "l40s",
    },
    "killarney_h100": {
        "kind": "slurm",
        "group": "external",
        "host": "killarney",
        "ip": "killarney.alliancecan.ca",
        "init": "source /etc/profile.d/modules.sh && module load slurm/killarney/24.05.7",
        "node_type": "8xH100-48GB",
        "partition": "gpubase_h100_b1,gpubase_h100_b2,gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5",
        "account": "",
        "sbatch_opts": "--account=aip-swasland --ntasks=1 --cpus-per-task=16 --mem=120gb --time=11:59:00 --gres=gpu:h100:4",
        "sbatch_opts_short": "--account=aip-swasland --ntasks=1 --cpus-per-task=16 --mem=120gb --time=2:59:00 --gres=gpu:h100:4",
        "sbatch_opts_24h": "--account=aip-swasland --ntasks=1 --cpus-per-task=16 --mem=120gb --time=23:59:00 --gres=gpu:h100:4",
        "sacct_account": "aip-swasland",
        "probe_cwd": "/scratch",
        "lab_users_dir": "$HOME/projects/aip-swasland",
        "gpu_groups": (8, 4, 2, 1),
        "gres_filter": "h100",
    },
    "vulcan": {
        "kind": "slurm",
        "group": "external",
        "host": "vulcan",
        "ip": "vulcan.alliancecan.ca",
        "init": "source ~/.bashrc",
        "node_type": "4xL40S-48GB",
        "account": "aip-swasland",
        "sbatch_opts": "--account=aip-swasland --ntasks=1 --cpus-per-task=16 --mem=120gb --time=11:59:00 --gres=gpu:l40s:4",
        "sbatch_opts_short": "--account=aip-swasland --ntasks=1 --cpus-per-task=8 --mem=60gb --time=2:59:00 --gres=gpu:l40s:2",
        "sbatch_opts_24h": "--account=aip-swasland --ntasks=1 --cpus-per-task=16 --mem=120gb --time=23:59:00 --gres=gpu:l40s:4",
        "sacct_account": "aip-swasland",
        "probe_cwd": "",
        "lab_users_dir": "$HOME/projects/aip-swasland",
        "gres_filter": "l40s",
    },
    "fir": {
        "kind": "slurm",
        "group": "external",
        "host": "fir",
        "ip": "fir.alliancecan.ca",
        "init": "source ~/.bashrc",
        "node_type": "4xH100-SXM5-80GB",
        "account": "def-swasland-ab",
        "share_account": "def-swasland-ab_gpu",
        "sbatch_opts": "--account=def-swasland-ab --ntasks=1 --cpus-per-task=12 --mem=200gb --time=11:59:00 --gres=gpu:h100:4",
        "sbatch_opts_short": "--account=def-swasland-ab --ntasks=1 --cpus-per-task=6 --mem=100gb --time=2:59:00 --partition=gpubase_bygpu_b1 --gres=gpu:h100:2",
        "sbatch_opts_24h": "--account=def-swasland-ab --ntasks=1 --cpus-per-task=12 --mem=200gb --time=23:59:00 --gres=gpu:h100:4",
        "sacct_account": "def-swasland-ab",
        "probe_cwd": "",
        "lab_users_dir": "/project/def-swasland-ab",
        "partition": "gpubase_bynode_b1,gpubase_bynode_b2,gpubase_bynode_b3,gpubase_bynode_b4,gpubase_bynode_b5,gpubase_bygpu_b1,gpubase_bygpu_b2,gpubase_bygpu_b3,gpubase_bygpu_b4,gpubase_bygpu_b5",
        "snapshot_gres_filter": "gpu:h100:4",
        "snapshot_node_filter": "h100",
    },
    "nibi": {
        "kind": "slurm",
        "group": "external",
        "host": "nibi",
        "ip": "nibi.alliancecan.ca",
        "init": "source ~/.bashrc",
        "node_type": "8xH100-SXM-80GB",
        "account": "def-swasland",
        "sbatch_opts": "--account=def-swasland --ntasks=1 --cpus-per-task=14 --mem=200gb --time=11:59:00 --partition=gpubase_bynode_b1 --gres=gpu:h100:4",
        "sbatch_opts_short": "--account=def-swasland --ntasks=1 --cpus-per-task=14 --mem=200gb --time=2:59:00 --partition=gpubase_bynode_b1 --gres=gpu:h100:4",
        "sbatch_opts_24h": "--account=def-swasland --ntasks=1 --cpus-per-task=14 --mem=200gb --time=23:59:00 --partition=gpubase_bynode_b1 --gres=gpu:h100:4",
        "sacct_account": "def-swasland",
        "probe_cwd": "",
        "lab_users_dir": "$HOME/projects/def-swasland",
        "gres_filter": "h100",
    },
    "rorqual": {
        "kind": "slurm",
        "group": "external",
        "host": "rorqual",
        "ip": "rorqual.alliancecan.ca",
        "init": "source ~/.bashrc",
        "node_type": "4xH100-SXM5-80GB",
        "account": "def-swasland-ab_gpu",
        "sbatch_opts": "--account=def-swasland-ab_gpu --ntasks=1 --cpus-per-task=16 --mem=120gb --time=11:59:00 --partition=gpubase_bynode_b2 --gres=gpu:h100:4",
        "sbatch_opts_short": "--account=def-swasland-ab_gpu --ntasks=1 --cpus-per-task=8 --mem=60gb --time=2:59:00 --partition=gpubase_bynode_b1 --gres=gpu:h100:2",
        "sbatch_opts_24h": "--account=def-swasland-ab_gpu --ntasks=1 --cpus-per-task=16 --mem=120gb --time=23:59:00 --partition=gpubase_bynode_b3 --gres=gpu:h100:4",
        "sacct_account": "def-swasland-ab_gpu",
        "probe_cwd": "",
        "lab_users_dir": "$HOME/links/projects/def-swasland-ab",
        "partition": "gpubase_bynode_b1,gpubase_bynode_b2,gpubase_bynode_b3,gpubase_bynode_b4,gpubase_bynode_b5,gpubase_bygpu_b1,gpubase_bygpu_b2,gpubase_bygpu_b3,gpubase_bygpu_b4,gpubase_bygpu_b5",
        "snapshot_gres_filter": "gpu:h100:4",
        "snapshot_node_filter": "h100",
    },
    "tamia": {
        "kind": "slurm",
        "group": "external",
        "host": "tamia",
        "ip": "tamia.alliancecan.ca",
        "init": "source ~/.bashrc",
        "node_type": "4xH100-HGX-80GB",
        "account": "aip-swasland",
        # whole-node allocation only — 2-GPU jobs not supported
        "sbatch_opts": "--account=aip-swasland --ntasks=1 --cpus-per-task=12 --mem=120gb --time=11:59:00 --gpus-per-node=h100:4",
        "sbatch_opts_short": "--account=aip-swasland --ntasks=1 --cpus-per-task=12 --mem=120gb --time=2:59:00 --gpus-per-node=h100:4",
        "sbatch_opts_24h": "--account=aip-swasland --ntasks=1 --cpus-per-task=12 --mem=120gb --time=23:59:00 --gpus-per-node=h100:4",
        "sacct_account": "aip-swasland",
        "probe_cwd": "",
        "lab_users_dir": "$HOME/links/projects/aip-swasland",
        "partition": "gpubase_bynode_b1,gpubase_bynode_b2,gpubase_bynode_b3",
        "snapshot_gres_filter": "gpu:h100:4",
        "snapshot_node_filter": "gpu:h100",
        "pending_gpu_groups": (4,),
    },
    "dgx": {
        "kind": "slurm",
        "group": "trail",
        "host": "trail_dgx",
        "ip": "192.168.42.200",
        "init": "source ~/.bashrc 2>/dev/null || true",
        "node_type": "4xA100-SXM4-40GB",
        "partition": "batch",
        "account": "",
        "sacct_account": "",
        "lab_users_dir": "",
        # SLURM gres counts include the on-board "NVIDIA DGX Display" GPU,
        # which isn't trainable. Use nvidia-smi (filtering by name) to get
        # the actual trainable-GPU usage; the SLURM counts still drive
        # running/pending and the rest of the snapshot.
        "gpu_source": "nvidia-smi",
        "gpu_exclude_pattern": "Display",
        "usable_gpus": 4,
    },
    "apollo": {
        "kind": "smi",
        "group": "trail",
        "host": "apollo",
        "ip": "129.97.163.137",
        "init": "true",
        "node_type": "8xV100-32GB",
    },
    "turing": {
        "kind": "smi",
        "group": "trail",
        "host": "trail_turing",
        "ip": "192.168.42.135",
        "init": "true",
        "node_type": "4xRTX6000-Ada-48GB",
    },
    "lovelace": {
        "kind": "smi",
        "group": "trail",
        "host": "trail_lovelace",
        "ip": "192.168.42.106",
        "init": "true",
        "node_type": "4xRTX6000-Ada-48GB",
    },
    "ums": {
        "kind": "smi",
        "group": "trail",
        "host": "trail_UMS",
        "ip": "192.168.42.235",
        "init": "true",
        "node_type": "1xRTX4090-24GB",
    },
    "um1": {
        "kind": "smi",
        "group": "trail",
        "host": "trail_UM1",
        "ip": "192.168.42.153",
        "init": "true",
        "node_type": "1xRTX4090-24GB",
    },
    "um2": {
        "kind": "smi",
        "group": "trail",
        "host": "trail_UM2",
        "ip": "192.168.42.134",
        "init": "true",
        "node_type": "1xRTX4090-24GB",
    },
    "um3": {
        "kind": "smi",
        "group": "trail",
        "host": "trail_UM3",
        "ip": "192.168.42.248",
        "init": "true",
        "node_type": "1xRTX4090-24GB",
    },
}
SNAPSHOT_TIMEOUT = 25
SUBMIT_TIMEOUT = 15
SSH_CONNECT_TIMEOUT = int(os.environ.get("QUEUE_TIME_SSH_CONNECT_TIMEOUT", "3"))
SSH_SERVER_ALIVE_INTERVAL = int(os.environ.get("QUEUE_TIME_SSH_SERVER_ALIVE_INTERVAL", "5"))
SSH_SERVER_ALIVE_COUNT_MAX = int(os.environ.get("QUEUE_TIME_SSH_SERVER_ALIVE_COUNT_MAX", "1"))
# Options applied to every `ssh` invocation. ConnectTimeout fails an
# unreachable host quickly instead of waiting for the OS-level TCP timeout;
# BatchMode prevents password prompts from hanging stdin in cron; ServerAlive
# kills mid-stream hangs (e.g. flaky VPN) after one missed keepalive by default.
SSH_OPTS = (
    "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
    "-o", "BatchMode=yes",
    "-o", f"ServerAliveInterval={SSH_SERVER_ALIVE_INTERVAL}",
    "-o", f"ServerAliveCountMax={SSH_SERVER_ALIVE_COUNT_MAX}",
)
GPU_TRES_RE = re.compile(r"^gres/gpu(?::[^=,]+)?=(\d+)$")
GPU_FROM_GRES_RE = re.compile(r"(?:gres[/:])?gpu(?::\w+)?:(\d+)")
PERCENTILE_LOW = 25.0
PERCENTILE_HIGH = 75.0

TIME_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("3h",  0,                  3 * 3600 + 1),
    ("12h", 3 * 3600 + 1,      12 * 3600 + 1),
    ("1d",  12 * 3600 + 1,     24 * 3600 + 1),
    ("3d",  24 * 3600 + 1,     72 * 3600 + 1),
    ("7d",  72 * 3600 + 1,     None),
)
GPU_GROUPS = (4, 2, 1)
LEVELFS_USER = os.environ.get("QUEUE_TIME_LEVELFS_USER", "spapais")
TRAIL_FAVICON_URL = (
    "https://static.wixstatic.com/media/"
    "40d842_8fc236af72fc4289aaab95c04bc9a6c6%7Emv2.png/"
    "v1/fill/w_32%2Ch_32%2Clg_1%2Cusm_0.66_1.00_0.01/"
    "40d842_8fc236af72fc4289aaab95c04bc9a6c6%7Emv2.png"
)


def _heatmap_gpu_groups(cluster: str) -> tuple[int, ...]:
    info = CLUSTER_INFO[cluster]
    groups = info.get("pending_gpu_groups", info.get("gpu_groups", GPU_GROUPS))
    return tuple(groups) if not isinstance(groups, tuple) else groups
TABLE_LOOKBACK_HOURS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh GPU queue tables/plot and submit probe jobs.",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=14,
        help="display window in days for plot/CSV (default: 14)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="print the full daily-median series instead of only the latest value",
    )
    parser.add_argument(
        "--start-time",
        default=os.environ.get("START_TIME", "now-30days"),
        help="SLURM accounting start time (default: now-14days)",
    )
    parser.add_argument(
        "--end-time",
        default=os.environ.get("END_TIME", "now"),
        help="SLURM accounting end time (default: END_TIME env or now)",
    )
    parser.add_argument(
        "--slurm-users",
        default=os.environ.get("SLURM_USERS", ""),
        help=(
            "comma-separated SLURM users to query (default: auto-discover from "
            "each cluster's lab project dir)"
        ),
    )
    parser.add_argument(
        "--time-limit",
        default=os.environ.get("TIME_LIMIT", "11:59:00"),
        help="filter to this SLURM time limit for the plot only (default: 11:59:00)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=int(os.environ.get("GPUS", "4")),
        help="filter to this GPU count for the plot only (default: 4)",
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
    default_html = Path(os.environ.get(
        "QUEUE_TIME_HTML", str(Path.home() / "queue-status" / "index.html"),
    ))
    default_refresh = int(os.environ.get("QUEUE_TIME_REFRESH", "7200"))
    parser.add_argument(
        "--html",
        type=Path,
        default=default_html,
        help=(
            "write a self-contained HTML page to this path "
            f"(default: {default_html}; env QUEUE_TIME_HTML)"
        ),
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="skip HTML generation (overrides --html default)",
    )
    parser.add_argument(
        "--refresh-seconds",
        type=int,
        default=default_refresh,
        help=(
            "auto-refresh interval baked into the HTML page "
            f"(default: {default_refresh}s; env QUEUE_TIME_REFRESH)"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--report",
        action="store_true",
        help="only refresh tables/plot; skip probe submission",
    )
    mode.add_argument(
        "--probe",
        nargs="?",
        const="all",
        default=None,
        metavar="SPEC",
        help=(
            "submit probe jobs and exit. Optional SPEC limits to one probe, "
            "e.g. n4g12h  →  {n|t|k|kh}[{N}g]{3h|12h|1d}"
        ),
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


def format_hours_short(seconds: float | int | None) -> str:
    """Decimal hours, e.g. '2.4h'. Returns en-dash for None."""
    if seconds is None:
        return "–"
    return f"{float(seconds) / 3600:.1f}h"


def _color_for_util_pct(pct: float | None) -> str | None:
    """HSL color for a 0-100 utilisation pct (0=red, 100=green)."""
    if pct is None:
        return None
    h = max(0.0, min(120.0, 1.2 * float(pct)))
    return f"hsl({h:.0f}, 65%, 38%)"


def _color_for_queue_seconds(seconds: float | None) -> str | None:
    """HSL color for queue-time seconds (0s=green, 4h+=red)."""
    if seconds is None:
        return None
    hours = max(0.0, float(seconds) / 3600.0)
    h = max(0.0, min(120.0, 120.0 - hours * 30.0))
    return f"hsl({h:.0f}, 65%, 38%)"


def format_staleness(seconds: int | None) -> str:
    """Human-readable cache age. None → 'no cached data'."""
    if seconds is None:
        return "no cached data"
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds) // 60}m ago"
    if seconds < SECONDS_PER_DAY:
        h, rem = divmod(int(seconds), 3600)
        return f"{h}h {rem // 60:02d}m ago"
    d, rem = divmod(int(seconds), SECONDS_PER_DAY)
    return f"{d}d {rem // 3600}h ago"


def _cache_path(kind: str, cluster: str) -> Path:
    return CACHE_DIR / f"{kind}_{cluster}.json"


def _save_cache(kind: str, cluster: str, payload: object) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_path(kind, cluster)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {"timestamp": int(datetime.now().timestamp()), "data": payload}
            ),
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception:
        # Cache writes are best-effort; never fail the report on cache errors.
        pass


def _load_cache(kind: str, cluster: str) -> tuple[object, int] | None:
    try:
        path = _cache_path(kind, cluster)
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj["data"], int(obj["timestamp"])
    except Exception:
        return None


TRAIL_HISTORY_RETENTION_DAYS = 30
# Window used for the lookback columns, the timeline plot, and the
# `_trail_*` aggregation helpers; column headers re-render automatically.
TRAIL_LOOKBACK_DAYS = 14


def _trail_history_path(cluster: str) -> Path:
    return CACHE_DIR / f"history_{cluster}.jsonl"


def _append_trail_history(cluster: str, snap: dict[str, object]) -> None:
    """Append a single utilisation sample to per-cluster JSONL history.

    Captures used/total GPUs, the user list, and a job count so we can
    aggregate lookback-window stats. Pruning happens lazily once the file grows.
    """
    used = snap.get("gpus_used")
    total = snap.get("gpus_total")
    if used is None or total is None:
        return
    usable = CLUSTER_INFO.get(cluster, {}).get("usable_gpus")
    if isinstance(usable, int):
        total = min(int(total), usable)
        used = min(int(used), usable)
    # Per-kind job count: SLURM running jobs for DGX, distinct GPU-using
    # containers for SMI hosts. Falls back to 0 when not measured.
    if snap.get("kind") == "smi":
        jobs = int(snap.get("containers") or 0)
    else:
        running = snap.get("running")
        jobs = int(running) if running is not None else 0
    users = snap.get("users") or []
    if not isinstance(users, list):
        users = []
    mem_total_bytes = snap.get("mem_total_bytes")
    mem_used_bytes = snap.get("mem_used_bytes")
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _trail_history_path(cluster)
        ts = int(datetime.now().timestamp())
        rec_dict: dict[str, object] = {
            "ts": ts,
            "used": int(used),
            "total": int(total),
            "jobs": jobs,
            "users": [str(u) for u in users],
        }
        user_gpu = snap.get("user_gpu") or {}
        if isinstance(user_gpu, dict):
            rec_dict["user_gpu"] = {
                str(user): metrics
                for user, metrics in user_gpu.items()
                if isinstance(metrics, dict)
            }
        if isinstance(mem_total_bytes, int) and mem_total_bytes > 0 \
                and isinstance(mem_used_bytes, int) and mem_used_bytes >= 0:
            rec_dict["mem_used_bytes"] = int(mem_used_bytes)
            rec_dict["mem_total_bytes"] = int(mem_total_bytes)
        record = json.dumps(rec_dict)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(record + "\n")
        # Prune (rewrite) when the file grows past ~3x the retention window
        # to avoid unbounded growth from cron sampling.
        cutoff = ts - TRAIL_HISTORY_RETENTION_DAYS * SECONDS_PER_DAY
        try:
            stat = path.stat()
            # Roughly 80B/line × 12/day × 30d ≈ 30KB. Above 200KB, prune.
            if stat.st_size > 200_000:
                kept = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        rec = json.loads(line)
                        if int(rec.get("ts", 0)) >= cutoff:
                            kept.append(line)
                    except Exception:
                        continue
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
                tmp.replace(path)
        except Exception:
            pass
    except Exception:
        pass


def _load_trail_history(
    cluster: str, days: int = 7
) -> list[dict[str, object]]:
    """Read the last `days` of samples for a cluster as a list of dicts.

    Each record has keys: ts, used, total, jobs, users (older records may
    only have ts/used/total — missing keys default to empty/zero).
    """
    path = _trail_history_path(cluster)
    if not path.exists():
        return []
    cutoff = int(datetime.now().timestamp()) - days * SECONDS_PER_DAY
    out: list[dict[str, object]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                ts = int(rec.get("ts", 0))
                if ts < cutoff:
                    continue
                users = rec.get("users") or []
                if not isinstance(users, list):
                    users = []
                mem_used = rec.get("mem_used_bytes")
                mem_total = rec.get("mem_total_bytes")
                user_gpu = rec.get("user_gpu") or {}
                if not isinstance(user_gpu, dict):
                    user_gpu = {}
                out.append({
                    "ts": ts,
                    "used": int(rec.get("used", 0)),
                    "total": int(rec.get("total", 0)),
                    "jobs": int(rec.get("jobs", 0)),
                    "users": [str(u) for u in users],
                    "mem_used_bytes": int(mem_used) if isinstance(mem_used, (int, float)) else None,
                    "mem_total_bytes": int(mem_total) if isinstance(mem_total, (int, float)) else None,
                    "user_gpu": user_gpu,
                })
            except Exception:
                continue
    except Exception:
        return []
    out.sort(key=lambda r: int(r["ts"]))  # type: ignore[arg-type]
    return out


def _snapshot_to_jsonable(snap: dict[str, object]) -> dict[str, object]:
    out = dict(snap)
    pbb = out.get("pending_by_bucket") or {}
    if isinstance(pbb, dict):
        out["pending_by_bucket"] = [[g, b, c] for (g, b), c in pbb.items()]
    return out


def _snapshot_from_jsonable(data: dict[str, object]) -> dict[str, object]:
    out = dict(data)
    pbb = out.get("pending_by_bucket") or []
    if isinstance(pbb, list):
        out["pending_by_bucket"] = {(int(g), str(b)): int(c) for g, b, c in pbb}
    return out


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


def parse_time_limit_seconds(value: str) -> int | None:
    """Parse 'HH:MM:SS' or 'D-HH:MM:SS' (sacct/squeue Timelimit) to seconds."""
    if not value or value in ("UNLIMITED", "INVALID", "Partition_Limit", "NOT_SET"):
        return None
    days = 0
    s = value
    if "-" in s:
        d, s = s.split("-", 1)
        try:
            days = int(d)
        except ValueError:
            return None
    parts = s.split(":")
    try:
        h = int(parts[0]) if len(parts) > 0 and parts[0] else 0
        m = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        sec = int(parts[2]) if len(parts) > 2 and parts[2] else 0
    except ValueError:
        return None
    return days * SECONDS_PER_DAY + h * 3600 + m * 60 + sec


def parse_slurm_duration_seconds(value: str) -> int | None:
    """Parse sacct CPU time strings such as '01:02:03' or '2-01:02:03'."""
    return parse_time_limit_seconds(value)


def parse_memory_bytes(value: str) -> int | None:
    """Parse Slurm memory strings such as 120G, 68206184K, or 192500Mc."""
    if not value:
        return None
    s = value.strip()
    if s in ("Unknown", "N/A", "0"):
        return None
    # ReqMem may end in c/n for per-cpu/per-node. The caller handles that
    # semantic; this parser converts only the magnitude.
    if s[-1:].lower() in ("c", "n"):
        s = s[:-1]
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)([kmgtp]?)b?$", s, re.IGNORECASE)
    if not m:
        return None
    value_float = float(m.group(1))
    unit = m.group(2).upper()
    scale = {
        "": 1,
        "K": 1024,
        "M": 1024 ** 2,
        "G": 1024 ** 3,
        "T": 1024 ** 4,
        "P": 1024 ** 5,
    }[unit]
    return int(value_float * scale)


def format_bytes_short(num_bytes: float | int | None) -> str:
    if num_bytes is None:
        return "—"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(value) < 1024.0 or unit == "PB":
            if unit == "B":
                return f"{value:.0f}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}PB"


def extract_tres_value(tres: str, key: str) -> str | None:
    for item in str(tres or "").split(","):
        if item.startswith(f"{key}="):
            return item.split("=", 1)[1]
    return None


def parse_req_mem_total_bytes(req_mem: str, alloc_cpus: int, req_tres: str) -> int | None:
    if req_mem:
        per_cpu = req_mem.strip().lower().endswith("c")
        parsed = parse_memory_bytes(req_mem)
        if parsed is not None:
            return parsed * max(1, alloc_cpus) if per_cpu else parsed
    tres_mem = extract_tres_value(req_tres, "mem")
    return parse_memory_bytes(tres_mem or "")


def extract_gpu_count(req_tres: str, alloc_tres: str) -> int | None:
    """Find the GPU count in 'gres/gpu(:type)?=N' TRES strings."""
    for tres in (alloc_tres, req_tres):
        for item in tres.split(","):
            m = GPU_TRES_RE.match(item)
            if m:
                return int(m.group(1))
    return None


def bucket_for_time_limit(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    for label, lo, hi in TIME_BUCKETS:
        if hi is None:
            if seconds >= lo:
                return label
        elif lo <= seconds < hi:
            return label
    return None


def query_cluster(
    cluster: str,
    info: dict[str, str],
    start_time: str,
    end_time: str,
    slurm_users_override: list[str] | None,
    timeout: int,
) -> list[dict[str, object]]:
    sacct_format = ",".join(SACCT_FIELDS)
    if slurm_users_override:
        users_assign = "USERS=" + shlex.quote(",".join(slurm_users_override))
    else:
        lab_dir = info.get("lab_users_dir", "")
        if not lab_dir:
            return []
        # Filter to entries that look like POSIX usernames (lowercase + digits).
        # Unquoted $u is safe here because grep guarantees [a-z0-9] only.
        # Verify via `id` so stray data dirs (datasets/, singularity/, etc.)
        # are dropped; sacct aborts entirely if any invalid user is supplied.
        users_assign = (
            f'USERS=$(ls -1 {lab_dir} 2>/dev/null'
            r" | grep -E '^[a-z][a-z0-9]{1,}$'"
            r" | while IFS= read -r u; do id -u $u >/dev/null 2>&1 && printf '%s\n' $u; done"
            r' | tr "\n" "," | sed "s/,$//")'
        )

    remote_cmd = (
        f"{info['init']} && {users_assign} && "
        'if [ -n "$USERS" ]; then '
        'sacct -u "$USERS" --parsable2 --noheader '
        f"--starttime {shlex.quote(start_time)} --endtime {shlex.quote(end_time)} "
        f"--format={shlex.quote(sacct_format)} 2>/dev/null; "
        "fi"
    )
    result = subprocess.run(
        ["ssh", *SSH_OPTS, info["host"], remote_cmd],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0 and not result.stdout.strip():
        stderr = result.stderr.strip() or "(no output)"
        raise RuntimeError(f"{cluster}: query failed: {stderr}")

    now_epoch = int(datetime.now().timestamp())
    rows: list[dict[str, object]] = []
    step_usage: dict[str, dict[str, object]] = {}
    top_level_lines: list[list[str]] = []
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) != len(SACCT_FIELDS):
            continue
        (
            job_id,
            user,
            submit,
            start,
            end,
            state,
            row_time_limit,
            alloc_cpus_str,
            req_mem,
            req_tres,
            alloc_tres,
            ave_rss,
            max_disk_read,
            max_disk_write,
            tres_usage_in_ave,
            tres_usage_in_tot,
        ) = fields

        if "." in job_id:
            base_job_id, step_name = job_id.split(".", 1)
            if step_name != "batch":
                continue
            cpu_total = parse_slurm_duration_seconds(
                extract_tres_value(tres_usage_in_tot, "cpu") or ""
            )
            ave_mem = parse_memory_bytes(ave_rss) or parse_memory_bytes(
                extract_tres_value(tres_usage_in_ave, "mem") or ""
            )
            disk_total = parse_memory_bytes(
                extract_tres_value(tres_usage_in_tot, "fs/disk") or ""
            )
            if disk_total is None:
                read_bytes = parse_memory_bytes(max_disk_read) or 0
                write_bytes = parse_memory_bytes(max_disk_write) or 0
                disk_total = read_bytes + write_bytes if read_bytes or write_bytes else None
            gpuutil = extract_tres_value(tres_usage_in_ave, "gres/gpuutil")
            step_usage[base_job_id] = {
                "cpu_total_seconds": cpu_total,
                "ave_rss_bytes": ave_mem,
                "disk_io_bytes": disk_total,
                "gpuutil": float(gpuutil) if gpuutil and gpuutil.replace(".", "", 1).isdigit() else None,
            }
            continue
        top_level_lines.append(fields)

    for fields in top_level_lines:
        (
            job_id,
            user,
            submit,
            start,
            end,
            state,
            row_time_limit,
            alloc_cpus_str,
            req_mem,
            req_tres,
            alloc_tres,
            ave_rss,
            max_disk_read,
            max_disk_write,
            tres_usage_in_ave,
            tres_usage_in_tot,
        ) = fields

        gres_filter = info.get("gres_filter", "")
        if gres_filter:
            tres_combined = req_tres + "," + alloc_tres
            if not any(item.startswith(f"gres/gpu:{gres_filter}=") for item in tres_combined.split(",")):
                continue

        gpus = extract_gpu_count(req_tres, alloc_tres)
        if gpus is None:
            continue
        time_limit_seconds = parse_time_limit_seconds(row_time_limit)
        if time_limit_seconds is None:
            continue
        try:
            alloc_cpus = int(alloc_cpus_str)
        except ValueError:
            alloc_cpus = 0
        submit_epoch = parse_slurm_time(submit)
        if submit_epoch is None:
            continue
        start_epoch = parse_slurm_time(start)
        end_epoch = parse_slurm_time(end)
        if start_epoch is not None:
            # Job started: queue wait = start - submit
            pending = False
            queue_seconds = max(0, start_epoch - submit_epoch)
        elif end_epoch is not None:
            # Cancelled/timed-out while pending: queue wait = end - submit (exact)
            pending = False
            queue_seconds = max(0, end_epoch - submit_epoch)
        else:
            # Still pending: use elapsed time as a lower-bound estimate
            pending = True
            queue_seconds = max(0, now_epoch - submit_epoch)

        # Run time: only meaningful when the job actually started.
        # Completed = end - start. Still running = now - start (capped at
        # time_limit so a stuck job doesn't inflate GPU-hours).
        if start_epoch is None:
            run_seconds = 0
        elif end_epoch is not None:
            run_seconds = max(0, end_epoch - start_epoch)
        else:
            run_seconds = max(0, min(now_epoch - start_epoch, time_limit_seconds))
        req_mem_bytes = parse_req_mem_total_bytes(req_mem, alloc_cpus, req_tres)
        usage = step_usage.get(job_id, {})

        rows.append(
            {
                "cluster": cluster,
                "job_id": job_id,
                "submit_epoch": submit_epoch,
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
                "queue_seconds": queue_seconds,
                "run_seconds": run_seconds,
                "time_limit_seconds": time_limit_seconds,
                "gpus": gpus,
                "pending": pending,
                "user": user,
                "state": state,
                "alloc_cpus": alloc_cpus,
                "req_mem_bytes": req_mem_bytes,
                "cpu_total_seconds": usage.get("cpu_total_seconds"),
                "ave_rss_bytes": usage.get("ave_rss_bytes"),
                "disk_io_bytes": usage.get("disk_io_bytes"),
                "gpuutil": usage.get("gpuutil"),
            }
        )
    return rows


def query_all_clusters(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[str], dict[str, int | None]]:
    """Query SLURM accounting on every cluster, with per-cluster cache fallback.

    Returns (rows, failures, stale_acct) where stale_acct maps the clusters
    whose data came from cache to the cache age in seconds (or None if no
    cached data was available).
    """
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    stale: dict[str, int | None] = {}
    explicit_users: list[str] | None
    if args.slurm_users:
        explicit_users = [u.strip() for u in args.slurm_users.split(",") if u.strip()]
    else:
        explicit_users = None

    now_epoch = int(datetime.now().timestamp())
    # sacct history is meaningful only for the external clusters that drive the
    # queue-time plots; TRAIL hosts (DGX/Apollo) are summarised live, not
    # retrospectively.
    sacct_clusters = {c: CLUSTER_INFO[c] for c in SACCT_CLUSTERS}
    with ThreadPoolExecutor(max_workers=len(sacct_clusters)) as executor:
        futures = {
            executor.submit(
                query_cluster,
                cluster,
                info,
                args.start_time,
                args.end_time,
                explicit_users,
                args.query_timeout,
            ): cluster
            for cluster, info in sacct_clusters.items()
        }
        for future in as_completed(futures):
            cluster = futures[future]
            try:
                cluster_rows = future.result()
                rows.extend(cluster_rows)
                _save_cache("acct", cluster, cluster_rows)
            except Exception as exc:
                failures.append(str(exc))
                cached = _load_cache("acct", cluster)
                if cached is None:
                    stale[cluster] = None
                else:
                    cached_rows, ts = cached
                    if isinstance(cached_rows, list):
                        rows.extend(cached_rows)
                    stale[cluster] = max(0, now_epoch - ts)
    rows.sort(key=lambda row: (str(row["cluster"]), int(row["submit_epoch"]), str(row["job_id"])))
    return rows, failures, stale


_GRES_PYSCRIPT = (
    "python3 -c 'import sys,re\n"
    "T=A=0\n"
    "for line in sys.stdin:\n"
    "    if not line.strip():\n"
    "        continue\n"
    "    c=re.search(r\"CfgTRES=\\S*?gres/gpu=(\\d+)\", line)\n"
    "    a=re.search(r\"AllocTRES=\\S*?gres/gpu=(\\d+)\", line)\n"
    "    if c: T+=int(c.group(1))\n"
    "    if a: A+=int(a.group(1))\n"
    "print(f\"GPUS {A} {T}\")'"
)


def query_snapshot(cluster: str, info: dict[str, str], timeout: int) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "kind": "slurm",
        "idle": None,
        "mix": None,
        "alloc": None,
        "down": None,
        "pending": None,
        "running": None,
        "levelfs": None,
        "gpus_total": None,
        "gpus_used": None,
        "gpu_nodes_total": None,
        "gpu_nodes_mig": None,
        "mem_total_bytes": None,
        "mem_used_bytes": None,
        "pending_by_bucket": {},
    }
    partition = shlex.quote(info["partition"])
    gres_filter = shlex.quote(info.get("snapshot_gres_filter", ""))
    node_filter = shlex.quote(
        info.get("snapshot_node_filter", info.get("snapshot_gres_filter", "gpu:"))
    )
    parts = [
        info["init"],
        f'sinfo -p {partition} --noheader -o "%n %t %G" 2>/dev/null | '
        f'awk -v gf={gres_filter} \'gf=="" || index($3, gf) > 0 {{print $1, $2}}\' | '
        'sort -k1,1 -u | '
        'awk \'{st=$2; gsub(/[^a-z]/,"",st); '
        'if(st=="idle") i++; else if(st=="mix") m++; else if(st=="alloc") a++; else d++} '
        'END{printf "STATES %d %d %d %d\\n", i+0, m+0, a+0, d+0}\'',
        f'sinfo -p {partition} --noheader -o "%n %G" 2>/dev/null | '
        f'awk -v nf={node_filter} \'nf=="" || index($2, nf) > 0 {{print $1, $2}}\' | '
        'sort -k1,1 -u | '
        'awk \'{total++; if(index($2, "nvidia_") > 0) mig++} '
        'END{printf "GPU_NODES %d %d\\n", total+0, mig+0}\'',
        f'echo "PENDING $(squeue -p {partition} -t PENDING --noheader 2>/dev/null | wc -l)"',
        'echo "PEND_JOBS_BEGIN"',
        # Use -O (long format) for tres-per-node and tres-per-job; awk joins into
        # timelimit|tres-per-node|tres-per-job.  GPU info can be in either TRES
        # field depending on how the job was submitted (--gres vs --tres-per-job).
        f"squeue -p {partition} -t PENDING -h -O 'TimeLimit,tres-per-node,tres-per-job'"
        f" 2>/dev/null | awk '{{print $1 \"|\" $2 \"|\" $3}}'",
        'echo "PEND_JOBS_END"',
    ]
    share_account = info.get("share_account") or info.get("account") or info.get("sacct_account")
    if share_account:
        account = shlex.quote(share_account)
        levelfs_user = shlex.quote(LEVELFS_USER)
        parts.append(
            f'LFS=$(sshare -l -A {account} -u {levelfs_user} --parsable2 --noheader 2>/dev/null | '
            'awk -F"|" \'$2=="" && $9!=""{print $9; exit}\'); '
            'echo "LEVELFS ${LFS:-N/A}"'
        )
    else:
        parts.append('echo "LEVELFS N/A"')
    if info.get("group") == "trail":
        # Per-GPU utilisation only matters for the small TRAIL clusters; on the
        # big external systems `scontrol show node -o` would dump thousands of
        # nodes and is unnecessary for the dashboard.
        parts.append(
            "scontrol show node -o 2>/dev/null | " + _GRES_PYSCRIPT
        )
        parts.append(
            f'echo "RUNNING $(squeue -p {partition} -t RUNNING --noheader 2>/dev/null | wc -l)"'
        )
        parts.append(
            f"squeue -p {partition} -t RUNNING -h -O 'UserName,tres-per-node,tres-per-job' "
            "2>/dev/null | awk '"
            "{g=0; for(i=2;i<=NF;i++){if(match($i,/(gres[\\/:])?gpu(:[^:[:space:]]+)?:([0-9]+)/,m)){g=m[3]+0; break}} "
            "if(g>0) printf \"USER_GPU %s %d\\n\", $1, g}'"
        )
        if info.get("gpu_source") == "nvidia-smi":
            # On hosts where SLURM gres counts include non-trainable GPUs
            # (e.g. DGX Display), trust nvidia-smi instead. We emit per-GPU
            # detail so the same data feeds the heatmap plot; the python
            # parser applies any name-based exclusion (e.g. "Display").
            parts.append(
                "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu "
                "--format=csv,noheader,nounits 2>/dev/null | "
                "awk -F',' '{"
                'gsub(/^ +| +$/, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); gsub(/ /, "", $4); '
                'printf "GPU %s|%s|%s|%s\\n", $1, $2, $3, $4'
                "}'"
            )
            parts.append(
                "free -b 2>/dev/null | awk '/^Mem:/ {printf \"MEM %.0f %.0f\\n\", $2 + 0, $2 - $7}'"
            )
    remote_cmd = "; ".join(parts)
    try:
        result = subprocess.run(
            ["ssh", *SSH_OPTS, info["host"], remote_cmd],
            check=False, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{cluster}: snapshot timed out after {timeout}s") from exc
    if result.returncode != 0:
        stderr = result.stderr.strip() or "(no output)"
        raise RuntimeError(f"{cluster}: snapshot failed: {stderr}")

    in_pend_jobs = False
    pending_by_bucket: dict[tuple[int, str], int] = defaultdict(int)
    per_gpu: list[dict[str, object]] = []
    smi_total = 0
    smi_used = 0
    user_gpu: dict[str, dict[str, float]] = defaultdict(lambda: {"allocated_gpus": 0.0})
    exclude_pattern = info.get("gpu_exclude_pattern", "")
    for line in result.stdout.splitlines():
        if line == "PEND_JOBS_BEGIN":
            in_pend_jobs = True
            continue
        if line == "PEND_JOBS_END":
            in_pend_jobs = False
            continue
        if line.startswith("GPU "):
            fields = line[4:].split("|")
            if len(fields) != 4:
                continue
            name = fields[0]
            if exclude_pattern and exclude_pattern in name:
                continue
            try:
                mem_used = int(fields[1])
                mem_total = int(fields[2])
                util = int(fields[3])
            except ValueError:
                continue
            per_gpu.append({
                "name": name,
                "mem_used": mem_used,
                "mem_total": mem_total,
                "util": util,
            })
            smi_total += 1
            if mem_used > 500 or util > 5:
                smi_used += 1
            continue
        if in_pend_jobs:
            if "|" not in line:
                continue
            parts = line.split("|", 2)
            tlim = parts[0].strip()
            tl_secs = parse_time_limit_seconds(tlim)
            bucket = bucket_for_time_limit(tl_secs)
            if bucket is None:
                continue
            # GPU count may be in tres-per-node (col 1) or tres-per-job (col 2)
            gpus = None
            for tres_str in parts[1:]:
                m = GPU_FROM_GRES_RE.search(tres_str.strip())
                if m:
                    g = int(m.group(1))
                    if g > 0:
                        gpus = g
                        break
            cluster_gpu_groups = info.get(
                "pending_gpu_groups",
                info.get("gpu_groups", GPU_GROUPS),
            )
            if gpus is None or gpus not in cluster_gpu_groups:
                continue
            pending_by_bucket[(gpus, bucket)] += 1
            continue
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
        elif tokens[0] == "GPUS" and len(tokens) == 3:
            try:
                snapshot["gpus_used"] = int(tokens[1])
                snapshot["gpus_total"] = int(tokens[2])
            except ValueError:
                pass
        elif tokens[0] == "GPU_NODES" and len(tokens) == 3:
            try:
                snapshot["gpu_nodes_total"] = int(tokens[1])
                snapshot["gpu_nodes_mig"] = int(tokens[2])
            except ValueError:
                pass
        elif tokens[0] == "RUNNING" and len(tokens) == 2:
            try:
                snapshot["running"] = int(tokens[1])
            except ValueError:
                pass
        elif tokens[0] == "USER_GPU" and len(tokens) == 3:
            try:
                user_gpu[tokens[1]]["allocated_gpus"] += float(tokens[2])
            except ValueError:
                pass
        elif tokens[0] == "MEM" and len(tokens) == 3:
            try:
                snapshot["mem_total_bytes"] = int(tokens[1])
                snapshot["mem_used_bytes"] = max(0, int(tokens[2]))
            except ValueError:
                pass
    snapshot["pending_by_bucket"] = dict(pending_by_bucket)
    snapshot["per_gpu"] = per_gpu
    if user_gpu:
        snapshot["user_gpu"] = dict(user_gpu)
        snapshot["users"] = sorted(user_gpu)
    if info.get("gpu_source") == "nvidia-smi" and per_gpu:
        # Prefer nvidia-smi (already filtered via gpu_exclude_pattern) so
        # non-trainable GPUs like the DGX Display don't get attributed as
        # utilisation. The cgroup-masking concern (DGX SSH potentially
        # hiding GPUs) is handled by only overriding when smi sees at
        # least `usable_gpus` trainable GPUs; otherwise fall back to
        # scontrol's CfgTRES/AllocTRES.
        usable = info.get("usable_gpus")
        if not isinstance(usable, int) or smi_total >= usable:
            snapshot["gpus_total"] = smi_total
            snapshot["gpus_used"] = smi_used
    return snapshot


def query_smi_snapshot(
    cluster: str, info: dict[str, str], timeout: int
) -> dict[str, object]:
    """Snapshot a non-SLURM lab host via nvidia-smi + docker mount inspection."""
    snapshot: dict[str, object] = {
        "kind": "smi",
        "gpus_total": None,
        "gpus_used": None,
        "mem_total_bytes": None,
        "mem_used_bytes": None,
        "procs": 0,
        "users": [],
        "containers": 0,
    }
    # Heredoc avoids quoting hell; the remote shell runs the exact script below.
    # When the ssh user lacks docker group access (e.g. Lovelace), `docker
    # inspect` fails — we mark those PIDs with the sentinel `_container_`
    # so the caller can count them separately rather than attributing to root.
    # GPU lines use `|` as the separator since GPU names contain spaces.
    remote_script = r"""
set -e
nvidia-smi --query-gpu=uuid,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits 2>/dev/null \
  | awk -F',' '{
      for(i=1;i<=5;i++) gsub(/^ +| +$/, "", $i);
      gsub(/ /, "", $3); gsub(/ /, "", $4); gsub(/ /, "", $5);
      printf "GPU %s|%s|%s|%s|%s\n", $1, $2, $3, $4, $5
    }'
free -b 2>/dev/null | awk '/^Mem:/ {printf "MEM %.0f %.0f\n", $2 + 0, $2 - $7}'
echo PROC_BEGIN
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
  | awk -F',' '{for(i=1;i<=3;i++) gsub(/^ +| +$/, "", $i); gsub(/ /, "", $3); print $1 "|" $2 "|" $3}' \
  | sort -u | while IFS='|' read pid gpu_uuid used_mem; do
    [ -z "$pid" ] && continue
    cgroup=$(head -1 /proc/$pid/cgroup 2>/dev/null || true)
    cid=""
    if [ -n "$cgroup" ]; then
        cid=$(echo "$cgroup" | grep -oE 'docker-[a-f0-9]+' | head -1 \
              | sed 's/^docker-//' | cut -c1-12)
    fi
    user=""
    if [ -n "$cid" ]; then
        user=$(docker inspect -f '{{range .Mounts}}{{println .Source}}{{end}}' "$cid" 2>/dev/null \
               | grep -oE '^/home/[^/]+' | head -1 | sed 's|^/home/||')
        [ -z "$user" ] && user=_container_
    else
        user=$(ps -o user= -p $pid 2>/dev/null | tr -d ' ')
    fi
    [ -z "$user" ] && user=unknown
    echo "PROC $pid ${gpu_uuid:-unknown} ${used_mem:-0} ${cid:-none} $user"
done
echo PROC_END
who 2>/dev/null | awk '{print "WHO " $1}' | sort -u
"""
    try:
        result = subprocess.run(
            ["ssh", *SSH_OPTS, info["host"], "bash -s"],
            input=remote_script,
            check=False, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{cluster}: snapshot timed out after {timeout}s") from exc
    if result.returncode != 0:
        stderr = result.stderr.strip() or "(no output)"
        raise RuntimeError(f"{cluster}: smi snapshot failed: {stderr}")

    gpus_total = 0
    gpus_used = 0
    per_gpu: list[dict[str, object]] = []
    in_proc = False
    procs: list[tuple[str, str, int, str, str]] = []  # (pid, gpu_uuid, used_mem, container, user)
    who_users: set[str] = set()
    exclude_pattern = info.get("gpu_exclude_pattern", "")
    for line in result.stdout.splitlines():
        if line == "PROC_BEGIN":
            in_proc = True
            continue
        if line == "PROC_END":
            in_proc = False
            continue
        if in_proc:
            toks = line.split()
            if len(toks) == 6 and toks[0] == "PROC":
                try:
                    used_mem = int(toks[3])
                except ValueError:
                    used_mem = 0
                procs.append((toks[1], toks[2], used_mem, toks[4], toks[5]))
            continue
        if line.startswith("GPU "):
            fields = line[4:].split("|")
            if len(fields) != 5:
                continue
            uuid = fields[0]
            name = fields[1]
            if exclude_pattern and exclude_pattern in name:
                continue
            try:
                mem_used = int(fields[2])
                mem_total = int(fields[3])
                util = int(fields[4])
            except ValueError:
                continue
            per_gpu.append({
                "uuid": uuid,
                "name": name,
                "mem_used": mem_used,
                "mem_total": mem_total,
                "util": util,
            })
            gpus_total += 1
            if mem_used > 500 or util > 5:
                gpus_used += 1
            continue
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "WHO" and len(tokens) == 2:
            who_users.add(tokens[1])
        elif tokens[0] == "MEM" and len(tokens) == 3:
            try:
                snapshot["mem_total_bytes"] = int(tokens[1])
                snapshot["mem_used_bytes"] = max(0, int(tokens[2]))
            except ValueError:
                pass

    # Treat _container_ as a sentinel meaning "Docker container we couldn't
    # introspect" — count it separately rather than mixing into the named
    # user list. Likewise, drop bare "root" for Docker-cgroup PIDs since the
    # PID-namespace owner is meaningless on the host.
    named_users: list[str] = []
    unattributed_containers = 0
    user_gpu: dict[str, dict[str, object]] = defaultdict(lambda: {
        "gpu_uuids": set(),
        "mem_used_mb": 0.0,
        "gpu_util_pct_sum": 0.0,
        "gpu_util_gpus": 0.0,
    })
    gpu_users: dict[str, set[str]] = defaultdict(set)
    for _, gpu_uuid, used_mem, cid, u in procs:
        if u == "_container_":
            unattributed_containers += 1
            continue
        if u in ("", "unknown"):
            continue
        if u == "root" and cid and cid != "none":
            unattributed_containers += 1
            continue
        named_users.append(u)
        metrics = user_gpu[u]
        gpu_uuids = metrics["gpu_uuids"]
        if isinstance(gpu_uuids, set) and gpu_uuid and gpu_uuid != "unknown":
            gpu_uuids.add(gpu_uuid)
            gpu_users[gpu_uuid].add(u)
        metrics["mem_used_mb"] = float(metrics["mem_used_mb"]) + max(0, used_mem)
    gpu_util_by_uuid = {
        str(g.get("uuid")): float(g.get("util", 0) or 0)
        for g in per_gpu
        if g.get("uuid")
    }
    for gpu_uuid, users_on_gpu in gpu_users.items():
        if not users_on_gpu:
            continue
        util_share = gpu_util_by_uuid.get(gpu_uuid, 0.0) / len(users_on_gpu)
        for u in users_on_gpu:
            user_gpu[u]["gpu_util_pct_sum"] = float(user_gpu[u]["gpu_util_pct_sum"]) + util_share
            user_gpu[u]["gpu_util_gpus"] = float(user_gpu[u]["gpu_util_gpus"]) + 1.0
    if not named_users and who_users:
        named_users = sorted(who_users)
    container_ids = {c for _, _, _, c, _ in procs if c and c != "none"}

    snapshot["gpus_total"] = gpus_total or None
    snapshot["gpus_used"] = gpus_used if gpus_total else None
    snapshot["procs"] = len(procs)
    snapshot["containers"] = len(container_ids)
    snapshot["unattributed_containers"] = unattributed_containers
    snapshot["per_gpu"] = per_gpu
    snapshot["user_gpu"] = {
        user: {
            "allocated_gpus": len(metrics["gpu_uuids"]) if isinstance(metrics["gpu_uuids"], set) else 0,
            "mem_used_mb": float(metrics["mem_used_mb"]),
            "gpu_util_pct_sum": float(metrics["gpu_util_pct_sum"]),
            "gpu_util_gpus": float(metrics["gpu_util_gpus"]),
        }
        for user, metrics in user_gpu.items()
    }
    # Stable, de-duplicated user list (ordered by first appearance).
    seen: set[str] = set()
    users_ordered: list[str] = []
    for u in named_users:
        if u in seen:
            continue
        seen.add(u)
        users_ordered.append(u)
    snapshot["users"] = users_ordered
    return snapshot


def query_all_snapshots(
    timeout: int,
) -> tuple[dict[str, dict[str, object]], dict[str, int | None]]:
    """Pull live snapshots, falling back to per-cluster cache on failure.

    Returns (snapshots, stale_snap) where stale_snap maps clusters whose
    snapshot came from cache to the cache age in seconds (None when no
    cached snapshot exists).
    """
    snapshots: dict[str, dict[str, object]] = {}
    stale: dict[str, int | None] = {}
    empty_slurm: dict[str, object] = {
        "kind": "slurm",
        "idle": None,
        "mix": None,
        "alloc": None,
        "down": None,
        "pending": None,
        "running": None,
        "levelfs": None,
        "gpus_total": None,
        "gpus_used": None,
        "mem_total_bytes": None,
        "mem_used_bytes": None,
        "pending_by_bucket": {},
        "per_gpu": [],
    }
    empty_smi: dict[str, object] = {
        "kind": "smi",
        "gpus_total": None,
        "gpus_used": None,
        "mem_total_bytes": None,
        "mem_used_bytes": None,
        "procs": 0,
        "users": [],
        "containers": 0,
        "unattributed_containers": 0,
        "per_gpu": [],
    }

    def _empty_for(c: str) -> dict[str, object]:
        return dict(empty_smi if CLUSTER_INFO[c].get("kind") == "smi" else empty_slurm)

    def _query(c: str, info: dict[str, str]) -> dict[str, object]:
        if info.get("kind") == "smi":
            return query_smi_snapshot(c, info, timeout)
        return query_snapshot(c, info, timeout)

    now_epoch = int(datetime.now().timestamp())
    with ThreadPoolExecutor(max_workers=len(CLUSTER_INFO)) as executor:
        futures = {
            executor.submit(_query, cluster, info): cluster
            for cluster, info in CLUSTER_INFO.items()
        }
        for future in as_completed(futures):
            cluster = futures[future]
            try:
                snap = future.result()
                snapshots[cluster] = snap
                _save_cache("snap", cluster, _snapshot_to_jsonable(snap))
                if CLUSTER_INFO[cluster].get("group") == "trail":
                    _append_trail_history(cluster, snap)
            except Exception:
                cached = _load_cache("snap", cluster)
                if cached is None:
                    snapshots[cluster] = _empty_for(cluster)
                    stale[cluster] = None
                else:
                    data, ts = cached
                    if isinstance(data, dict):
                        snapshots[cluster] = _snapshot_from_jsonable(data)
                    else:
                        snapshots[cluster] = _empty_for(cluster)
                    stale[cluster] = max(0, now_epoch - ts)
    return snapshots, stale


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


def aggregate_bucket_stats(
    rows: list[dict[str, object]],
    lookback_hours: int = TABLE_LOOKBACK_HOURS,
    now: int | None = None,
) -> dict[tuple[str, int, str], dict[str, object]]:
    """Per (cluster, gpus, time-limit-bucket): median + count + latest_submit."""
    if now is None:
        now = int(datetime.now().timestamp())
    cutoff = now - lookback_hours * 3600
    by_key: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row["submit_epoch"]) < cutoff:
            continue
        gpus = int(row["gpus"])
        if gpus not in GPU_GROUPS:
            continue
        bucket = bucket_for_time_limit(int(row["time_limit_seconds"]))
        if bucket is None:
            continue
        by_key[(str(row["cluster"]), gpus, bucket)].append(row)

    result: dict[tuple[str, int, str], dict[str, object]] = {}
    for key, key_rows in by_key.items():
        qs_values = [int(r["queue_seconds"]) for r in key_rows]
        latest = max(int(r["submit_epoch"]) for r in key_rows)
        result[key] = {
            "median_seconds": percentile(qs_values, 50.0),
            "count": len(qs_values),
            "latest_submit": latest,
        }
    return result


def compute_cluster_snapshot_rows(
    snapshots: dict[str, dict[str, object]],
    rows: list[dict[str, object]] | None = None,
    now: int | None = None,
    stale_acct: dict[str, int | None] | None = None,
    clusters: tuple[str, ...] = TERMINAL_CLUSTERS,
) -> list[dict[str, object]]:
    if now is None:
        now = int(datetime.now().timestamp())
    cutoff_7d = now - TRAIL_LOOKBACK_DAYS * SECONDS_PER_DAY

    # Pre-aggregate TRAIL members' last-N-day stats from sacct rows.
    gpu_hours_by_cluster: dict[str, float] = defaultdict(float)
    queue_secs_by_cluster: dict[str, list[int]] = defaultdict(list)
    users_by_cluster: dict[str, set[str]] = defaultdict(set)
    if rows:
        for r in rows:
            if int(r["submit_epoch"]) < cutoff_7d:  # type: ignore[arg-type]
                continue
            if r.get("pending"):
                continue
            cluster_name = str(r["cluster"])
            run_secs = int(r.get("run_seconds", 0) or 0)  # type: ignore[arg-type]
            gpus = int(r.get("gpus", 0) or 0)  # type: ignore[arg-type]
            if run_secs > 0 and gpus > 0:
                gpu_hours_by_cluster[cluster_name] += gpus * run_secs / 3600.0
            queue_secs_by_cluster[cluster_name].append(int(r.get("queue_seconds", 0) or 0))  # type: ignore[arg-type]
            u = str(r.get("user") or "").strip()
            if u:
                users_by_cluster[cluster_name].add(u)

    result: list[dict[str, object]] = []
    for cluster in clusters:
        info = CLUSTER_INFO[cluster]
        snap = snapshots.get(cluster, {})
        gpu_nodes_total = snap.get("gpu_nodes_total")
        gpu_nodes_mig = snap.get("gpu_nodes_mig")
        if isinstance(gpu_nodes_total, int):
            active_str = str(gpu_nodes_total)
            if isinstance(gpu_nodes_mig, int) and gpu_nodes_mig > 0:
                active_str = f"{gpu_nodes_total} ({gpu_nodes_mig} MIG)"
        else:
            idle, mix, alloc = snap.get("idle"), snap.get("mix"), snap.get("alloc")
            if None not in (idle, mix, alloc):
                active_str = str(int(idle) + int(mix) + int(alloc))  # type: ignore[arg-type]
            else:
                active_str = "N/A"
        pending = snap.get("pending")
        pending_str = "N/A" if pending is None else str(pending)
        lfs = snap.get("levelfs")
        share_account = info.get("share_account") or info.get("account") or info.get("sacct_account")
        if lfs is None:
            lfs_str = "1.000" if not share_account else "N/A"
        else:
            lfs_str = f"{float(lfs):.3f}"

        # Outage detection: cluster appears in stale_acct (sacct query failed)
        # AND value is None (no cache to fall back to). In that case the 0
        # counts are spurious — show N/A instead.
        sacct_no_data = (
            stale_acct is not None
            and cluster in stale_acct
            and stale_acct[cluster] is None
        )
        qtime_7d_secs: float | None = None
        weekly_target = info.get("weekly_gpu_hrs_target", "")
        if rows is None or sacct_no_data:
            gpu_hrs_7d_str = "N/A"
            qtime_7d_str = "N/A"
            users_7d_str = "N/A"
        else:
            used = f"{gpu_hours_by_cluster.get(cluster, 0.0):.0f}"
            if weekly_target:
                period_target = round(int(weekly_target) * TRAIL_LOOKBACK_DAYS / 7)
                gpu_hrs_7d_str = f"{used} / {period_target}"
            else:
                gpu_hrs_7d_str = used
            q_list = queue_secs_by_cluster.get(cluster, [])
            if q_list:
                qtime_7d_secs = sum(q_list) / len(q_list)
                qtime_7d_str = format_hours_short(qtime_7d_secs)
            else:
                qtime_7d_str = "—"
            user_set = users_by_cluster.get(cluster, set())
            if not user_set:
                users_7d_str = "0"
            else:
                user_list = sorted(user_set)
                users_7d_str = f"{len(user_list)} ({', '.join(user_list)})"

        result.append(
            {
                "cluster": _CLUSTER_ABBR.get(cluster, cluster),
                "location": CLUSTER_LOCATION.get(cluster, "DRAC"),
                "node_type": info["node_type"],
                "active_nodes": active_str,
                "levelfs": lfs_str,
                "trail_gpu_hrs_7d": gpu_hrs_7d_str,
                "trail_qtime_7d": qtime_7d_str,
                "trail_qtime_7d_secs": qtime_7d_secs,
                "trail_users_7d": users_7d_str,
            }
        )
    return result


def format_ratio_pct(used: float, requested: float) -> str:
    if requested <= 0:
        return "—"
    return f"{100.0 * used / requested:.0f}%"


def compute_member_usage_rows(
    rows: list[dict[str, object]],
    clusters: tuple[str, ...],
    now: int | None = None,
    days: int = TRAIL_LOOKBACK_DAYS,
) -> list[dict[str, object]]:
    """Aggregate member resource usage over a cluster cohort."""
    if now is None:
        now = int(datetime.now().timestamp())
    cutoff = now - days * SECONDS_PER_DAY
    cluster_set = set(clusters)
    by_user: dict[str, dict[str, object]] = defaultdict(lambda: {
        "gpu_hours": 0.0,
        "time_used_seconds": 0.0,
        "time_req_seconds": 0.0,
        "cpu_used_seconds": 0.0,
        "cpu_req_seconds": 0.0,
        "mem_used_byte_seconds": 0.0,
        "mem_req_byte_seconds": 0.0,
        "disk_io_bytes": 0.0,
        "low_activity_gpu_hours": 0.0,
        "gpuutil_weighted": 0.0,
        "gpuutil_gpu_hours": 0.0,
        "jobs": 0,
        "clusters": set(),
    })
    for r in rows:
        if int(r.get("submit_epoch", 0) or 0) < cutoff:
            continue
        if r.get("pending"):
            continue
        cluster = str(r.get("cluster") or "")
        if cluster not in cluster_set:
            continue
        user = str(r.get("user") or "").strip()
        if not user:
            continue
        gpus = int(r.get("gpus", 0) or 0)
        run_seconds = int(r.get("run_seconds", 0) or 0)
        time_limit_seconds = int(r.get("time_limit_seconds", 0) or 0)
        if gpus <= 0 or run_seconds <= 0:
            continue
        user_stats = by_user[user]
        gpu_hours = gpus * run_seconds / 3600.0
        user_stats["gpu_hours"] = float(user_stats["gpu_hours"]) + gpu_hours
        user_stats["time_used_seconds"] = float(user_stats["time_used_seconds"]) + run_seconds
        user_stats["time_req_seconds"] = float(user_stats["time_req_seconds"]) + time_limit_seconds
        user_stats["jobs"] = int(user_stats["jobs"]) + 1
        user_clusters = user_stats["clusters"]
        if isinstance(user_clusters, set):
            user_clusters.add(cluster)

        alloc_cpus = int(r.get("alloc_cpus", 0) or 0)
        cpu_req_seconds = alloc_cpus * run_seconds
        cpu_used_seconds = r.get("cpu_total_seconds")
        if isinstance(cpu_used_seconds, (int, float)) and cpu_used_seconds >= 0 and cpu_req_seconds > 0:
            user_stats["cpu_used_seconds"] = float(user_stats["cpu_used_seconds"]) + float(cpu_used_seconds)
            user_stats["cpu_req_seconds"] = float(user_stats["cpu_req_seconds"]) + cpu_req_seconds

        ave_rss_bytes = r.get("ave_rss_bytes")
        req_mem_bytes = r.get("req_mem_bytes")
        if (
            isinstance(ave_rss_bytes, (int, float))
            and isinstance(req_mem_bytes, (int, float))
            and ave_rss_bytes >= 0
            and req_mem_bytes > 0
        ):
            user_stats["mem_used_byte_seconds"] = (
                float(user_stats["mem_used_byte_seconds"]) + float(ave_rss_bytes) * run_seconds
            )
            user_stats["mem_req_byte_seconds"] = (
                float(user_stats["mem_req_byte_seconds"]) + float(req_mem_bytes) * run_seconds
            )

        disk_io_bytes = r.get("disk_io_bytes")
        if isinstance(disk_io_bytes, (int, float)) and disk_io_bytes > 0:
            user_stats["disk_io_bytes"] = float(user_stats["disk_io_bytes"]) + float(disk_io_bytes)

        gpuutil = r.get("gpuutil")
        if isinstance(gpuutil, (int, float)) and gpuutil >= 0:
            avg_gpu_util_pct = min(100.0, float(gpuutil) / max(1, gpus))
            user_stats["gpuutil_weighted"] = (
                float(user_stats["gpuutil_weighted"]) + avg_gpu_util_pct * gpu_hours
            )
            user_stats["gpuutil_gpu_hours"] = float(user_stats["gpuutil_gpu_hours"]) + gpu_hours

        cpu_ratio = (
            float(cpu_used_seconds) / cpu_req_seconds
            if isinstance(cpu_used_seconds, (int, float)) and cpu_req_seconds > 0
            else None
        )
        mem_ratio = (
            float(ave_rss_bytes) / float(req_mem_bytes)
            if isinstance(ave_rss_bytes, (int, float))
            and isinstance(req_mem_bytes, (int, float))
            and req_mem_bytes > 0
            else None
        )
        per_gpu_gpuutil = (
            float(gpuutil) / max(1, gpus)
            if isinstance(gpuutil, (int, float)) and gpuutil >= 0
            else None
        )
        cpus_per_gpu_used = (
            float(cpu_used_seconds) / (run_seconds * gpus)
            if isinstance(cpu_used_seconds, (int, float)) and run_seconds > 0 and gpus > 0
            else None
        )
        mem_gb_used = (
            float(ave_rss_bytes) / (1024 ** 3)
            if isinstance(ave_rss_bytes, (int, float))
            else None
        )
        if (
            gpu_hours >= 1.0
            and run_seconds >= 10 * 60
            and cpus_per_gpu_used is not None
            and cpus_per_gpu_used > 0
            and cpus_per_gpu_used < 1.0
            and mem_gb_used is not None
            and mem_gb_used < 10.0
            and (per_gpu_gpuutil is None or per_gpu_gpuutil < 30)
        ):
            user_stats["low_activity_gpu_hours"] = (
                float(user_stats["low_activity_gpu_hours"]) + gpu_hours
            )

    result: list[dict[str, object]] = []
    for user, stats in by_user.items():
        gpu_hours = float(stats["gpu_hours"])
        gpuutil_gpu_hours = float(stats["gpuutil_gpu_hours"])
        if gpuutil_gpu_hours >= 0.5:
            gpuutil_avg = float(stats["gpuutil_weighted"]) / gpuutil_gpu_hours
            gpuutil_display = f"{gpuutil_avg:.0f}% ({gpuutil_gpu_hours:.0f} GPU hrs)"
        else:
            gpuutil_display = "-"
        result.append({
            "member": user,
            "gpu_hours": gpu_hours,
            "gpu_hours_display": f"{gpu_hours:.0f}",
            "clusters": (
                ", ".join(
                    _CLUSTER_ABBR.get(c, c)
                    for c in sorted(stats["clusters"])  # type: ignore[arg-type]
                )
                if isinstance(stats["clusters"], set) and stats["clusters"]
                else "-"
            ),
            "time_ratio": format_ratio_pct(
                float(stats["time_used_seconds"]),
                float(stats["time_req_seconds"]),
            ),
            "cpu_ratio": format_ratio_pct(
                float(stats["cpu_used_seconds"]),
                float(stats["cpu_req_seconds"]),
            ),
            "mem_ratio": format_ratio_pct(
                float(stats["mem_used_byte_seconds"]),
                float(stats["mem_req_byte_seconds"]),
            ),
            "disk_io": format_bytes_short(float(stats["disk_io_bytes"])),
            "low_activity_gpu_hours": float(stats["low_activity_gpu_hours"]),
            "low_activity_display": f"{float(stats['low_activity_gpu_hours']):.0f}",
            "gpuutil_display": gpuutil_display,
            "jobs": int(stats["jobs"]),
        })
    result.sort(key=lambda r: (-float(r["gpu_hours"]), str(r["member"])))
    return result


def _trail_avg_occupancy_pct(cluster: str, days: int = 7) -> float | None:
    """Mean of `gpus_used / gpus_total` across the last `days` history samples.

    Returns None when history is empty or every sample reports `total=0`.
    """
    hist = _load_trail_history(cluster, days=days)
    if not hist:
        return None
    fractions: list[float] = []
    for rec in hist:
        total = int(rec["total"])  # type: ignore[arg-type]
        if total <= 0:
            continue
        used = int(rec["used"])  # type: ignore[arg-type]
        fractions.append(min(1.0, max(0.0, used / total)))
    if not fractions:
        return None
    return 100.0 * (sum(fractions) / len(fractions))


def _trail_avg_mem_pct(cluster: str, days: int = 7) -> float | None:
    """Mean of `mem_used / mem_total` (system RAM) across `days` of samples.

    Returns None when no sample in the window carries the optional mem fields.
    """
    hist = _load_trail_history(cluster, days=days)
    if not hist:
        return None
    fractions: list[float] = []
    for rec in hist:
        total = rec.get("mem_total_bytes")
        used = rec.get("mem_used_bytes")
        if not isinstance(total, int) or total <= 0:
            continue
        if not isinstance(used, int) or used < 0:
            continue
        fractions.append(min(1.0, max(0.0, used / total)))
    if not fractions:
        return None
    return 100.0 * (sum(fractions) / len(fractions))


def _trail_unique_users_list(cluster: str, days: int = 7) -> list[str] | None:
    """Sorted list of distinct users that appeared in any sample over `days`.

    None when no history yet (so the column reads "—" instead of empty).
    """
    hist = _load_trail_history(cluster, days=days)
    if not hist:
        return None
    seen: set[str] = set()
    for rec in hist:
        for u in rec.get("users") or []:  # type: ignore[union-attr]
            if u:
                seen.add(str(u))
    return sorted(seen)


def compute_utias_member_usage_rows(
    days: int = TRAIL_LOOKBACK_DAYS,
    now: int | None = None,
) -> list[dict[str, object]]:
    """Approximate per-member UTIAS usage from sampled host history.

    New samples carry per-user GPU process data where available. Older samples
    only have host-level used GPU counts, so they fall back to dividing each
    sample interval equally across named active users.
    """
    if now is None:
        now = int(datetime.now().timestamp())
    by_user: dict[str, dict[str, object]] = defaultdict(lambda: {
        "gpu_hours": 0.0,
        "gpu_util_weighted": 0.0,
        "gpu_util_gpu_hours": 0.0,
        "samples": 0,
        "days_used": set(),
        "clusters": set(),
    })
    max_interval = 6 * 3600
    for cluster in TRAIL_CLUSTERS:
        hist = _load_trail_history(cluster, days=days)
        if not hist:
            continue
        for idx, rec in enumerate(hist):
            ts = int(rec.get("ts", 0) or 0)
            if ts <= 0:
                continue
            if idx + 1 < len(hist):
                next_ts = int(hist[idx + 1].get("ts", ts) or ts)
            else:
                next_ts = now
            interval = max(0, min(max_interval, next_ts - ts))
            if interval <= 0:
                continue
            user_gpu = rec.get("user_gpu") or {}
            if isinstance(user_gpu, dict) and user_gpu:
                for user, metrics in user_gpu.items():
                    if not isinstance(metrics, dict):
                        continue
                    allocated_gpus = float(metrics.get("allocated_gpus", 0.0) or 0.0)
                    if allocated_gpus <= 0:
                        continue
                    gpu_hours = allocated_gpus * interval / 3600.0
                    stats = by_user[str(user)]
                    stats["gpu_hours"] = float(stats["gpu_hours"]) + gpu_hours
                    stats["samples"] = int(stats["samples"]) + 1
                    days_used = stats["days_used"]
                    if isinstance(days_used, set):
                        days_used.add(datetime.fromtimestamp(ts).date().isoformat())
                    clusters = stats["clusters"]
                    if isinstance(clusters, set):
                        clusters.add(cluster)
                    gpu_util_gpus = float(metrics.get("gpu_util_gpus", 0.0) or 0.0)
                    gpu_util_pct_sum = float(metrics.get("gpu_util_pct_sum", 0.0) or 0.0)
                    if gpu_util_gpus > 0:
                        avg_util = gpu_util_pct_sum / gpu_util_gpus
                        stats["gpu_util_weighted"] = (
                            float(stats["gpu_util_weighted"]) + avg_util * gpu_hours
                        )
                        stats["gpu_util_gpu_hours"] = (
                            float(stats["gpu_util_gpu_hours"]) + gpu_hours
                        )
                continue

            users = [str(u) for u in rec.get("users") or [] if str(u)]
            if not users:
                continue
            used = int(rec.get("used", 0) or 0)
            if used <= 0:
                continue
            share_gpu_hours = used * interval / 3600.0 / len(users)
            for user in users:
                stats = by_user[user]
                stats["gpu_hours"] = float(stats["gpu_hours"]) + share_gpu_hours
                stats["samples"] = int(stats["samples"]) + 1
                days_used = stats["days_used"]
                if isinstance(days_used, set):
                    days_used.add(datetime.fromtimestamp(ts).date().isoformat())
                clusters = stats["clusters"]
                if isinstance(clusters, set):
                    clusters.add(cluster)

    result: list[dict[str, object]] = []
    for user, stats in by_user.items():
        clusters = stats["clusters"]
        cluster_labels = (
            ", ".join(_CLUSTER_ABBR.get(c, c) for c in sorted(clusters))
            if isinstance(clusters, set) and clusters else "-"
        )
        gpu_hours = float(stats["gpu_hours"])
        util_gpu_hours = float(stats["gpu_util_gpu_hours"])
        gpu_util_display = (
            f"{float(stats['gpu_util_weighted']) / util_gpu_hours:.0f}% ({util_gpu_hours:.0f} GPU hrs)"
            if util_gpu_hours >= 0.5 else "-"
        )
        result.append({
            "member": user,
            "gpu_hours": gpu_hours,
            "gpu_hours_display": f"{gpu_hours:.0f}",
            "gpu_util_display": gpu_util_display,
            "servers": cluster_labels,
            "samples": int(stats["samples"]),
            "days_used": len(stats["days_used"]) if isinstance(stats["days_used"], set) else 0,
        })
    result.sort(key=lambda r: (-float(r["gpu_hours"]), str(r["member"])))
    return result


def utias_history_available_days(
    days: int = TRAIL_LOOKBACK_DAYS,
    now: int | None = None,
) -> float | None:
    if now is None:
        now = int(datetime.now().timestamp())
    oldest: int | None = None
    newest: int | None = None
    cutoff = now - days * SECONDS_PER_DAY
    for cluster in TRAIL_CLUSTERS:
        for rec in _load_trail_history(cluster, days=days):
            ts = int(rec.get("ts", 0) or 0)
            if ts < cutoff:
                continue
            oldest = ts if oldest is None else min(oldest, ts)
            newest = ts if newest is None else max(newest, ts)
    if oldest is None:
        return None
    end = max(newest or oldest, now)
    return max(0.0, min(float(days), (end - oldest) / SECONDS_PER_DAY))


def compute_trail_snapshot_rows(
    snapshots: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Build rows for the TRAIL-cluster table.

    Columns: Cluster · Node Type · GPUs (used/total) · Users · Pending.
    Pending is N/A for non-SLURM hosts; users are summarised as
    "<count> (<top user>+N)" when the list gets long.
    """
    result: list[dict[str, object]] = []
    for cluster in TRAIL_CLUSTERS:
        if cluster not in CLUSTER_INFO:
            continue
        info = CLUSTER_INFO[cluster]
        snap = snapshots.get(cluster, {})
        gpus_total = snap.get("gpus_total")
        gpus_used = snap.get("gpus_used")
        # Per-cluster cap: e.g. DGX's 5th GPU is display-only and not
        # schedulable, so we present 4 even though SLURM reports 5.
        usable = info.get("usable_gpus")
        if isinstance(usable, int):
            if gpus_total is not None:
                gpus_total = min(int(gpus_total), usable)
            if gpus_used is not None:
                gpus_used = min(int(gpus_used), usable)
        if gpus_total is None:
            gpus_str = "N/A"
        elif gpus_used is None:
            gpus_str = f"?/{int(gpus_total)}"
        else:
            gpus_str = f"{int(gpus_used)}/{int(gpus_total)}"

        if info.get("kind") == "smi":
            users = snap.get("users") or []
            if not isinstance(users, list):
                users = []
            if not users:
                users_str = "0"
            elif len(users) == 1:
                users_str = f"1 ({users[0]})"
            else:
                users_str = f"{len(users)} ({users[0]} +{len(users) - 1})"
        else:
            # SLURM TRAIL clusters don't have per-user breakdown.
            users_str = "—"

        avg_pct = _trail_avg_occupancy_pct(cluster, days=TRAIL_LOOKBACK_DAYS)
        util_7d_str = "—" if avg_pct is None else f"{avg_pct:.0f}%"

        mem_total = snap.get("mem_total_bytes")
        mem_used = snap.get("mem_used_bytes")
        if isinstance(mem_total, int) and mem_total > 0 and isinstance(mem_used, int):
            mem_pct = 100.0 * mem_used / mem_total
            mem_util_str = f"{mem_pct:.0f}%"
        else:
            mem_util_str = "—"

        avg_mem_pct = _trail_avg_mem_pct(cluster, days=TRAIL_LOOKBACK_DAYS)
        mem_util_7d_str = "—" if avg_mem_pct is None else f"{avg_mem_pct:.0f}%"

        users_7d_list = _trail_unique_users_list(cluster, days=TRAIL_LOOKBACK_DAYS)
        if users_7d_list is None:
            users_7d_str = "—"
        elif not users_7d_list:
            users_7d_str = "0"
        else:
            users_7d_str = f"{len(users_7d_list)} ({', '.join(users_7d_list)})"

        result.append(
            {
                "cluster": _CLUSTER_ABBR.get(cluster, cluster),
                "node_type": info.get("node_type", ""),
                "gpus": gpus_str,
                "mem_util": mem_util_str,
                "users": users_str,
                "util_7d": util_7d_str,
                "util_7d_pct": avg_pct,
                "mem_util_7d": mem_util_7d_str,
                "mem_util_7d_pct": avg_mem_pct,
                "users_7d": users_7d_str,
            }
        )
    return result


def compute_bucket_table_rows(
    rows: list[dict[str, object]],
    snapshots: dict[str, dict[str, object]],
    gpus: int,
    now: int | None = None,
) -> list[dict[str, object]]:
    if now is None:
        now = int(datetime.now().timestamp())
    aggregates = aggregate_bucket_stats(rows, now=now)

    rows_out: list[dict[str, object]] = []
    for cluster in TERMINAL_CLUSTERS:
        snap = snapshots.get(cluster, {})
        pending_by_bucket = snap.get("pending_by_bucket") or {}
        if not isinstance(pending_by_bucket, dict):
            pending_by_bucket = {}
        cells: list[dict[str, object]] = []
        for label, _, _ in TIME_BUCKETS:
            agg = aggregates.get((cluster, gpus, label))
            pending = int(pending_by_bucket.get((gpus, label), 0))
            if agg is None:
                cells.append(
                    {
                        "label": label,
                        "median_seconds": None,
                        "age_hours": None,
                        "pending": pending,
                        "count": 0,
                        "is_best": False,
                    }
                )
            else:
                cells.append(
                    {
                        "label": label,
                        "median_seconds": float(agg["median_seconds"]),  # type: ignore[arg-type]
                        "age_hours": (now - int(agg["latest_submit"])) / 3600,  # type: ignore[arg-type]
                        "pending": pending,
                        "count": int(agg["count"]),  # type: ignore[arg-type]
                        "is_best": False,
                    }
                )
        rows_out.append({"cluster": _CLUSTER_ABBR.get(cluster, cluster), "cells": cells})

    for label, _, _ in TIME_BUCKETS:
        best_cluster = None
        best_median: float | None = None
        for r in rows_out:
            cell = next(
                (c for c in r["cells"] if c["label"] == label),  # type: ignore[arg-type,index]
                None,
            )
            if cell is None or cell["median_seconds"] is None:
                continue
            ms = float(cell["median_seconds"])  # type: ignore[arg-type]
            if best_median is None or ms < best_median:
                best_cluster = r["cluster"]
                best_median = ms
        if best_cluster is not None:
            for r in rows_out:
                if r["cluster"] == best_cluster:
                    cell = next(
                        (c for c in r["cells"] if c["label"] == label),  # type: ignore[arg-type,index]
                        None,
                    )
                    if cell is not None:
                        cell["is_best"] = True
    return rows_out


def format_bucket_cell_text(cell: dict[str, object], with_pending: bool = False) -> str:
    if cell["median_seconds"] is None:
        return "–"
    median = format_hours_short(float(cell["median_seconds"]))  # type: ignore[arg-type]
    age = float(cell["age_hours"])  # type: ignore[arg-type]
    pending = int(cell["pending"])  # type: ignore[arg-type]
    if with_pending and pending > 0:
        return f"{median} ({age:.1f}h ago, {pending} pend)"
    return f"{median} ({age:.1f}h ago)"


def _box_table(
    headers: list[str], widths: list[int], body: list[list[str]],
) -> list[str]:
    inner = ["─" * (w + 1) for w in widths]
    top = "┌" + "┬".join(inner) + "┐"
    mid = "├" + "┼".join(inner) + "┤"
    bot = "└" + "┴".join(inner) + "┘"

    def fmt_row(cells: list[str]) -> str:
        return "│" + "│".join(
            f" {cell:<{widths[i]}}" for i, cell in enumerate(cells)
        ) + "│"

    out = [top, fmt_row(headers), mid]
    for row in body:
        out.append(fmt_row(row))
    out.append(bot)
    return out


def print_banner() -> None:
    title = "  ✦   TRAIL  GPU  CLUSTER  QUEUE  STATUS   ✦  "
    inner_w = 92

    def _starfield(n: int, offset: int = 0) -> str:
        pattern = " ·   *   .   ✦   ·   ★   .   ·   *   ✦   ·   .   ★   ·   "
        rep = pattern * (n // len(pattern) + 2)
        return rep[offset:offset + n]

    side = (inner_w - len(title)) // 2
    left_stars = _starfield(side, 0)
    right_stars = _starfield(inner_w - len(title) - side, 7)
    print("╔" + "═" * inner_w + "╗")
    print("║" + _starfield(inner_w, 3) + "║")
    print("║" + left_stars + title + right_stars + "║")
    print("║" + _starfield(inner_w, 11) + "║")
    print("╚" + "═" * inner_w + "╝")


def print_cluster_snapshot(
    snapshots: dict[str, dict[str, object]],
    rows: list[dict[str, object]] | None = None,
    stale_acct: dict[str, int | None] | None = None,
) -> None:
    snap_rows = compute_cluster_snapshot_rows(
        snapshots, rows=rows, stale_acct=stale_acct,
    )
    n_days = TRAIL_LOOKBACK_DAYS
    headers = [
        "Cluster", "Node Type", "GPU NODES", "LEVELFS",
        f"{n_days}D GPU HRS", f"{n_days}D QUEUE TIME", f"{n_days}D USERS",
    ]
    widths = [14, 14, 9, 7, 12, 14, 28]
    body = [
        [
            str(r["cluster"]),
            str(r["node_type"]),
            str(r["active_nodes"]),
            str(r["levelfs"]),
            str(r["trail_gpu_hrs_7d"]),
            str(r["trail_qtime_7d"]),
            str(r["trail_users_7d"]),
        ]
        for r in snap_rows
    ]
    for line in _box_table(headers, widths, body):
        print(line)


def print_trail_snapshot(snapshots: dict[str, dict[str, object]]) -> None:
    rows = compute_trail_snapshot_rows(snapshots)
    if not rows:
        return
    headers = [
        "Cluster", "Node Type", "GPU UTIL", "MEM UTIL", "Users",
        f"{TRAIL_LOOKBACK_DAYS}D GPU UTIL", f"{TRAIL_LOOKBACK_DAYS}D MEM UTIL",
        f"{TRAIL_LOOKBACK_DAYS}D USERS",
    ]
    widths = [14, 18, 9, 9, 16, 11, 11, 24]
    body = [
        [
            str(r["cluster"]),
            str(r["node_type"]),
            str(r["gpus"]),
            str(r["mem_util"]),
            str(r["users"]),
            str(r["util_7d"]),
            str(r["mem_util_7d"]),
            str(r["users_7d"]),
        ]
        for r in rows
    ]
    for line in _box_table(headers, widths, body):
        print(line)


def print_member_usage(
    title: str,
    rows: list[dict[str, object]],
    clusters: tuple[str, ...],
) -> None:
    print(title)
    usage_rows = compute_member_usage_rows(rows, clusters)
    headers = [
        "Member",
        f"{TRAIL_LOOKBACK_DAYS}D GPU HRS",
        f"{TRAIL_LOOKBACK_DAYS}D GPU Util",
        f"{TRAIL_LOOKBACK_DAYS}D Inactive GPU Hrs",
        f"{TRAIL_LOOKBACK_DAYS}D CLUSTERS",
        f"{TRAIL_LOOKBACK_DAYS}D CPU Used/Req",
        f"{TRAIL_LOOKBACK_DAYS}D Mem Used/Req",
    ]
    widths = [16, 12, 22, 27, 28, 17, 17]
    body = [
        [
            str(r["member"]),
            str(r["gpu_hours_display"]),
            str(r["gpuutil_display"]),
            str(r["low_activity_display"]),
            str(r["clusters"]),
            str(r["cpu_ratio"]),
            str(r["mem_ratio"]),
        ]
        for r in usage_rows
    ]
    if not body:
        body = [["—", "0", "-", "0", "-", "—", "—"]]
    for line in _box_table(headers, widths, body):
        print(line)


def print_utias_member_usage() -> None:
    print("Member Usage UTIAS Server:")
    usage_rows = compute_utias_member_usage_rows()
    headers = [
        "Member",
        f"{TRAIL_LOOKBACK_DAYS}D GPU HRS",
        f"{TRAIL_LOOKBACK_DAYS}D GPU Util",
        f"{TRAIL_LOOKBACK_DAYS}D Servers",
        f"{TRAIL_LOOKBACK_DAYS}D DAYS USED",
    ]
    widths = [16, 12, 22, 30, 11]
    body = [
        [
            str(r["member"]),
            str(r["gpu_hours_display"]),
            str(r["gpu_util_display"]),
            str(r["servers"]),
            str(r["days_used"]),
        ]
        for r in usage_rows
    ]
    if not body:
        body = [["—", "0", "-", "-", "0"]]
    for line in _box_table(headers, widths, body):
        print(line)


def print_bucket_table(
    rows: list[dict[str, object]],
    snapshots: dict[str, dict[str, object]],
    gpus: int,
) -> None:
    print()
    print(f"{gpus}-GPU jobs (median queue time over last {TABLE_LOOKBACK_HOURS}h):")
    table_rows = compute_bucket_table_rows(rows, snapshots, gpus)
    headers = ["Cluster"] + [label for label, _, _ in TIME_BUCKETS]
    widths = [14] + [18] * len(TIME_BUCKETS)
    body: list[list[str]] = []
    for r in table_rows:
        row_cells = [str(r["cluster"])]
        for cell in r["cells"]:  # type: ignore[arg-type]
            text = format_bucket_cell_text(cell, with_pending=False)
            if cell.get("is_best"):
                text = "*" + text
            row_cells.append(text)
        body.append(row_cells)
    for line in _box_table(headers, widths, body):
        print(line)


def print_all(series: dict[str, list[dict[str, object]]], window_days: int) -> None:
    low_label = f"daily_p{int(PERCENTILE_LOW)}_seconds"
    high_label = f"daily_p{int(PERCENTILE_HIGH)}_seconds"
    print(f"cluster,day,daily_median_seconds,{low_label},{high_label},samples")

    def fmt(value: object) -> str:
        return "" if value is None else str(int(round(float(value))))

    for cluster in TERMINAL_CLUSTERS:
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
    rows: list[dict[str, object]],
    plot_path: Path,
    window_days: int,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    def _bucket_label(cluster: str, gpus: int, bucket: str) -> str:
        return f"{_CLUSTER_ABBR.get(cluster, cluster)} {bucket} {gpus} GPU"

    def _top_bucket_keys(
        rows: list[dict[str, object]],
        window_days: int,
        limit: int = 5,
    ) -> list[tuple[str, int, str]]:
        latest = latest_epoch(rows)
        if latest is None:
            return []
        cutoff = latest - window_days * SECONDS_PER_DAY
        counts: dict[tuple[str, int, str], int] = defaultdict(int)
        last_seen: dict[tuple[str, int, str], int] = defaultdict(int)
        queue_vals: dict[tuple[str, int, str], list[int]] = defaultdict(list)
        for row in rows:
            submit_epoch = int(row["submit_epoch"])
            if submit_epoch < cutoff:
                continue
            cluster = str(row["cluster"])
            gpus = int(row["gpus"])
            if cluster not in DEFAULT_CLUSTERS or gpus not in GPU_GROUPS:
                continue
            bucket = bucket_for_time_limit(int(row["time_limit_seconds"]))
            if bucket is None:
                continue
            key = (cluster, gpus, bucket)
            counts[key] += 1
            queue_vals[key].append(int(row["queue_seconds"]))
            if submit_epoch > last_seen[key]:
                last_seen[key] = submit_epoch

        # Exclude combinations whose median queue time is near-zero — they
        # start immediately and produce an invisible line at y=0.
        MIN_MEDIAN_SECONDS = 60
        eligible = {k for k, vals in queue_vals.items() if percentile(vals, 50) >= MIN_MEDIAN_SECONDS}

        cluster_order = {cluster: idx for idx, cluster in enumerate(DEFAULT_CLUSTERS)}
        gpu_order = {gpus: idx for idx, gpus in enumerate(GPU_GROUPS)}
        bucket_order = {label: idx for idx, (label, _, _) in enumerate(TIME_BUCKETS)}
        ordered = sorted(
            (k for k in counts if k in eligible),
            key=lambda key: (
                -counts[key],
                cluster_order.get(key[0], len(cluster_order)),
                bucket_order.get(key[2], len(bucket_order)),
                gpu_order.get(key[1], len(gpu_order)),
                -last_seen.get(key, 0),
            ),
        )
        return ordered[:limit]

    def _bucket_series(
        rows: list[dict[str, object]],
        selected_keys: list[tuple[str, int, str]],
        window_days: int,
    ) -> tuple[
        dict[tuple[str, int, str], list[dict[str, object]]],
        dict[tuple[str, int, str], list[int]],
    ]:
        if not rows or not selected_keys:
            return {}, {}
        latest = latest_epoch(rows)
        if latest is None:
            return {}, {}

        earliest_day = day_start(min(int(row["submit_epoch"]) for row in rows))
        max_day = day_start(latest)
        plot_start_day = max(earliest_day, max_day - (window_days - 1) * SECONDS_PER_DAY)
        cutoff = latest - window_days * SECONDS_PER_DAY
        selected = set(selected_keys)
        values_by_key_day: dict[tuple[str, int, str], dict[int, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        values_by_key: dict[tuple[str, int, str], list[int]] = defaultdict(list)

        for row in rows:
            submit_epoch = int(row["submit_epoch"])
            if submit_epoch < cutoff:
                continue
            cluster = str(row["cluster"])
            gpus = int(row["gpus"])
            if cluster not in DEFAULT_CLUSTERS or gpus not in GPU_GROUPS:
                continue
            bucket = bucket_for_time_limit(int(row["time_limit_seconds"]))
            if bucket is None:
                continue
            key = (cluster, gpus, bucket)
            if key not in selected:
                continue
            queue_seconds = int(row["queue_seconds"])
            values_by_key_day[key][day_start(submit_epoch)].append(queue_seconds)
            values_by_key[key].append(queue_seconds)

        series: dict[tuple[str, int, str], list[dict[str, object]]] = {}
        for key in selected_keys:
            per_day = values_by_key_day.get(key, {})
            days: list[dict[str, object]] = []
            day = plot_start_day
            while day <= max_day:
                day_values = per_day.get(day, [])
                days.append(
                    {
                        "day": day,
                        "daily_median": percentile(day_values, 50.0),
                        "daily_low": percentile(day_values, PERCENTILE_LOW),
                        "daily_high": percentile(day_values, PERCENTILE_HIGH),
                        "samples": len(day_values),
                    }
                )
                day += SECONDS_PER_DAY
            series[key] = days
        return series, values_by_key

    top_keys = _top_bucket_keys(rows, window_days, limit=5)
    series, bucket_values = _bucket_series(rows, top_keys, window_days)
    _latest = latest_epoch(rows)
    _max_day = day_start(_latest) if _latest is not None else int(datetime.now().timestamp())

    import math as _math
    all_medians = [
        float(row["daily_median"])
        for key in top_keys
        for row in series.get(key, [])
        if row["daily_median"] is not None
    ]
    max_median = max(all_medians) if all_medians else 3600
    cap_hours = min(max(1, _math.ceil(max_median / 3600)), 24)
    cap = cap_hours * 3600

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_trend, ax_hours) = plt.subplots(
        2,
        1,
        figsize=(14, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
    )

    def _plot_daily_gpu_hours_ax(ax: object) -> None:
        colors = {"narval": "#1f77b4", "trillium": "#2ca02c", "killarney": "#d62728"}
        cluster_labels = {"narval": "Narval", "trillium": "Trillium", "killarney": "Killarney"}

        if not rows:
            ax.text(
                0.5,
                0.5,
                "no data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#9ca3af",
                fontsize=13,
            )
            ax.axis("off")
            return

        earliest_day = day_start(min(int(r["submit_epoch"]) for r in rows))
        max_day = day_start(max(int(r["submit_epoch"]) for r in rows))
        plot_start = max(earliest_day, max_day - (window_days - 1) * SECONDS_PER_DAY)

        gpu_hrs_by_cluster_day: dict[str, dict[int, float]] = {c: defaultdict(float) for c in colors}
        for row in rows:
            cluster = str(row["cluster"])
            if cluster not in colors:
                continue
            run_secs = int(row.get("run_seconds") or 0)
            gpus = int(row.get("gpus") or 0)
            if run_secs > 0 and gpus > 0:
                gpu_hrs_by_cluster_day[cluster][day_start(int(row["submit_epoch"]))] += gpus * run_secs / 3600

        for cluster in DEFAULT_CLUSTERS:
            if cluster not in colors:
                continue
            data = gpu_hrs_by_cluster_day[cluster]
            days, gpu_hrs = [], []
            d = plot_start
            while d <= max_day:
                days.append(day_label(d))
                gpu_hrs.append(data.get(d, 0.0))
                d += SECONDS_PER_DAY
            ax.plot(days, gpu_hrs, color=colors[cluster], linewidth=2.0, marker="o", markersize=4,
                    label=cluster_labels[cluster])

        ax.set_ylabel("GPU hours", fontsize=15)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, linewidth=0.5, alpha=0.35)
        ax.legend(fontsize=13, loc="upper right")
        ax.set_xlim(
            day_label(max_day - (window_days - 1) * SECONDS_PER_DAY),
            day_label(max_day + SECONDS_PER_DAY),
        )
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    if not top_keys:
        ax_trend.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            transform=ax_trend.transAxes,
            color="#9ca3af",
            fontsize=13,
        )
        ax_trend.axis("off")
    else:
        cmap = plt.get_cmap("tab10")

        for idx, key in enumerate(top_keys):
            cluster, gpus, bucket = key
            cluster_series = series.get(key, [])
            if not cluster_series:
                continue
            days = [day_label(int(row["day"])) for row in cluster_series]
            daily_median = _to_floats([row["daily_median"] for row in cluster_series])
            daily_low = _to_floats([row["daily_low"] for row in cluster_series])
            daily_high = _to_floats([row["daily_high"] for row in cluster_series])
            color = cmap(idx % 10)
            display_name = _bucket_label(cluster, gpus, bucket)
            sample_count = len(bucket_values.get(key, []))

            # Drop days with no data so the line connects across gaps instead of breaking.
            valid = [i for i, v in enumerate(daily_median) if not np.isnan(v)]
            vdays = [days[i] for i in valid]
            vmedian = [daily_median[i] for i in valid]
            vlow = [daily_low[i] if not np.isnan(daily_low[i]) else daily_median[i] for i in valid]
            vhigh = [daily_high[i] if not np.isnan(daily_high[i]) else daily_median[i] for i in valid]

            ax_trend.fill_between(
                vdays,
                vlow,
                vhigh,
                color=color,
                alpha=0.18,
                linewidth=0,
                label="_nolegend_",
            )
            ax_trend.plot(
                vdays,
                vmedian,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=4,
                label=f"{display_name} (n={sample_count})",
            )

        ax_trend.set_ylabel("Queue Time (hrs) - Median & IQR", fontsize=15)
        ax_trend.set_ylim(0, cap)
        ax_trend.yaxis.set_major_locator(mticker.MultipleLocator(3600))
        ax_trend.yaxis.set_major_formatter(lambda value, _: f"{int(round(value / 3600))}")
        ax_trend.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_trend.tick_params(axis="both", labelsize=14)
        ax_trend.grid(True, linewidth=0.5, alpha=0.35)

        handles, labels = ax_trend.get_legend_handles_labels()
        handle_for_label = dict(zip(labels, handles))
        ordered: list[tuple[object, str]] = []
        for key in top_keys:
            cluster, gpus, bucket = key
            line_label = f"{_bucket_label(cluster, gpus, bucket)} (n={len(bucket_values.get(key, []))})"
            if line_label in handle_for_label:
                ordered.append((handle_for_label[line_label], line_label))
        ax_trend.legend(
            [h for h, _ in ordered],
            [l for _, l in ordered],
            ncol=2,
            fontsize=13,
            loc="upper right",
        )

    _plot_daily_gpu_hours_ax(ax_hours)
    ax_hours.set_xlabel("Day", fontsize=15)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def plot_cluster_location_map(
    clusters: tuple[str, ...] = MAP_CLUSTERS,
) -> bytes:
    """PNG Plate Carree lon/lat map of Canada cluster locations."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import io as _io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Coarse southern Canada outline in Plate Carree lon/lat coordinates.
    canada_outline = [
        (-141.0, 65.0), (-138.0, 62.0), (-134.5, 58.5), (-130.5, 54.0),
        (-127.8, 50.3), (-124.8, 48.3), (-123.2, 49.0), (-110.0, 49.0),
        (-95.2, 49.0), (-88.5, 48.0), (-84.8, 46.5), (-82.5, 42.0),
        (-79.2, 43.0), (-75.3, 44.8), (-71.4, 45.0), (-67.8, 47.1),
        (-62.8, 45.9), (-57.5, 48.0), (-54.0, 52.0), (-58.0, 56.5),
        (-64.0, 60.5), (-72.0, 64.0), (-76.0, 65.0), (-141.0, 65.0),
    ]
    province_borders = [
        [(-120.0, 49.0), (-120.0, 57.0)],  # BC / AB
        [(-110.0, 49.0), (-110.0, 57.0)],  # AB / SK
        [(-102.0, 49.0), (-102.0, 57.0)],  # SK / MB
        [(-95.2, 49.0), (-95.1, 52.8), (-94.8, 56.0), (-94.0, 57.0)],  # MB / ON
        [(-74.7303, 45.0213), (-75.7003, 45.4201), (-79.4663, 46.3168),
         (-79.4663, 57.0)],  # ON / QC
        [(-71.5, 45.0), (-70.8, 47.0), (-70.5, 57.0)],  # QC / NB
    ]
    offsets = [
        (0.0, 0.0), (0.55, 0.32), (-0.55, 0.32), (0.55, -0.32),
        (-0.55, -0.32), (0.0, 0.65), (0.0, -0.65),
    ]
    map_label_override = {
        "killarney": "Killarney",
        "killarney_h100": "Killarney",
        "utias": "TRAIL",
    }
    map_color_override = {
        "narval": "#f97316",
    }
    label_offsets = {
        "nibi": (-0.25, 0.22, "right", "bottom"),
        "trillium": (0.18, -0.18, "left", "top"),
        "killarney": (0.25, 0.10, "left", "bottom"),
        "rorqual": (-0.25, 0.22, "right", "bottom"),
        "narval": (0.0, -0.28, "center", "top"),
        "tamia": (0.25, 0.22, "left", "bottom"),
        "utias": (0.0, 0.32, "center", "bottom"),
        "apollo": (-0.18, -0.18, "right", "top"),
    }
    marker_y_offsets = {
        "narval": -0.18,
        "trillium": -0.18,
        "killarney": -0.18,
    }
    fig, ax = plt.subplots(figsize=(13, 6.7))
    ix, iy = zip(*canada_outline)
    ax.fill(ix, iy, facecolor="#f8fafc", edgecolor="#334155", linewidth=1.4, zorder=0)
    for border in province_borders:
        bx, by = zip(*border)
        ax.plot(bx, by, color="#94a3b8", linewidth=0.9, linestyle="-", zorder=1)

    plotted_by_site: dict[tuple[float, float], int] = defaultdict(int)
    plotted_labels: set[str] = set()
    for cluster in clusters:
        geo = CLUSTER_GEO.get(cluster)
        if not geo:
            continue
        label = map_label_override.get(cluster, _CLUSTER_ABBR.get(cluster, cluster))
        if label in plotted_labels:
            continue
        plotted_labels.add(label)
        lon = float(geo["lon"])
        lat = float(geo["lat"])
        key = (round(lon, 1), round(lat, 1))
        idx = plotted_by_site[key]
        plotted_by_site[key] += 1
        dx, dy = offsets[idx % len(offsets)]
        color = map_color_override.get(cluster, _CLUSTER_COLOR.get(cluster, "#111827"))
        x, y = lon + dx, lat + dy + marker_y_offsets.get(cluster, 0.0)
        ax.scatter(
            x, y, s=95, color=color, edgecolor="white", linewidth=1.5,
            zorder=3, clip_on=True,
        )
        tx, ty, ha, va = label_offsets.get(cluster, (0.45, 0.12, "left", "center"))
        ax.text(
            x + tx,
            y + ty,
            label,
            ha=ha,
            va=va,
            fontsize=10,
            fontweight="bold",
            color=color,
            zorder=4,
            clip_on=True,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-120, -110, -100, -90, -80, -70])
    ax.set_yticks([45, 50, 55])
    ax.set_xlim(-129.67, -61.06)
    ax.set_ylim(42, 55)
    ax.grid(False)
    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        labelbottom=False,
        labelleft=False,
    )
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_facecolor("#e0f2fe")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    buf = _io.BytesIO()
    fig.savefig(buf, dpi=150, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


_CLUSTER_ABBR = {"narval": "Narval", "trillium": "Trillium", "killarney": "Killarney-L40S", "killarney_h100": "Killarney-H100", "fir": "Fir", "rorqual": "Rorqual", "tamia": "TamIA", "vulcan": "Vulcan", "nibi": "Nibi", "utias": "UTIAS", "dgx": "DGX", "apollo": "Apollo", "turing": "Turing", "lovelace": "Lovelace", "ums": "UMS", "um1": "UM1", "um2": "UM2", "um3": "UM3"}
_CLUSTER_COLOR = {"narval": "#2563eb", "trillium": "#2ca02c", "killarney": "#d62728", "killarney_h100": "#ff7f0e", "fir": "#9467bd", "rorqual": "#17becf", "tamia": "#8c564b", "vulcan": "#eab308", "nibi": "#ec4899", "utias": "#475569"}


def _fmt_queue(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    h = seconds / 3600
    if h < 1:
        return f"{h * 60:.0f}m"
    return f"{int(round(h))}h"


def plot_split_queue_heatmap(
    rows: list[dict[str, object]],
    title: str,
    now: int | None = None,
    median_lookback_hours: int = 14 * 24,
    clusters: tuple[str, ...] = DEFAULT_CLUSTERS,
) -> bytes:
    """PNG heatmap with split-triangle cells: upper-left = last observed, lower-right = 14d median.

    Each cell's diagonal divides it into:
      - upper-left triangle (↖): most-recent completed-job queue time, OR longest current pending
        wait if that exceeds it (matches plot_last_queue_heatmap semantics);
      - lower-right triangle (↘): median queue time over the last `median_lookback_hours`.
    Colors share one log-scale colorbar so divergence between live state and typical state is
    visible at a glance.
    """
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import io as _io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import numpy as np

    if now is None:
        now = int(datetime.now().timestamp())

    last_comp: dict[tuple, tuple[int, int]] = {}
    max_pend: dict[tuple, int] = {}
    for row in rows:
        cluster = str(row["cluster"])
        gpus = int(row["gpus"])
        if gpus not in _heatmap_gpu_groups(cluster):
            continue
        bucket = bucket_for_time_limit(int(row["time_limit_seconds"]))
        if bucket is None:
            continue
        key = (cluster, gpus, bucket)
        qs = int(row["queue_seconds"])
        pending = bool(row["pending"])
        submit_epoch = int(row["submit_epoch"])
        if pending:
            if key not in max_pend or qs > max_pend[key]:
                max_pend[key] = qs
        else:
            if key not in last_comp or submit_epoch > last_comp[key][0]:
                last_comp[key] = (submit_epoch, qs)

    aggregates = aggregate_bucket_stats(rows, now=now, lookback_hours=median_lookback_hours)

    cluster_col_start: dict[str, int] = {}
    col_keys: list[tuple[str, int]] = []
    for cluster in clusters:
        cluster_col_start[cluster] = len(col_keys)
        col_keys.extend((cluster, g) for g in _heatmap_gpu_groups(cluster))
    col_labels = [f"{g}GPU" for _, g in col_keys]
    row_labels = [label for label, _, _ in TIME_BUCKETS]
    n_rows, n_cols = len(TIME_BUCKETS), len(col_keys)

    last_h = np.full((n_rows, n_cols), np.nan)
    med_h = np.full((n_rows, n_cols), np.nan)
    last_age_d = np.full((n_rows, n_cols), np.nan)
    last_is_now = np.zeros((n_rows, n_cols), dtype=bool)
    med_n = np.zeros((n_rows, n_cols), dtype=int)

    for ri, (bucket_label, _, _) in enumerate(TIME_BUCKETS):
        for ci, (cluster, gpus) in enumerate(col_keys):
            key = (cluster, gpus, bucket_label)
            comp = last_comp.get(key)
            pend_qs = max_pend.get(key)
            comp_qs = comp[1] if comp else None
            comp_epoch = comp[0] if comp else None
            if pend_qs is not None and (comp_qs is None or pend_qs >= comp_qs):
                last_h[ri, ci] = pend_qs / 3600
                last_age_d[ri, ci] = 0.0
                last_is_now[ri, ci] = True
            elif comp_qs is not None:
                last_h[ri, ci] = comp_qs / 3600
                last_age_d[ri, ci] = (now - comp_epoch) / 86400
            agg = aggregates.get(key)
            if agg and agg["median_seconds"] is not None:
                med_h[ri, ci] = float(agg["median_seconds"]) / 3600
                med_n[ri, ci] = int(agg["count"])

    finite = np.concatenate([last_h[~np.isnan(last_h)], med_h[~np.isnan(med_h)]])
    if finite.size:
        vmin = max(float(finite.min()), 1 / 60)
        vmax = max(float(finite.max()), vmin * 2)
    else:
        vmin, vmax = 1 / 60, 24.0

    cmap = plt.cm.RdYlGn_r  # type: ignore[attr-defined]
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    def _txt_color(rgba):
        luma = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        return "black" if luma > 0.45 else "white"

    fig, ax = plt.subplots(figsize=(max(12.5, 1.35 * n_cols), 5.3))
    for ri in range(n_rows):
        for ci in range(n_cols):
            ul_tri = [(ci - 0.5, ri - 0.5), (ci - 0.5, ri + 0.5), (ci + 0.5, ri + 0.5)]
            lr_tri = [(ci - 0.5, ri - 0.5), (ci + 0.5, ri - 0.5), (ci + 0.5, ri + 0.5)]

            ul_val = last_h[ri, ci]
            if np.isnan(ul_val):
                ax.add_patch(Polygon(ul_tri, closed=True, color="#e5e7eb", zorder=0))
                ax.text(ci - 0.20, ri + 0.20, "–",
                        ha="center", va="center", fontsize=13, color="#9ca3af")
            else:
                color = cmap(norm(max(ul_val, vmin)))
                ax.add_patch(Polygon(ul_tri, closed=True, color=color, zorder=0))
                tc = _txt_color(color)
                ax.text(ci - 0.18, ri + 0.26, _fmt_queue(ul_val * 3600),
                        ha="center", va="center", fontsize=13, fontweight="bold", color=tc)
                d = last_age_d[ri, ci]
                if last_is_now[ri, ci]:
                    age_str = "now"
                elif d < 1 / 24:
                    age_str = f"{d * 24 * 60:.0f}m"
                elif d < 1:
                    age_str = f"{d * 24:.0f}h"
                else:
                    age_str = f"{int(round(d))}d"
                ax.text(ci - 0.27, ri + 0.07, age_str,
                        ha="center", va="center", fontsize=11.5, color=tc)

            lr_val = med_h[ri, ci]
            if np.isnan(lr_val):
                ax.add_patch(Polygon(lr_tri, closed=True, color="#e5e7eb", zorder=0))
                ax.text(ci + 0.20, ri - 0.20, "–",
                        ha="center", va="center", fontsize=13, color="#9ca3af")
            else:
                color = cmap(norm(max(lr_val, vmin)))
                ax.add_patch(Polygon(lr_tri, closed=True, color=color, zorder=0))
                tc = _txt_color(color)
                ax.text(ci + 0.18, ri - 0.18, _fmt_queue(lr_val * 3600),
                        ha="center", va="center", fontsize=13, fontweight="bold", color=tc)
                ax.text(ci + 0.27, ri - 0.36, f"n={med_n[ri, ci]}",
                        ha="center", va="center", fontsize=11.5, color=tc)

    for cluster in list(clusters)[1:]:
        ax.axvline(cluster_col_start[cluster] - 0.5, color="#374151", linewidth=2)
    for gi, cluster in enumerate(clusters):
        n_g = len(_heatmap_gpu_groups(cluster))
        cx = cluster_col_start[cluster] + (n_g - 1) / 2
        ax.text(cx, n_rows - 0.5 + 0.10, _CLUSTER_ABBR[cluster].upper(),
                ha="center", va="bottom", fontsize=17, fontweight="bold",
                color=_CLUSTER_COLOR[cluster], transform=ax.transData)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5 + 0.45)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=14)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=15)
    ax.set_ylabel("max time limit", fontsize=14, color="#6b7280")
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # type: ignore[attr-defined]
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.01)
    cb.set_label("Queue time (h)", fontsize=14)
    cb.ax.tick_params(labelsize=12)

    fig.tight_layout()
    buf = _io.BytesIO()
    fig.savefig(buf, dpi=150, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_distributions(
    rows: list[dict[str, object]],
    title: str = "",
    now: int | None = None,
    lookback_hours: int = TABLE_LOOKBACK_HOURS,
) -> bytes:
    """PNG with one box-plot subplot per cluster.

    Within each subplot, x-axis cycles through (GPU group × time bucket)
    ordered ascending by GPU count: 1G3h, 1G12h, 1G1d, 2G3h, 2G12h, ...
    The ``title`` argument is kept for caller compatibility but is not
    rendered (the surrounding HTML already labels the section).
    """
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import io as _io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if now is None:
        now = int(datetime.now().timestamp())
    cutoff = now - lookback_hours * 3600

    # Buckets to display per GPU group — short/normal/day, no 3d/7d here.
    bucket_labels = [b for b, _, _ in TIME_BUCKETS if b in ("3h", "12h", "1d")]
    abbr = _CLUSTER_ABBR

    clusters = [c for c in DEFAULT_CLUSTERS if CLUSTER_INFO[c].get("lab_users_dir")]
    n_clusters = len(clusters)
    fig, axes = plt.subplots(1, n_clusters, figsize=(4.7 * n_clusters, 5.0), sharey=True)
    if n_clusters == 1:
        axes = [axes]

    all_box_data: list[list[float]] = []
    n_buckets = len(bucket_labels)
    for ci, cluster in enumerate(clusters):
        ax = axes[ci]
        cluster_color = _CLUSTER_COLOR[cluster]
        # Drop 8GPU jobs from this view — usage too sparse to be informative.
        gpu_groups = sorted(
            g for g in CLUSTER_INFO[cluster].get("gpu_groups", GPU_GROUPS) if g < 8
        )
        col_keys = [(g, b) for g in gpu_groups for b in bucket_labels]
        n_cols = len(col_keys)

        box_data, box_labels, positions = [], [], []
        for pos, (gpus, bucket_label) in enumerate(col_keys):
            vals = [
                int(r["queue_seconds"]) / 3600
                for r in rows
                if (str(r["cluster"]) == cluster
                    and int(r["gpus"]) == gpus
                    and bucket_for_time_limit(int(r["time_limit_seconds"])) == bucket_label
                    and int(r["submit_epoch"]) >= cutoff)
            ]
            if len(vals) >= 2:
                box_data.append(vals)
                all_box_data.append(vals)
                box_labels.append(f"{gpus}G{bucket_label}")
                positions.append(pos)

        if not box_data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#9ca3af", fontsize=11)
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(
                [f"{g}G{b}" for g, b in col_keys], fontsize=10, rotation=45, ha="right",
            )
        else:
            bplot = ax.boxplot(box_data, positions=positions, widths=0.55,
                               patch_artist=True, showfliers=True,
                               flierprops={"marker": ".", "markersize": 4, "alpha": 0.5})
            for patch in bplot["boxes"]:
                patch.set_facecolor(cluster_color)
                patch.set_alpha(0.55)
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(
                [f"{g}G{b}" for g, b in col_keys], fontsize=10, rotation=45, ha="right",
            )
            ax.grid(True, axis="y", alpha=0.3, linewidth=0.8)
        ax.set_xlim(-0.5, n_cols - 0.5)

        # Vertical separators between GPU groups within this subplot.
        for gi in range(1, len(gpu_groups)):
            ax.axvline(gi * n_buckets - 0.5, color="#d1d5db", linewidth=1, linestyle="--")

        ax.set_title(abbr[cluster], fontsize=12, fontweight="bold", color=cluster_color)
        ax.set_ylabel("Queue time (h)" if ci == 0 else "", fontsize=11)
        ax.tick_params(labelsize=10)

    if all_box_data:
        import statistics
        max_median = max(statistics.median(v) for v in all_box_data)
        axes[0].set_ylim(0, max_median + 1)
    else:
        axes[0].set_ylim(0, 12)

    fig.tight_layout()
    buf = _io.BytesIO()
    fig.savefig(buf, dpi=150, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_trail_gpu_grid(
    snapshots: dict[str, dict[str, object]],
) -> bytes:
    """PNG grid: rows = TRAIL hosts, columns = GPU index.

    Cell color encodes memory utilisation (0-100%); cell text shows compute
    utilisation. Cells beyond a host's GPU count are greyed out.
    """
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import io as _io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np

    rows: list[tuple[str, list[dict[str, object]]]] = []
    for cluster in TRAIL_CLUSTERS:
        if cluster not in CLUSTER_INFO:
            continue
        snap = snapshots.get(cluster, {})
        per_gpu = snap.get("per_gpu") or []
        if not isinstance(per_gpu, list):
            per_gpu = []
        rows.append((cluster, per_gpu))
    if not rows:
        rows = [("(no hosts)", [])]
    max_gpus = max((len(g) for _, g in rows), default=1)
    max_gpus = max(max_gpus, 1)

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * max_gpus + 2.5), 0.85 * n_rows + 1.2))

    cmap = plt.cm.RdYlGn_r  # type: ignore[attr-defined]
    norm = mcolors.Normalize(vmin=0, vmax=100)

    def _txt_color(rgba):
        luma = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        return "black" if luma > 0.55 else "white"

    for ri, (cluster, per_gpu) in enumerate(rows):
        for ci in range(max_gpus):
            if ci < len(per_gpu):
                g = per_gpu[ci]
                mem_used = float(g.get("mem_used", 0))
                mem_total = float(g.get("mem_total", 0)) or 1.0
                util = float(g.get("util", 0))
                mem_pct = max(0.0, min(100.0, 100.0 * mem_used / mem_total))
                color = cmap(norm(mem_pct))
                ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1, color=color, zorder=0))
                ax.text(
                    ci, ri,
                    f"{int(round(mem_pct))}%\n{int(round(util))}u",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color=_txt_color(color),
                )
            else:
                ax.add_patch(plt.Rectangle(
                    (ci - 0.5, ri - 0.5), 1, 1, color="#e5e7eb", zorder=0,
                ))
                ax.text(ci, ri, "–", ha="center", va="center", fontsize=11, color="#9ca3af")

    ax.set_xticks(range(max_gpus))
    ax.set_xticklabels([f"GPU {i}" for i in range(max_gpus)], fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [_CLUSTER_ABBR.get(c, c) for c, _ in rows], fontsize=11, fontweight="bold",
    )
    ax.set_xlim(-0.5, max_gpus - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # type: ignore[attr-defined]
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Memory used (%)", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    fig.tight_layout()
    buf = _io.BytesIO()
    fig.savefig(buf, dpi=150, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_trail_utilization_timeline(
    days: int = 7,
) -> bytes:
    """PNG small-multiples plot: % GPUs in use vs time, one row per TRAIL host.

    Samples are read from `_append_trail_history` and bucketed into hourly
    means so per-refresh noise doesn't drown the signal. Each host gets its
    own subplot sharing the x-axis. If no host has data yet, an explanatory
    placeholder is drawn instead.
    """
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import io as _io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    series: list[tuple[str, list[dict[str, object]]]] = []
    for cluster in TRAIL_CLUSTERS:
        if cluster not in CLUSTER_INFO:
            continue
        hist = _load_trail_history(cluster, days=days)
        series.append((cluster, hist))

    plotted = [(c, h) for c, h in series if h]

    if not plotted:
        fig, ax = plt.subplots(figsize=(13, 4.0))
        ax.text(
            0.5, 0.5,
            (
                "No history yet — samples accumulate every refresh.\n"
                "Plot fills in over the next few days."
            ),
            ha="center", va="center",
            transform=ax.transAxes,
            color="#6b7280", fontsize=12,
        )
        ax.axis("off")
        fig.tight_layout()
        buf = _io.BytesIO()
        fig.savefig(buf, dpi=150, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    palette = {
        "dgx": "#7c3aed",      # purple
        "apollo": "#0ea5e9",   # cyan
        "turing": "#f97316",   # orange
        "lovelace": "#db2777", # magenta
        "ums": "#16a34a",      # green
        "um1": "#ca8a04",      # gold
        "um2": "#0d9488",      # teal
        "um3": "#dc2626",      # red
    }

    def _hourly_mean(hist: list[dict[str, object]]) -> tuple[list[datetime], list[float]]:
        bins: dict[int, list[float]] = defaultdict(list)
        for r in hist:
            total = int(r["total"])  # type: ignore[arg-type]
            if total <= 0:
                continue
            used = int(r["used"])  # type: ignore[arg-type]
            ts = int(r["ts"])  # type: ignore[arg-type]
            bucket = ts - (ts % 3600)
            bins[bucket].append(100.0 * used / total)
        if not bins:
            return [], []
        items = sorted(bins.items())
        xs = [datetime.fromtimestamp(t) for t, _ in items]
        ys = [sum(vs) / len(vs) for _, vs in items]
        return xs, ys

    n = len(plotted)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(13, max(2.6, 0.9 * n + 1.2)),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    if n == 1:
        axes = [axes]

    window_end = datetime.now()
    window_start = window_end - timedelta(days=days)

    for ax, (cluster, hist) in zip(axes, plotted):
        xs, ys = _hourly_mean(hist)
        color = palette.get(cluster, "#374151")
        if xs:
            ax.fill_between(xs, ys, 0, color=color, alpha=0.22, linewidth=0)
            ax.plot(xs, ys, color=color, linewidth=1.4)
        ax.set_xlim(window_start, window_end)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(["0", "50", "100"], fontsize=8)
        ax.set_ylabel(
            _CLUSTER_ABBR.get(cluster, cluster),
            rotation=0, ha="right", va="center",
            fontsize=10, fontweight="600",
            labelpad=8,
        )
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        ax.grid(True, axis="x", alpha=0.15, linewidth=0.5)
        ax.tick_params(axis="x", labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[0].set_title(
        f"% of GPUs in use — hourly mean, last {days} days",
        fontsize=11, loc="left", pad=6,
    )

    fig.tight_layout()
    buf = _io.BytesIO()
    fig.savefig(buf, dpi=150, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_pending_heatmap(
    snapshots: dict[str, dict[str, object]],
    title: str = "Current Pending Jobs — All Cluster Users (live squeue)",
    clusters: tuple[str, ...] = DEFAULT_CLUSTERS,
) -> bytes:
    """PNG heatmap of pending job counts from squeue (all users, all groups)."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import io as _io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np

    col_keys = [(c, g) for c in clusters for g in _heatmap_gpu_groups(c)]
    col_labels = [f"{g}GPU" for _, g in col_keys]
    row_labels = [label for label, _, _ in TIME_BUCKETS]
    n_rows, n_cols = len(TIME_BUCKETS), len(col_keys)

    data = np.zeros((n_rows, n_cols), dtype=float)
    for ri, (bucket_label, _, _) in enumerate(TIME_BUCKETS):
        for ci, (cluster, gpus) in enumerate(col_keys):
            pbb = snapshots.get(cluster, {}).get("pending_by_bucket") or {}
            data[ri, ci] = float(pbb.get((gpus, bucket_label), 0))

    has = data > 0
    vmax = float(data.max()) if has.any() else 1.0
    vmin = 0.5

    cmap = plt.cm.YlOrRd  # type: ignore[attr-defined]
    norm = mcolors.LogNorm(vmin=vmin, vmax=max(vmax, 1.0))

    # Pre-compute per-cluster column start indices for separators and labels.
    cluster_col_start: dict[str, int] = {}
    ci = 0
    for c in clusters:
        cluster_col_start[c] = ci
        ci += len(_heatmap_gpu_groups(c))

    fig, ax = plt.subplots(figsize=(max(13.5, 1.45 * n_cols), 5.5))
    for ri in range(n_rows):
        for ci in range(n_cols):
            count = int(data[ri, ci])
            if count == 0:
                ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1, color="#e5e7eb", zorder=0))
                ax.text(ci, ri, "0", ha="center", va="center", fontsize=20, color="#9ca3af")
            else:
                color = cmap(norm(count))
                ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1, color=color, zorder=0))
                luma = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                txt_color = "black" if luma > 0.45 else "white"
                ax.text(ci, ri, str(count), ha="center", va="center",
                        fontsize=20, fontweight="bold", color=txt_color)

    for cluster in list(clusters)[1:]:
        ax.axvline(cluster_col_start[cluster] - 0.5, color="#374151", linewidth=2)
    for cluster in clusters:
        n_g = len(_heatmap_gpu_groups(cluster))
        cx = cluster_col_start[cluster] + (n_g - 1) / 2
        ax.text(cx, n_rows - 0.5 + 0.10, _CLUSTER_ABBR[cluster].upper(),
                ha="center", va="bottom", fontsize=17, fontweight="bold",
                color=_CLUSTER_COLOR[cluster], transform=ax.transData)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5 + 0.45)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=15)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=16)
    ax.set_ylabel("max time limit", fontsize=14, color="#6b7280")
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    ax.set_title("")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # type: ignore[attr-defined]
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.01)
    cb.set_label("# pending jobs", fontsize=14)
    cb.ax.tick_params(labelsize=12)

    fig.tight_layout()
    buf = _io.BytesIO()
    fig.savefig(buf, dpi=150, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def compute_recommendation_rows(
    rows: list[dict[str, object]],
    now: int | None = None,
) -> list[dict[str, object]]:
    """For each (bucket, gpus) cell: best cluster + median + count."""
    aggregates = aggregate_bucket_stats(rows, now=now)
    result = []
    for bucket_label, _, _ in TIME_BUCKETS:
        row: dict[str, object] = {"bucket": bucket_label}
        for gpus in GPU_GROUPS:
            best: dict[str, object] | None = None
            for cluster in DEFAULT_CLUSTERS:
                agg = aggregates.get((cluster, gpus, bucket_label))
                if not agg or agg["median_seconds"] is None:
                    continue
                if best is None or float(agg["median_seconds"]) < float(best["median_seconds"]):  # type: ignore[arg-type]
                    best = {"cluster": cluster, **agg}
            row[gpus] = best
        result.append(row)
    return result


def write_html(
    html_path: Path,
    plot_path: Path | None,
    rows: list[dict[str, object]],
    snapshots: dict[str, dict[str, object]],
    refresh_seconds: int,
    window_days: int = 7,
    stale: dict[str, int | None] | None = None,
    stale_acct: dict[str, int | None] | None = None,
) -> None:
    stale = stale or {}
    # Map abbreviated cluster label → staleness key for table decoration.
    stale_by_abbr = {
        _CLUSTER_ABBR.get(c, c): age for c, age in stale.items()
    }
    def _cluster_summary_table_html(
        clusters: tuple[str, ...],
        *,
        include_location: bool = False,
        location_last: bool = False,
        include_levelfs: bool = True,
        include_users: bool = True,
    ) -> str:
        snap_rows = compute_cluster_snapshot_rows(
            snapshots, rows=rows, stale_acct=stale_acct, clusters=clusters,
        )
        html_rows = []
        for r in snap_rows:
            cluster_label = str(r["cluster"])
            row_class = ""
            cluster_cell = f'<td class="cluster">{html.escape(cluster_label)}</td>'
            if cluster_label in stale_by_abbr:
                age = stale_by_abbr[cluster_label]
                tag_text = "outage; no cache" if age is None else f"stale · {format_staleness(age)}"
                row_class = ' class="stale"'
                cluster_cell = (
                    '<td class="cluster">'
                    f"{html.escape(cluster_label)} "
                    f'<span class="stale-tag">({html.escape(tag_text)})</span>'
                    "</td>"
                )
            qcolor = _color_for_queue_seconds(r.get("trail_qtime_7d_secs"))  # type: ignore[arg-type]
            qtime_cell = (
                f'<td style="color:{qcolor}; font-weight:600;">{html.escape(str(r["trail_qtime_7d"]))}</td>'
                if qcolor
                else f'<td>{html.escape(str(r["trail_qtime_7d"]))}</td>'
            )
            cells = [cluster_cell]
            if include_location and not location_last:
                cells.append(f'<td>{html.escape(str(r["location"]))}</td>')
            cells.extend([
                f'<td>{html.escape(str(r["node_type"]))}</td>',
                f'<td>{html.escape(str(r["active_nodes"]))}</td>',
            ])
            if include_levelfs:
                cells.append(f'<td>{html.escape(str(r["levelfs"]))}</td>')
            cells.extend([
                f'<td>{html.escape(str(r["trail_gpu_hrs_7d"]))}</td>',
                qtime_cell,
            ])
            if include_users:
                cells.append(f'<td>{html.escape(str(r["trail_users_7d"]))}</td>')
            if include_location and location_last:
                cells.append(f'<td>{html.escape(str(r["location"]))}</td>')
            html_rows.append(f"        <tr{row_class}>" + "".join(cells) + "</tr>")
        return "\n".join(html_rows)

    drac_allocated_table_html = _cluster_summary_table_html(DRAC_ALLOCATED_SUMMARY_CLUSTERS)
    other_cluster_table_html = _cluster_summary_table_html(
        OTHER_SUMMARY_CLUSTERS,
        include_location=True,
        location_last=True,
        include_users=False,
    )
    def _member_usage_table_html(clusters: tuple[str, ...]) -> str:
        usage_rows = compute_member_usage_rows(rows, clusters)
        html_rows = []
        for r in usage_rows:
            html_rows.append(
                "        <tr>"
                f'<td class="cluster">{html.escape(str(r["member"]))}</td>'
                f'<td>{html.escape(str(r["gpu_hours_display"]))}</td>'
                f'<td>{html.escape(str(r["gpuutil_display"]))}</td>'
                f'<td>{html.escape(str(r["low_activity_display"]))}</td>'
                f'<td>{html.escape(str(r["clusters"]))}</td>'
                "</tr>"
            )
        return "\n".join(html_rows) if html_rows else (
            '        <tr><td colspan="5" class="muted">No member usage in this window.</td></tr>'
        )

    allocated_member_usage_table_html = _member_usage_table_html(
        MEMBER_ALLOCATED_USAGE_CLUSTERS
    )
    other_member_usage_table_html = _member_usage_table_html(
        MEMBER_OTHER_USAGE_CLUSTERS
    )

    utias_member_usage_rows = compute_utias_member_usage_rows()
    utias_member_usage_html_rows = []
    for r in utias_member_usage_rows:
        utias_member_usage_html_rows.append(
            "        <tr>"
            f'<td class="cluster">{html.escape(str(r["member"]))}</td>'
            f'<td>{html.escape(str(r["gpu_hours_display"]))}</td>'
            f'<td>{html.escape(str(r["servers"]))}</td>'
            f'<td>{html.escape(str(r["days_used"]))}</td>'
            "</tr>"
        )
    utias_member_usage_table_html = (
        "\n".join(utias_member_usage_html_rows)
        if utias_member_usage_html_rows
        else '        <tr><td colspan="4" class="muted">No UTIAS member usage in this window.</td></tr>'
    )

    def _cluster_list_text(clusters: tuple[str, ...]) -> str:
        subtitle_abbr = {**_CLUSTER_ABBR, "killarney": "Killarney", "killarney_h100": "Killarney"}
        labels: list[str] = []
        for c in clusters:
            label = subtitle_abbr.get(c, c)
            if label not in labels:
                labels.append(label)
        return ", ".join(labels)

    def _gpu_util_cluster_list_text(clusters: tuple[str, ...]) -> str:
        util_clusters = tuple(c for c in clusters if c in GPU_UTIL_USAGE_CLUSTERS)
        return _cluster_list_text(util_clusters) if util_clusters else "none"

    allocated_member_usage_subtitle = (
        f"{_cluster_list_text(MEMBER_ALLOCATED_USAGE_CLUSTERS)} - "
        f"GPU util only on {_gpu_util_cluster_list_text(MEMBER_ALLOCATED_USAGE_CLUSTERS)} - "
        "Inactive = cpus/gpu<1 & mem<10GB"
    )
    other_member_usage_subtitle = (
        f"{_cluster_list_text(MEMBER_OTHER_USAGE_CLUSTERS)} - "
        f"GPU util only on {_gpu_util_cluster_list_text(MEMBER_OTHER_USAGE_CLUSTERS)} - "
        "Inactive = cpus/gpu<1 & mem<10GB"
    )
    utias_member_usage_subtitle = (
        f"{_cluster_list_text(TRAIL_CLUSTERS)}"
    )
    utias_days_available = utias_history_available_days()
    if utias_days_available is None:
        utias_history_alert_html = (
            '  <p class="usage-alert">⚠ No UTIAS member history is available yet; '
            "this table will populate as new samples are collected.</p>"
        )
    elif utias_days_available < TRAIL_LOOKBACK_DAYS - 0.5:
        utias_history_alert_html = (
            '  <p class="usage-alert">⚠ Only showing data from '
            f"{max(1, round(utias_days_available))} days temporarily due to recent tracking changes.</p>"
        )
    else:
        utias_history_alert_html = ""

    trail_rows = compute_trail_snapshot_rows(snapshots)
    trail_html_rows: list[str] = []
    for r in trail_rows:
        cluster_label = str(r["cluster"])
        row_class = ""
        cluster_cell = f'<td class="cluster">{html.escape(cluster_label)}</td>'
        if cluster_label in stale_by_abbr:
            age = stale_by_abbr[cluster_label]
            tag_text = "outage; no cache" if age is None else f"stale · {format_staleness(age)}"
            row_class = ' class="stale"'
            cluster_cell = (
                '<td class="cluster">'
                f"{html.escape(cluster_label)} "
                f'<span class="stale-tag">({html.escape(tag_text)})</span>'
                "</td>"
            )
        ucolor = _color_for_util_pct(r.get("util_7d_pct"))  # type: ignore[arg-type]
        util_cell = (
            f'<td style="color:{ucolor}; font-weight:600;">{html.escape(str(r["util_7d"]))}</td>'
            if ucolor
            else f'<td>{html.escape(str(r["util_7d"]))}</td>'
        )
        trail_html_rows.append(
            f"        <tr{row_class}>"
            f"{cluster_cell}"
            f'<td>{html.escape(str(r["node_type"]))}</td>'
            f'<td>{html.escape(str(r["gpus"]))}</td>'
            f'<td>{html.escape(str(r["users"]))}</td>'
            f'{util_cell}'
            f'<td>{html.escape(str(r["users_7d"]))}</td>'
            "</tr>"
        )
    trail_table_html = "\n".join(trail_html_rows) if trail_html_rows else (
        '        <tr><td colspan="6" class="muted">No TRAIL hosts configured.</td></tr>'
    )

    if stale:
        items = []
        for cluster in sorted(stale):
            label = _CLUSTER_ABBR.get(cluster, cluster)
            ip = CLUSTER_INFO.get(cluster, {}).get("ip", "")
            host_str = (
                f' <span class="host-tag">[{html.escape(ip)}]</span>' if ip else ""
            )
            age = stale[cluster]
            if age is None:
                detail = "outage; no cached data available"
            else:
                detail = f"last successful update {format_staleness(age)}"
            items.append(
                f"<li><b>{html.escape(label)}</b>{host_str} — {html.escape(detail)}</li>"
            )
        stale_banner_html = (
            '  <div class="stale-banner" role="status">\n'
            '    <strong>⚠ Outage detected — some cluster data is stale.</strong>\n'
            '    <span class="stale-banner-sub">Showing the last successful query for each affected cluster.</span>\n'
            f'    <ul>\n{chr(10).join("      " + i for i in items)}\n    </ul>\n'
            '  </div>'
        )
    else:
        stale_banner_html = ""

    # --- inline PNG helper ---
    def _png_data_url(png_bytes: bytes) -> str:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _img(png_bytes: bytes, alt: str) -> str:
        return f'<img src="{_png_data_url(png_bytes)}" alt="{html.escape(alt)}">'

    # --- history plot ---
    if plot_path is not None and plot_path.exists():
        trend_html = _img(plot_path.read_bytes(), "Historical trends and daily GPU hours")
    else:
        trend_html = '<p class="muted">Historical trends plot not available.</p>'

    # --- cluster locations ---
    try:
        cluster_map_style = (
            f' style="--cluster-map-bg: url({_png_data_url(plot_cluster_location_map())});"'
        )
    except Exception:
        cluster_map_style = ""

    # --- heatmaps / distribution (TRAIL members) ---
    now_epoch = int(datetime.now().timestamp())
    try:
        heatmap_queue_html = _img(
            plot_split_queue_heatmap(
                rows,
                "Queue Time — External Clusters (↖ last observed, ↘ 14d median)",
                now=now_epoch,
                clusters=QUEUE_HEATMAP_CLUSTERS,
            ),
            "Combined queue time heatmap for external clusters",
        )
    except Exception:
        heatmap_queue_html = '<p class="muted">Combined heatmap unavailable.</p>'

    # --- pending heatmap (squeue, all cluster users) ---
    try:
        heatmap_pending_html = _img(
            plot_pending_heatmap(snapshots, clusters=PENDING_HEATMAP_CLUSTERS),
            "Pending jobs heatmap, all cluster users",
        )
    except Exception:
        heatmap_pending_html = '<p class="muted">Pending heatmap unavailable.</p>'

    # --- TRAIL per-host per-GPU heatmap ---
    try:
        trail_grid_html = _img(
            plot_trail_gpu_grid(snapshots),
            "Per-GPU memory and utilisation across TRAIL hosts",
        )
    except Exception:
        trail_grid_html = '<p class="muted">TRAIL GPU grid unavailable.</p>'

    # --- TRAIL utilization timeline (sampled per refresh; fills in over time) ---
    try:
        trail_timeline_html = _img(
            plot_trail_utilization_timeline(days=window_days),
            "TRAIL GPU utilisation over time",
        )
    except Exception:
        trail_timeline_html = '<p class="muted">TRAIL utilisation timeline unavailable.</p>'

    # Optional inline TRAIL logo. Looks for env override first, otherwise a
    # `trail_logo.{svg,png}` next to the output HTML. SVGs are inlined as
    # markup; PNGs are base64-encoded.
    logo_html = ""
    favicon_url = os.environ.get("QUEUE_TIME_FAVICON", TRAIL_FAVICON_URL)
    favicon_html = (
        f'<link rel="icon" type="image/png" href="{html.escape(favicon_url, quote=True)}">\n'
        f'<link rel="shortcut icon" type="image/png" href="{html.escape(favicon_url, quote=True)}">'
    )
    logo_env = os.environ.get("QUEUE_TIME_LOGO")
    candidates: list[Path] = []
    if logo_env:
        candidates.append(Path(logo_env))
    else:
        for ext in ("svg", "png"):
            candidates.append(html_path.parent / f"trail_logo.{ext}")
    for cand in candidates:
        try:
            if not cand.exists():
                continue
            if cand.suffix.lower() == ".svg":
                svg_text = cand.read_text(encoding="utf-8")
                # Strip any XML prolog so the SVG inlines cleanly inside <body>.
                svg_text = re.sub(r"^<\?xml[^?]*\?>\s*", "", svg_text)
                logo_html = f'<span class="logo">{svg_text}</span>'
            else:
                b64 = base64.b64encode(cand.read_bytes()).decode("ascii")
                logo_html = (
                    '<span class="logo">'
                    f'<img src="data:image/png;base64,{b64}" alt="TRAIL">'
                    '</span>'
                )
            break
        except Exception:
            continue

    update_epoch = int(datetime.now().timestamp())
    timestamp = datetime.fromtimestamp(update_epoch).strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    refresh = max(30, int(refresh_seconds))
    if refresh % 3600 == 0:
        refresh_label = f"{refresh // 3600}h"
    elif refresh % 60 == 0:
        refresh_label = f"{refresh // 60}m"
    else:
        refresh_label = f"{refresh}s"
    subtitle = (
        f'Auto-refresh every {refresh_label} - Last update '
        f'<span id="last-update-age" data-updated-epoch="{update_epoch}">just now</span>'
    )

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{refresh}">
<meta name="viewport" content="width=device-width, initial-scale=1">
{favicon_html}
<title>TRAIL Compute Status Dashboard</title>
<style>
  :root {{
    --bg: #fafafa;
    --card: #ffffff;
    --text: #1d1d1f;
    --muted: #6b7280;
    --border: #e5e7eb;
    --accent: #15803d;
    --warn: #b45309;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    max-width: 1280px;
    margin: 2rem auto;
    padding: 0 1.25rem 3rem;
    color: var(--text);
    background: var(--bg);
    line-height: 1.5;
  }}
  h1 {{ font-size: 1.6rem; margin: 0 0 0.25rem; font-weight: 600; }}
  .subtitle {{ color: var(--muted); margin: 0 0 1.5rem; font-size: 0.9rem; }}
  h2.section {{ font-size: 1rem; margin: 2rem 0 0.5rem; font-weight: 600; border-top: 1px solid var(--border); padding-top: 1rem; }}
  h2.section .muted {{ font-weight: 400; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
  }}
  th, td {{
    padding: 0.55rem 0.8rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  th {{
    font-weight: 600;
    background: #f3f4f6;
    color: var(--muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  tr:last-child td {{ border-bottom: none; }}
  td.cluster {{ font-weight: 600; text-transform: capitalize; }}
  td.cell.best {{ color: var(--accent); font-weight: 700; }}
  td.cell.empty {{ color: var(--muted); }}
  td.bucket {{ font-weight: 600; }}
  td.rec-empty {{ color: var(--muted); }}
  td.rec-cell {{ vertical-align: top; }}
  .rec-server {{ display: block; font-weight: 700; text-transform: capitalize; }}
  .rec-detail {{ display: block; font-size: 0.8rem; color: var(--muted); }}
  img {{
    max-width: 100%;
    height: auto;
    display: block;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--card);
    margin: 0.5rem 0 1rem;
  }}
  .ts {{ color: var(--muted); font-size: 0.85rem; margin-top: 1.25rem; }}
  .muted {{ color: var(--muted); }}
  .usage-alert {{
    margin: -0.15rem 0 0.65rem;
    color: #92400e;
    background: #fffbeb;
    border: 1px solid #f59e0b;
    border-left: 4px solid #b45309;
    border-radius: 6px;
    padding: 0.45rem 0.7rem;
    font-size: 0.85rem;
  }}
  .cohort-divider {{
    margin: 2.5rem 0 1.25rem;
    padding: 0.5rem 1rem;
    background: #1d1d1f;
    color: #ffffff;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }}
  .stale-banner {{
    background: #fef3c7;
    border: 1px solid #f59e0b;
    border-left: 4px solid #b45309;
    color: #78350f;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    margin: 0 0 1.25rem;
    font-size: 0.9rem;
  }}
  .stale-banner strong {{ color: #78350f; }}
  .stale-banner-sub {{ display: block; color: #92400e; font-weight: 400; margin-top: 0.15rem; }}
  .stale-banner ul {{ margin: 0.4rem 0 0; padding-left: 1.25rem; }}
  .stale-banner li {{ margin: 0.1rem 0; }}
  tr.stale td {{ background: #fffbeb; }}
  .stale-tag {{ color: #b45309; font-weight: 500; font-size: 0.82rem; }}
  .host-tag {{ color: #6b7280; font-size: 0.82rem; font-weight: 400; }}
  .header {{
    display: flex;
    align-items: stretch;
    justify-content: space-between;
    gap: 1rem;
    min-height: 260px;
    margin: 0 0 1.5rem;
    padding: 1.4rem 1.5rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background:
      radial-gradient(ellipse at bottom left, rgba(255,255,255,0.96) 0%, rgba(255,255,255,0.78) 33%, rgba(255,255,255,0.18) 66%, rgba(255,255,255,0) 100%),
      radial-gradient(ellipse at top right, rgba(255,255,255,0.92) 0%, rgba(255,255,255,0.64) 33%, rgba(255,255,255,0.16) 66%, rgba(255,255,255,0) 100%),
      var(--cluster-map-bg, linear-gradient(135deg, #ffffff, #f3f4f6));
    background-size: 58% 56%, 42% 42%, contain;
    background-repeat: no-repeat;
    background-position: bottom left, top right, center;
    overflow: hidden;
  }}
  .header h1 {{ margin: 0; }}
  .header .subtitle {{ margin: 0.15rem 0 0; }}
  .header-text {{ display: flex; flex-direction: column; align-self: flex-end; }}
  .logo {{
    flex: 0 0 auto;
    align-self: flex-start;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    height: 56px;
    width: auto;
    border: none;
    border-radius: 0;
    background: transparent;
    margin: 0;
  }}
  .logo svg, .logo img {{
    height: 56px; width: auto; max-width: 220px;
    border: none; border-radius: 0; background: transparent; margin: 0;
    display: block;
  }}
</style>
<script>
  function updateLastUpdateAge() {{
    const el = document.getElementById("last-update-age");
    if (!el) return;
    const updated = Number(el.dataset.updatedEpoch || 0);
    if (!updated) return;
    const elapsed = Math.max(0, Math.floor(Date.now() / 1000) - updated);
    let label;
    if (elapsed < 60) {{
      label = "just now";
    }} else if (elapsed < 3600) {{
      label = `${{Math.floor(elapsed / 60)}}m ago`;
    }} else if (elapsed < 86400) {{
      label = `${{(elapsed / 3600).toFixed(1)}}h ago`;
    }} else {{
      label = `${{(elapsed / 86400).toFixed(1)}}d ago`;
    }}
    el.textContent = label;
  }}
  window.addEventListener("DOMContentLoaded", () => {{
    updateLastUpdateAge();
    setInterval(updateLastUpdateAge, 30000);
  }});
</script>
</head>
<body>
  <div class="header"{cluster_map_style}>
    <div class="header-text">
      <h1>TRAIL Compute Status Dashboard</h1>
      <p class="subtitle">{subtitle}</p>
    </div>
    {logo_html}
  </div>
{stale_banner_html}

  <div class="cohort-divider">Member Usage</div>

  <h2 class="section">Member Usage Allocated Cluster <span class="muted">- {html.escape(allocated_member_usage_subtitle)}</span></h2>
  <table>
    <thead>
      <tr>
        <th>Member</th><th>{TRAIL_LOOKBACK_DAYS}D GPU HRS</th><th>{TRAIL_LOOKBACK_DAYS}D GPU Util</th><th>{TRAIL_LOOKBACK_DAYS}D Inactive GPU Hrs</th><th>{TRAIL_LOOKBACK_DAYS}D CLUSTERS</th>
      </tr>
    </thead>
    <tbody>
{allocated_member_usage_table_html}
    </tbody>
  </table>

  <h2 class="section">Member Usage Other Cluster <span class="muted">- {html.escape(other_member_usage_subtitle)}</span></h2>
  <table>
    <thead>
      <tr>
        <th>Member</th><th>{TRAIL_LOOKBACK_DAYS}D GPU HRS</th><th>{TRAIL_LOOKBACK_DAYS}D GPU Util</th><th>{TRAIL_LOOKBACK_DAYS}D Inactive GPU Hrs</th><th>{TRAIL_LOOKBACK_DAYS}D CLUSTERS</th>
      </tr>
    </thead>
    <tbody>
{other_member_usage_table_html}
    </tbody>
  </table>

  <h2 class="section">Member Usage UTIAS Server <span class="muted">- {html.escape(utias_member_usage_subtitle)}</span></h2>
{utias_history_alert_html}
  <table>
    <thead>
      <tr>
        <th>Member</th><th>{TRAIL_LOOKBACK_DAYS}D GPU HRS</th><th>{TRAIL_LOOKBACK_DAYS}D Servers</th><th>{TRAIL_LOOKBACK_DAYS}D DAYS USED</th>
      </tr>
    </thead>
    <tbody>
{utias_member_usage_table_html}
    </tbody>
  </table>

  <div class="cohort-divider">Cluster Usage</div>

  <h2 class="section">Allocated Cluster Usage</h2>
  <table>
    <thead>
      <tr>
        <th>Cluster</th><th>Node Type</th><th>GPU NODES</th><th>LEVELFS</th><th>{TRAIL_LOOKBACK_DAYS}D GPU HRS</th><th>{TRAIL_LOOKBACK_DAYS}D QUEUE TIME</th><th>{TRAIL_LOOKBACK_DAYS}D USERS</th>
      </tr>
    </thead>
    <tbody>
{drac_allocated_table_html}
    </tbody>
  </table>

  <h2 class="section">UTIAS Server Usage</h2>
{utias_history_alert_html}
  <table>
    <thead>
      <tr>
        <th>Cluster</th><th>Node Type</th><th>GPU UTIL</th><th>Users</th><th>{TRAIL_LOOKBACK_DAYS}D GPU UTIL</th><th>{TRAIL_LOOKBACK_DAYS}D USERS</th>
      </tr>
    </thead>
    <tbody>
{trail_table_html}
    </tbody>
  </table>

  <h2 class="section">Other Cluster Usage</h2>
  <table>
    <thead>
      <tr>
        <th>Cluster</th><th>Node Type</th><th>GPU NODES</th><th>LEVELFS</th><th>{TRAIL_LOOKBACK_DAYS}D GPU HRS</th><th>{TRAIL_LOOKBACK_DAYS}D QUEUE TIME</th><th>Location</th>
      </tr>
    </thead>
    <tbody>
{other_cluster_table_html}
    </tbody>
  </table>

  <div class="cohort-divider">UTIAS SERVERS historical data</div>

  <h2 class="section">GPU utilisation over time <span class="muted">· last {window_days} days, % of GPUs in use, sampled each refresh</span></h2>
  {trail_timeline_html}

  <div class="cohort-divider">External clusters historical data</div>

  <h2 class="section">Queue time, last observed vs 14d median <span class="muted">· external clusters · ↖ most recent job (or longest current pending) · ↘ median over last 14 days</span></h2>
  {heatmap_queue_html}

  <h2 class="section">Historical Trends - 14d, queue times only shown for top 5 ...</h2>
  {trend_html}

  <div class="cohort-divider">External clusters live data</div>
  {heatmap_pending_html}

  <p class="ts">Last updated {html.escape(timestamp)}</p>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(document, encoding="utf-8")


_BUCKET_PROBE_KEY: dict[str, str] = {
    "3h": "sbatch_opts_short",
    "12h": "sbatch_opts",
    "1d": "sbatch_opts_24h",
}
_PROBE_CLUSTER_ABBR: dict[str, str] = {
    "n": "narval",
    "t": "trillium",
    "k": "killarney",
    "kh": "killarney_h100",
    "v": "vulcan",
    "f": "fir",
    "nb": "nibi",
    "rq": "rorqual",
    "ta": "tamia",
}
_PROBE_SPEC_RE = re.compile(r"^(kh|rq|ta|nb|n|t|k|v|f)(?:(\d+)g)?(3h|12h|1d)$")


def _parse_probe_spec(spec: str) -> tuple[str, dict, str, str]:
    """Parse a probe spec string like 'n4g12h' -> (cluster, info, sbatch_opts, bucket)."""
    m = _PROBE_SPEC_RE.match(spec.lower())
    if not m:
        raise SystemExit(
            f"Invalid probe spec {spec!r}. "
            "Format: {n|t|k|kh}[{N}g]{3h|12h|1d}  e.g. n4g12h"
        )
    abbr, gpus_str, bucket = m.group(1), m.group(2), m.group(3)
    cluster = _PROBE_CLUSTER_ABBR[abbr]
    info = CLUSTER_INFO[cluster]
    base_key = _BUCKET_PROBE_KEY[bucket]
    gpu_key = f"{base_key}_{gpus_str}gpu" if gpus_str else None
    opts_key = gpu_key if gpu_key and info.get(gpu_key) else base_key
    opts = info.get(opts_key, "")
    if not opts:
        raise SystemExit(f"No sbatch opts configured for {cluster} {bucket}")
    return cluster, info, opts, bucket


def submit_probe(
    cluster: str, info: dict[str, str], sbatch_opts: str, timeout: int
) -> str:
    parts = [info["init"]]
    if info.get("probe_cwd"):
        parts.append(f"cd {shlex.quote(info['probe_cwd'])}")
    parts.append(
        f"sbatch {sbatch_opts} --job-name=queue_probe "
        "--output=/dev/null --wrap='true'"
    )
    remote_cmd = " && ".join(parts)
    result = subprocess.run(
        ["ssh", *SSH_OPTS, info["host"], remote_cmd],
        check=False, capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "<no output>"
        raise RuntimeError(f"submit failed (exit={result.returncode}): {msg}")
    for line in result.stdout.splitlines():
        if line.startswith("Submitted batch job"):
            return line.split()[-1]
    raise RuntimeError(f"no job ID in output: {result.stdout.strip()}")


def submit_all_probes(timeout: int, spec: str = "all") -> int:
    rocket = [
        r"        /\        ",
        r"       /  \       ",
        r"      /----\      ",
        r"      |    |      ",
        r"     /|____|\     ",
        r"    /_|____|_\    ",
        r"       /  \       ",
        r"        **        ",
        r"       ****       ",
    ]

    if spec == "all":
        probe_specs = [_parse_probe_spec(s) for s in ("n4g12h", "n1g1d", "t4g12h", "k4g12h")]
    else:
        probe_specs = [_parse_probe_spec(spec)]

    failures = 0
    probe_lines: list[str] = []
    with ThreadPoolExecutor(max_workers=len(probe_specs)) as pool:
        futures = {
            pool.submit(submit_probe, cluster, info, opts, timeout): (cluster, tag)
            for cluster, info, opts, tag in probe_specs
        }
        for future in as_completed(futures):
            cluster, tag = futures[future]
            try:
                job_id = future.result()
                probe_lines.append(
                    f"▲  [probe:{cluster:<9} {tag:<3}]  in flight  →  job {job_id}"
                )
            except Exception as exc:
                failures += 1
                probe_lines.append(
                    f"✗  [probe:{cluster:<9} {tag:<3}]  abort      →  {exc}"
                )

    n_total = len(probe_specs)
    if failures:
        closer = (
            f"··· {n_total - failures}/{n_total} probe(s) airborne, "
            f"{failures} aborted; telemetry next run ···"
        )
    else:
        closer = f"··· all {n_total} probes airborne; telemetry returns on next run ···"

    info_lines = [
        "",
        "·  *  ·  ✦   PROBE SEQUENCE START   ✦  ·  *  ·",
        "",
        *probe_lines,
        "",
        closer,
    ]

    rocket_w = len(rocket[0])
    pad = " " * rocket_w
    total = max(len(rocket), len(info_lines))
    for i in range(total):
        rline = rocket[i] if i < len(rocket) else pad
        iline = info_lines[i] if i < len(info_lines) else ""
        if iline:
            print(f"{rline}   {iline}")
        else:
            print(rline)
    print()
    return failures


def merge_staleness(
    stale_acct: dict[str, int | None],
    stale_snap: dict[str, int | None],
) -> dict[str, int | None]:
    """Combine accounting + snapshot staleness into one per-cluster map.

    A cluster appears in the result iff at least one source fell back to
    cache (or had no cache). The reported age is the older of the two
    cached sources we actually have; None means no usable cache existed
    on any failing side.
    """
    merged: dict[str, int | None] = {}
    clusters = set(stale_acct) | set(stale_snap)
    for cluster in clusters:
        ages = [
            x for x in (stale_acct.get(cluster), stale_snap.get(cluster))
            if isinstance(x, int)
        ]
        merged[cluster] = max(ages) if ages else None
    return merged


def run_report(args: argparse.Namespace) -> int:
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows_future = pool.submit(query_all_clusters, args)
        snapshots_future = pool.submit(query_all_snapshots, SNAPSHOT_TIMEOUT)
        rows, failures, stale_acct = rows_future.result()
        snapshots, stale_snap = snapshots_future.result()
    for failure in failures:
        print(failure)

    stale = {
        cluster: age
        for cluster, age in merge_staleness(stale_acct, stale_snap).items()
        if cluster in DASHBOARD_STALE_CLUSTERS
    }
    if stale:
        print()
        print("⚠ Stale cluster data (using cached values from a previous run):")
        for cluster in sorted(stale):
            label = _CLUSTER_ABBR.get(cluster, cluster)
            ip = CLUSTER_INFO.get(cluster, {}).get("ip", "")
            ip_str = f" [{ip}]" if ip else ""
            print(f"   {label}{ip_str}: {format_staleness(stale[cluster])}")

    print_banner()
    print()
    print("TRAIL members usage:")
    print_cluster_snapshot(snapshots, rows=rows, stale_acct=stale_acct)
    print()
    print("UTIAS servers:")
    print_trail_snapshot(snapshots)
    print()
    print_member_usage(
        "Member Usage Allocated Cluster:",
        rows,
        MEMBER_ALLOCATED_USAGE_CLUSTERS,
    )
    print()
    print_member_usage(
        "Member Usage Other Cluster:",
        rows,
        MEMBER_OTHER_USAGE_CLUSTERS,
    )
    print()
    print_utias_member_usage()
    for gpus in GPU_GROUPS:
        print_bucket_table(rows, snapshots, gpus)

    plot_time_limit_secs = parse_time_limit_seconds(args.time_limit)
    plot_bucket = bucket_for_time_limit(plot_time_limit_secs)
    plot_rows = [
        r
        for r in rows
        if int(r["gpus"]) == args.gpus
        and (
            plot_bucket is None
            or bucket_for_time_limit(int(r["time_limit_seconds"])) == plot_bucket
        )
    ]
    series = build_daily_series(plot_rows, args.window)
    if args.all:
        print_all(series, args.window)
    if not args.no_plot:
        plot_series(rows, args.plot, args.window)
        print(f"Plot written to {args.plot}")
    if args.html is not None and not args.no_html:
        plot_for_html = args.plot if not args.no_plot else None
        write_html(
            args.html, plot_for_html, rows, snapshots, args.refresh_seconds,
            window_days=args.window, stale=stale, stale_acct=stale_acct,
        )
        print(f"HTML written to {args.html}")
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    if args.window <= 0:
        raise SystemExit("--window must be positive")

    if args.probe is not None:
        return 1 if submit_all_probes(SUBMIT_TIMEOUT, spec=args.probe) else 0

    return run_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
