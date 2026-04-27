#!/usr/bin/env bash
# GPU queue snapshot, probe, and plot for Narval, Trillium, Killarney.
# Captures cluster-wide node/pending/LevelFS snapshot, submits a 4-GPU no-op
# probe job to each cluster to measure actual queue time, then refreshes the
# rolling queue-time plot via tools/plot_queue_time.py.

REPO="$(cd "$(dirname "$0")/.." && pwd)"
QUEUE_CSV="$REPO/reports/queue_time.csv"
QUEUE_PLOT="$REPO/reports/queue_time_plot.png"
SNAPSHOT_TIMEOUT=45
SUBMIT_TIMEOUT=15
QUERY_TIMEOUT=10
POLL_SECONDS=120
MAX_WAIT_SECONDS=28800
SSH_OPTS=(-o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o BatchMode=yes)

mkdir -p "$REPO/reports"
[[ -f "$QUEUE_CSV" ]] || echo "timestamp,record_type,cluster,idle,mix,alloc,down,pending,fairshare,job_id,submit_time,start_time,queue_seconds,status" > "$QUEUE_CSV"

echo "Probe results file: $QUEUE_CSV"

NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# ---------------------------------------------------------------------------
# Parallel snapshot collection
# ---------------------------------------------------------------------------

remote_snapshot() {
  local host="$1" init="$2" part="$3" account="$4"
  timeout "$SNAPSHOT_TIMEOUT" ssh "${SSH_OPTS[@]}" "$host" "bash -s -- $(printf '%q' "$init") $(printf '%q' "$part") $(printf '%q' "$account")" <<'REMOTE'
eval "$1"; part="$2"; account="$3"
STATES=$(sinfo -p "$part" --noheader -o '%n %t' 2>/dev/null | sort -k1,1 -u | awk '
  {st=$2; gsub(/[^a-z]/,"",st)
   if(st=="idle") i++; else if(st=="mix") m++; else if(st=="alloc") a++; else d++}
  END{printf "%d %d %d %d\n", i+0, m+0, a+0, d+0}')
PENDING=$(squeue -p "$part" -t PENDING --noheader 2>/dev/null | wc -l)
# sshare -l columns: Account|User|RawShares|NormShares|RawUsage|NormUsage|EffectvUsage|FairShare|LevelFS|...
# Group row has empty User; we want that row's LevelFS ($9).
# Empty account skips the lookup (e.g. Killarney does not expose GPU LevelFS).
if [[ -n "$account" ]]; then
  LFS=$(sshare -l -A "$account" --parsable2 --noheader 2>/dev/null | awk -F'|' '$2==""{print $9; exit}')
fi
echo "STATES $STATES"
echo "PENDING $PENDING"
echo "LEVELFS ${LFS:-N/A}"
REMOTE
}

T_N=$(mktemp); T_T=$(mktemp); T_K=$(mktemp)

remote_snapshot narval       "source ~/.bashrc"                                                         gpubase_bygpu_b1 rrg-swasland_gpu > "$T_N" 2>&1 &
remote_snapshot trillium_gpu "source ~/.bashrc"                                                         compute          rrg-swasland     > "$T_T" 2>&1 &
remote_snapshot killarney    "source /etc/profile.d/modules.sh && module load slurm/killarney/24.05.7" gpubase_l40s_b1  ""               > "$T_K" 2>&1 &

wait

# ---------------------------------------------------------------------------
# Parse helper
# ---------------------------------------------------------------------------

parse() {
  local f="$1" key="$2" col="$3"
  awk "/^${key}/{print \$$col}" "$f"
}

last_queue() {
  awk -F',' -v c="$1" '
    $2=="probe" && $3==c && $13 ~ /^[0-9]+$/ { last=$13 }
    END { if (last != "") printf "%d", last }' "$QUEUE_CSV"
}

N_IDLE=$(parse "$T_N" STATES 2); N_MIX=$(parse "$T_N" STATES 3)
N_ALLOC=$(parse "$T_N" STATES 4); N_DOWN=$(parse "$T_N" STATES 5)
N_PEND=$(parse "$T_N" PENDING 2); N_FS=$(parse "$T_N" LEVELFS 2)
N_LAST=$(last_queue narval)

T_IDLE=$(parse "$T_T" STATES 2); T_MIX=$(parse "$T_T" STATES 3)
T_ALLOC=$(parse "$T_T" STATES 4); T_DOWN=$(parse "$T_T" STATES 5)
T_PEND=$(parse "$T_T" PENDING 2); T_FS=$(parse "$T_T" LEVELFS 2)
T_LAST=$(last_queue trillium)

K_IDLE=$(parse "$T_K" STATES 2); K_MIX=$(parse "$T_K" STATES 3)
K_ALLOC=$(parse "$T_K" STATES 4); K_DOWN=$(parse "$T_K" STATES 5)
K_PEND=$(parse "$T_K" PENDING 2); K_FS=$(parse "$T_K" LEVELFS 2)
K_LAST=$(last_queue killarney)

for var in N_IDLE N_MIX N_ALLOC N_DOWN N_PEND N_FS N_LAST \
           T_IDLE T_MIX T_ALLOC T_DOWN T_PEND T_FS T_LAST \
           K_IDLE K_MIX K_ALLOC K_DOWN K_PEND K_FS K_LAST; do
  [[ -z "${!var}" ]] && printf -v "$var" "N/A"
done

# Killarney exposes no per-group LevelFS account; fairshare is uniform at 1.0.
[[ "$K_FS" == "N/A" ]] && K_FS="1.0"

rm -f "$T_N" "$T_T" "$T_K"

# ---------------------------------------------------------------------------
# Append snapshot CSV rows
# ---------------------------------------------------------------------------

echo "$NOW,snapshot,narval,$N_IDLE,$N_MIX,$N_ALLOC,$N_DOWN,$N_PEND,$N_FS,,,,,"     >> "$QUEUE_CSV"
echo "$NOW,snapshot,trillium,$T_IDLE,$T_MIX,$T_ALLOC,$T_DOWN,$T_PEND,$T_FS,,,,,"   >> "$QUEUE_CSV"
echo "$NOW,snapshot,killarney,$K_IDLE,$K_MIX,$K_ALLOC,$K_DOWN,$K_PEND,$K_FS,,,,,"  >> "$QUEUE_CSV"

# ---------------------------------------------------------------------------
# Print + verdict
# ---------------------------------------------------------------------------

print_row() {
  local cluster="$1" node_type="$2" idle="$3" mix="$4" alloc="$5" pend="$6" fs="$7" last="$8"
  local active ratio fs_fmt last_fmt
  if [[ "$idle" == "N/A" || "$mix" == "N/A" || "$alloc" == "N/A" ]]; then
    active="N/A"
  else
    active=$((idle + mix + alloc))
  fi
  if [[ "$active" == "N/A" || "$active" -eq 0 || "$pend" == "N/A" ]]; then
    ratio="N/A"
  else
    ratio=$(awk -v p="$pend" -v a="$active" 'BEGIN{printf "%.2f", p/a}')
  fi
  if [[ "$fs" == "N/A" ]]; then
    fs_fmt="N/A"
  else
    fs_fmt=$(awk -v f="$fs" 'BEGIN{printf "%.3f", f}')
  fi
  if [[ "$last" == "N/A" ]]; then
    last_fmt="N/A"
  else
    last_fmt="${last}s"
  fi
  printf '%-10s  %-12s  %7s  %10s  %9s  %8s  %10s\n' \
    "$cluster" "$node_type" "$active" "$pend" "$ratio" "$fs_fmt" "$last_fmt"
}

printf '\n'
printf '%-10s  %-12s  %7s  %10s  %9s  %8s  %10s\n' \
  "Cluster" "Node Type" "Nodes" "Jobs Queue" "Jobs/Node" "LevelFS" "Queue Time"
print_row narval    4xA100-80GB "$N_IDLE" "$N_MIX" "$N_ALLOC" "$N_PEND" "$N_FS" "$N_LAST"
print_row trillium  4xH100-80GB "$T_IDLE" "$T_MIX" "$T_ALLOC" "$T_PEND" "$T_FS" "$T_LAST"
print_row killarney 4xL40S-48GB "$K_IDLE" "$K_MIX" "$K_ALLOC" "$K_PEND" "$K_FS" "$K_LAST"
printf '\n'

verdict() {
  # Recommend the cluster with the fastest last successful probe.
  # First arg is the label, e.g. "Pre-probe recommendation".
  local label="${1:-Recommendation}"
  local best="" best_qs=999999 stats qs
  for cluster in narval trillium killarney; do
    stats=$(awk -F',' -v c="$cluster" '
      $2=="probe" && $3==c && $13 ~ /^[0-9]+$/ { last=$13 }
      END { if (last != "") printf "%d", last }' "$QUEUE_CSV")
    [[ -z "$stats" ]] && continue
    qs=$stats
    if [[ $qs -lt $best_qs ]]; then
      best=$cluster; best_qs=$qs
    fi
  done
  if [[ -z "$best" ]]; then
    echo "${label}: no probe history yet."
  else
    echo "${label}: submit to $best (last queue ${best_qs}s)"
  fi
}

verdict "Pre-probe recommendation"

# ---------------------------------------------------------------------------
# Probe jobs
# Each probe submits a 4-GPU job with the same scheduler resource request as
# the corresponding run wrapper, waits for SLURM accounting to report Submit
# and Start timestamps, then records actual queue time to queue_time.csv.
# ---------------------------------------------------------------------------

submit_probe() {
  local cluster="$1" host="$2" prefix="$3" sbatch_opts="$4"
  local submitted_at out job_id elapsed acct submit start state queue_seconds recorded_at

  submitted_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  out=$(timeout "$SUBMIT_TIMEOUT" ssh "${SSH_OPTS[@]}" "$host" \
    "$prefix sbatch $sbatch_opts --job-name=queue_probe --output=/dev/null --wrap='true'" 2>&1)
  rc=$?
  if [[ $rc -ne 0 ]]; then
    if [[ $rc -eq 124 ]]; then
      echo "[probe:$cluster] submit timed out after ${SUBMIT_TIMEOUT}s (ssh hung; partial output: ${out:-<none>})" >&2
    else
      echo "[probe:$cluster] submit failed (exit=$rc): ${out:-<no output>}" >&2
    fi
    return 1
  fi

  job_id=$(awk '/Submitted batch job/{print $NF}' <<< "$out")
  if [[ -z "$job_id" ]]; then
    echo "[probe:$cluster] no job ID: $out" >&2
    return 1
  fi

  echo "[probe:$cluster] submitted $job_id"
  elapsed=0
  while [[ $elapsed -lt $MAX_WAIT_SECONDS ]]; do
    sleep "$POLL_SECONDS"
    elapsed=$((elapsed + POLL_SECONDS))
    acct=$(timeout "$QUERY_TIMEOUT" ssh "${SSH_OPTS[@]}" "$host" \
      "$prefix sacct -j $job_id -X --noheader --parsable2 --format=Submit,Start,State 2>/dev/null | head -n 1" 2>/dev/null)
    IFS='|' read -r submit start state <<< "$acct"
    if [[ -n "$submit" && -n "$start" && "$start" != "Unknown" ]]; then
      queue_seconds=$(( $(date -d "$start" +%s) - $(date -d "$submit" +%s) ))
      recorded_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
      echo "$recorded_at,probe,$cluster,,,,,,,$job_id,$submit,$start,$queue_seconds,$state" >> "$QUEUE_CSV"
      echo "[probe:$cluster] $job_id queued for ${queue_seconds}s"
      return 0
    fi
  done

  recorded_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "$recorded_at,probe,$cluster,,,,,,,$job_id,$submitted_at,TIMEOUT,>$MAX_WAIT_SECONDS,timeout" >> "$QUEUE_CSV"
  timeout "$QUERY_TIMEOUT" ssh "${SSH_OPTS[@]}" "$host" "$prefix scancel $job_id" 2>/dev/null || true
}

submit_probe \
  narval narval \
  "source ~/.bashrc &&" \
  "--account=rrg-swasland --ntasks=1 --cpus-per-task=12 --mem=120gb --time=11:59:00 --gres=gpu:a100:4" &

submit_probe \
  trillium trillium_gpu \
  "source ~/.bashrc &&" \
  "--account=rrg-swasland --ntasks=1 --cpus-per-task=24 --time=11:59:00 --gpus-per-node=4" &

submit_probe \
  killarney killarney \
  "source /etc/profile.d/modules.sh && module load slurm/killarney/24.05.7 && cd /scratch &&" \
  "--account=aip-swasland --ntasks=1 --cpus-per-task=16 --mem=120gb --time=11:59:00 --gres=gpu:l40s:4" &

echo "Probe jobs submitted."
wait

echo
verdict "Post-probe recommendation"

# ---------------------------------------------------------------------------
# Refresh queue-time plot from live SLURM accounting
# ---------------------------------------------------------------------------

echo
echo "Queue times plot: $QUEUE_PLOT"
python3 "$REPO/tools/plot_queue_time.py"
