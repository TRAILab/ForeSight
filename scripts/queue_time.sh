#!/usr/bin/env bash
# GPU queue snapshot and probe for Narval, Trillium, Killarney.
# Submits a 4-GPU no-op probe job to each cluster and records actual queue time.

REPO="$(cd "$(dirname "$0")/.." && pwd)"
QUEUE_CSV="$REPO/reports/queue_time.csv"
SNAPSHOT_TIMEOUT=45
SUBMIT_TIMEOUT=15
QUERY_TIMEOUT=10
POLL_SECONDS=120
MAX_WAIT_SECONDS=28800

mkdir -p "$REPO/reports"
[[ -f "$QUEUE_CSV" ]] || echo "timestamp,record_type,cluster,idle,mix,alloc,down,pending,fairshare,job_id,submit_time,start_time,queue_seconds,status" > "$QUEUE_CSV"

NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# ---------------------------------------------------------------------------
# Parallel snapshot collection
# ---------------------------------------------------------------------------

remote_snapshot() {
  local host="$1" init="$2" part="$3" account="$4"
  timeout "$SNAPSHOT_TIMEOUT" ssh "$host" "bash -s -- $(printf '%q' "$init") $(printf '%q' "$part") $(printf '%q' "$account")" <<'REMOTE'
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

N_IDLE=$(parse "$T_N" STATES 2); N_MIX=$(parse "$T_N" STATES 3)
N_ALLOC=$(parse "$T_N" STATES 4); N_DOWN=$(parse "$T_N" STATES 5)
N_PEND=$(parse "$T_N" PENDING 2); N_FS=$(parse "$T_N" LEVELFS 2)

T_IDLE=$(parse "$T_T" STATES 2); T_MIX=$(parse "$T_T" STATES 3)
T_ALLOC=$(parse "$T_T" STATES 4); T_DOWN=$(parse "$T_T" STATES 5)
T_PEND=$(parse "$T_T" PENDING 2); T_FS=$(parse "$T_T" LEVELFS 2)

K_IDLE=$(parse "$T_K" STATES 2); K_MIX=$(parse "$T_K" STATES 3)
K_ALLOC=$(parse "$T_K" STATES 4); K_DOWN=$(parse "$T_K" STATES 5)
K_PEND=$(parse "$T_K" PENDING 2); K_FS=$(parse "$T_K" LEVELFS 2)

for var in N_IDLE N_MIX N_ALLOC N_DOWN N_PEND N_FS \
           T_IDLE T_MIX T_ALLOC T_DOWN T_PEND T_FS \
           K_IDLE K_MIX K_ALLOC K_DOWN K_PEND K_FS; do
  [[ -z "${!var}" ]] && printf -v "$var" "N/A"
done

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

printf '\n'
printf '**Narval** (A100 80GB, 4 GPU/node)\n'
printf '  Nodes — idle: %s  mix: %s  alloc: %s  down: %s\n' "$N_IDLE" "$N_MIX" "$N_ALLOC" "$N_DOWN"
printf '  Pending GPU jobs: %s\n' "$N_PEND"
printf '  Group LevelFS: %s\n' "$N_FS"
printf '\n'
printf '**Trillium** (H100 80GB, 4 GPU/node)\n'
printf '  Nodes — idle: %s  mix: %s  alloc: %s  down: %s\n' "$T_IDLE" "$T_MIX" "$T_ALLOC" "$T_DOWN"
printf '  Pending GPU jobs: %s\n' "$T_PEND"
printf '  Group LevelFS: %s\n' "$T_FS"
printf '\n'
printf '**Killarney** (L40S 48GB, 4 GPU/node)\n'
printf '  Nodes — idle: %s  mix: %s  alloc: %s  down: %s\n' "$K_IDLE" "$K_MIX" "$K_ALLOC" "$K_DOWN"
printf '  Pending GPU jobs: %s\n' "$K_PEND"
printf '  Group LevelFS: %s\n' "$K_FS"
printf '\n'

verdict() {
  local best="" best_pend=999999 best_reason="" pend fs idle mix best_fs=0
  for cluster in narval trillium killarney; do
    case $cluster in
      narval)    pend=$N_PEND; fs=$N_FS; idle=$N_IDLE; mix=$N_MIX ;;
      trillium)  pend=$T_PEND; fs=$T_FS; idle=$T_IDLE; mix=$T_MIX ;;
      killarney) pend=$K_PEND; fs=$K_FS; idle=$K_IDLE; mix=$K_MIX ;;
    esac
    [[ "$pend" == "N/A" ]] && continue
    if [[ $pend -lt $best_pend ]] || \
       [[ $pend -eq $best_pend && "$fs" != "N/A" && $(awk "BEGIN{print ($fs > $best_fs)}") -eq 1 ]]; then
      best=$cluster; best_pend=$pend; best_fs=$fs
      if [[ "$idle" -gt 0 ]] 2>/dev/null; then
        best_reason="$idle idle nodes → near-instant start"
      elif [[ "$mix" -gt 0 ]] 2>/dev/null; then
        best_reason="$pend pending jobs, $mix mix nodes draining"
      else
        best_reason="fewest pending jobs ($pend) with LevelFS $fs"
      fi
    fi
  done
  echo "Submit to **$best**: $best_reason."
}

verdict

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
  out=$(timeout "$SUBMIT_TIMEOUT" ssh "$host" \
    "$prefix sbatch $sbatch_opts --job-name=queue_probe --output=/dev/null --wrap='true'" 2>&1) || {
      echo "[probe:$cluster] submit failed: $out" >&2
      return 1
    }

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
    acct=$(timeout "$QUERY_TIMEOUT" ssh "$host" \
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
  timeout "$QUERY_TIMEOUT" ssh "$host" "$prefix scancel $job_id" 2>/dev/null || true
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

echo "Probe jobs submitted. Waiting for SLURM accounting results; results -> $QUEUE_CSV"
wait
