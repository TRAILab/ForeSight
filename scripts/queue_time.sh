#!/usr/bin/env bash
# GPU queue snapshot for Narval, Trillium, Killarney.
# Usage: queue_time.sh [--probe]
#   --probe  submit a 4-GPU no-op job to each cluster and record actual queue time

REPO="$(cd "$(dirname "$0")/.." && pwd)"
METRICS="$REPO/logs/queue_metrics"
SNAP_CSV="$METRICS/snapshots.csv"
PROBE_CSV="$METRICS/probes.csv"
PROBE=false
[[ "${1:-}" == "--probe" ]] && PROBE=true

mkdir -p "$METRICS"
[[ -f "$SNAP_CSV" ]] || echo "timestamp,cluster,idle,mix,alloc,down,pending,fairshare" > "$SNAP_CSV"
[[ -f "$PROBE_CSV" ]] || echo "submit_time,cluster,job_id,start_time,queue_seconds" > "$PROBE_CSV"

NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# ---------------------------------------------------------------------------
# Parallel snapshot collection
# ---------------------------------------------------------------------------

T_N=$(mktemp); T_T=$(mktemp); T_K=$(mktemp)

(timeout 45 ssh narval bash -s << 'EOF'
source ~/.bashrc
STATES=$(sinfo -p gpubase_bygpu_b1 --noheader -o '%n %t' 2>/dev/null | sort -k1,1 -u | awk '
  {st=$2; gsub(/[^a-z]/,"",st)
   if(st=="idle") i++; else if(st=="mix") m++; else if(st=="alloc") a++; else d++}
  END{printf "%d %d %d %d\n", i+0, m+0, a+0, d+0}')
PENDING=$(squeue -p gpubase_bygpu_b1 --noheader -o '%T' 2>/dev/null | { grep -c PENDING || echo 0; })
FS=$(sshare -u spapais --noheader 2>/dev/null | grep spapais | awk '{print $NF}' | sort -n | tail -1 || echo "N/A")
echo "STATES $STATES"
echo "PENDING $PENDING"
echo "FAIRSHARE ${FS:-N/A}"
EOF
) > "$T_N" 2>&1 &

(timeout 45 ssh trillium_gpu bash -s << 'EOF'
source ~/.bashrc
STATES=$(sinfo -p compute --noheader -o '%n %t' 2>/dev/null | sort -k1,1 -u | awk '
  {st=$2; gsub(/[^a-z]/,"",st)
   if(st=="idle") i++; else if(st=="mix") m++; else if(st=="alloc") a++; else d++}
  END{printf "%d %d %d %d\n", i+0, m+0, a+0, d+0}')
PENDING=$(squeue -p compute --noheader -o '%T' 2>/dev/null | { grep -c PENDING || echo 0; })
FS=$(sshare -u spapais --noheader 2>/dev/null | grep spapais | awk '{print $NF}' | sort -n | tail -1 || echo "N/A")
echo "STATES $STATES"
echo "PENDING $PENDING"
echo "FAIRSHARE ${FS:-N/A}"
EOF
) > "$T_T" 2>&1 &

(timeout 45 ssh killarney bash -s << 'EOF'
source /etc/profile.d/modules.sh
module load slurm/killarney/24.05.7
STATES=$(sinfo -p gpubase_l40s_b1 --noheader -o '%n %t' 2>/dev/null | sort -k1,1 -u | awk '
  {st=$2; gsub(/[^a-z]/,"",st)
   if(st=="idle") i++; else if(st=="mix") m++; else if(st=="alloc") a++; else d++}
  END{printf "%d %d %d %d\n", i+0, m+0, a+0, d+0}')
PENDING=$(squeue -p gpubase_l40s_b1 --noheader -o '%T' 2>/dev/null | { grep -c PENDING || echo 0; })
FS=$(sshare -u spapais --noheader 2>/dev/null | grep spapais | awk '{print $NF}' | sort -n | tail -1 || echo "N/A")
echo "STATES $STATES"
echo "PENDING $PENDING"
echo "FAIRSHARE ${FS:-N/A}"
EOF
) > "$T_K" 2>&1 &

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
N_PEND=$(parse "$T_N" PENDING 2); N_FS=$(parse "$T_N" FAIRSHARE 2)

T_IDLE=$(parse "$T_T" STATES 2); T_MIX=$(parse "$T_T" STATES 3)
T_ALLOC=$(parse "$T_T" STATES 4); T_DOWN=$(parse "$T_T" STATES 5)
T_PEND=$(parse "$T_T" PENDING 2); T_FS=$(parse "$T_T" FAIRSHARE 2)

K_IDLE=$(parse "$T_K" STATES 2); K_MIX=$(parse "$T_K" STATES 3)
K_ALLOC=$(parse "$T_K" STATES 4); K_DOWN=$(parse "$T_K" STATES 5)
K_PEND=$(parse "$T_K" PENDING 2); K_FS=$(parse "$T_K" FAIRSHARE 2)

# Default N/A when SSH failed
for var in N_IDLE N_MIX N_ALLOC N_DOWN N_PEND N_FS \
           T_IDLE T_MIX T_ALLOC T_DOWN T_PEND T_FS \
           K_IDLE K_MIX K_ALLOC K_DOWN K_PEND K_FS; do
  [[ -z "${!var}" ]] && printf -v "$var" "N/A"
done

rm -f "$T_N" "$T_T" "$T_K"

# ---------------------------------------------------------------------------
# Append snapshot CSV
# ---------------------------------------------------------------------------

echo "$NOW,narval,$N_IDLE,$N_MIX,$N_ALLOC,$N_DOWN,$N_PEND,$N_FS"     >> "$SNAP_CSV"
echo "$NOW,trillium,$T_IDLE,$T_MIX,$T_ALLOC,$T_DOWN,$T_PEND,$T_FS"   >> "$SNAP_CSV"
echo "$NOW,killarney,$K_IDLE,$K_MIX,$K_ALLOC,$K_DOWN,$K_PEND,$K_FS"  >> "$SNAP_CSV"

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

verdict() {
  local best="" best_pend=999999 best_reason=""
  for cluster in narval trillium killarney; do
    case $cluster in
      narval)    pend=$N_PEND; fs=$N_FS; idle=$N_IDLE; mix=$N_MIX ;;
      trillium)  pend=$T_PEND; fs=$T_FS; idle=$T_IDLE; mix=$T_MIX ;;
      killarney) pend=$K_PEND; fs=$K_FS; idle=$K_IDLE; mix=$K_MIX ;;
    esac
    [[ "$pend" == "N/A" ]] && continue
    if [[ $pend -lt $best_pend ]] || \
       [[ $pend -eq $best_pend && "$fs" != "N/A" && $(awk "BEGIN{print ($fs > ${best_fs:-0})}") -eq 1 ]]; then
      best=$cluster; best_pend=$pend; best_fs=$fs
      if [[ "$idle" -gt 0 ]] 2>/dev/null; then
        best_reason="$idle idle nodes → near-instant start"
      elif [[ "$mix" -gt 0 ]] 2>/dev/null; then
        best_reason="$pend pending jobs, $mix mix nodes draining"
      else
        best_reason="fewest pending jobs ($pend) with fairshare $fs"
      fi
    fi
  done
  echo "Submit to **$best**: $best_reason."
}

# ---------------------------------------------------------------------------
# Print output
# ---------------------------------------------------------------------------

printf '\n'
printf '**Narval** (A100 80GB, 4 GPU/node)\n'
printf '  Nodes — idle: %s  mix: %s  alloc: %s  down: %s\n' "$N_IDLE" "$N_MIX" "$N_ALLOC" "$N_DOWN"
printf '  Pending GPU jobs: %s\n' "$N_PEND"
printf '  Your fairshare: %s\n' "$N_FS"
printf '\n'
printf '**Trillium** (H100 80GB, 4 GPU/node)\n'
printf '  Nodes — idle: %s  mix: %s  alloc: %s  down: %s\n' "$T_IDLE" "$T_MIX" "$T_ALLOC" "$T_DOWN"
printf '  Pending GPU jobs: %s\n' "$T_PEND"
printf '  Your fairshare: %s\n' "$T_FS"
printf '\n'
printf '**Killarney** (L40S 48GB, 4 GPU/node)\n'
printf '  Nodes — idle: %s  mix: %s  alloc: %s  down: %s\n' "$K_IDLE" "$K_MIX" "$K_ALLOC" "$K_DOWN"
printf '  Pending GPU jobs: %s\n' "$K_PEND"
printf '  Your fairshare: %s\n' "$K_FS"
printf '\n'

verdict

# ---------------------------------------------------------------------------
# Probe jobs (--probe only)
# Each probe submits a 4-GPU job with the same scheduler resource request as
# the corresponding run wrapper, polls until it starts, records queue time to
# probes.csv, then cancels the job.
# ---------------------------------------------------------------------------

if $PROBE; then

  # Narval
  (
    ST=$(date -u +%Y-%m-%dT%H:%M:%SZ); SE=$(date +%s)
    OUT=$(timeout 15 ssh narval \
      "source ~/.bashrc && sbatch \
        --account=rrg-swasland --ntasks=1 --cpus-per-task=12 --mem=120gb \
        --time=11:59:00 --gres=gpu:a100:4 \
        --job-name=queue_probe --output=/dev/null \
        --wrap='sleep 120'" 2>&1) || { echo "[probe:narval] submit failed: $OUT" >&2; exit 1; }
    JID=$(awk '/Submitted batch job/{print $NF}' <<< "$OUT")
    [[ -z "$JID" ]] && { echo "[probe:narval] no job ID: $OUT" >&2; exit 1; }
    echo "[probe:narval] submitted $JID"
    elapsed=0
    while [[ $elapsed -lt 14400 ]]; do
      sleep 60; elapsed=$((elapsed + 60))
      STATE=$(timeout 10 ssh narval \
        "source ~/.bashrc && squeue -j $JID --noheader -o '%T' 2>/dev/null" 2>/dev/null | tr -d ' ')
      if [[ "$STATE" == "RUNNING" || "$STATE" == "COMPLETING" || -z "$STATE" ]]; then
        QS=$(( $(date +%s) - SE ))
        echo "$ST,narval,$JID,$(date -u +%Y-%m-%dT%H:%M:%SZ),$QS" >> "$PROBE_CSV"
        echo "[probe:narval] $JID started: ${QS}s queue time"
        timeout 10 ssh narval "source ~/.bashrc && scancel $JID" 2>/dev/null || true
        exit 0
      fi
    done
    echo "$ST,narval,$JID,TIMEOUT,>14400" >> "$PROBE_CSV"
    timeout 10 ssh narval "source ~/.bashrc && scancel $JID" 2>/dev/null || true
  ) &

  # Trillium
  (
    ST=$(date -u +%Y-%m-%dT%H:%M:%SZ); SE=$(date +%s)
    OUT=$(timeout 15 ssh trillium_gpu \
      "source ~/.bashrc && sbatch \
        --account=rrg-swasland --ntasks=1 --cpus-per-task=24 \
        --time=11:59:00 --gpus-per-node=4 \
        --job-name=queue_probe --output=/dev/null \
        --wrap='sleep 120'" 2>&1) || { echo "[probe:trillium] submit failed: $OUT" >&2; exit 1; }
    JID=$(awk '/Submitted batch job/{print $NF}' <<< "$OUT")
    [[ -z "$JID" ]] && { echo "[probe:trillium] no job ID: $OUT" >&2; exit 1; }
    echo "[probe:trillium] submitted $JID"
    elapsed=0
    while [[ $elapsed -lt 14400 ]]; do
      sleep 60; elapsed=$((elapsed + 60))
      STATE=$(timeout 10 ssh trillium_gpu \
        "source ~/.bashrc && squeue -j $JID --noheader -o '%T' 2>/dev/null" 2>/dev/null | tr -d ' ')
      if [[ "$STATE" == "RUNNING" || "$STATE" == "COMPLETING" || -z "$STATE" ]]; then
        QS=$(( $(date +%s) - SE ))
        echo "$ST,trillium,$JID,$(date -u +%Y-%m-%dT%H:%M:%SZ),$QS" >> "$PROBE_CSV"
        echo "[probe:trillium] $JID started: ${QS}s queue time"
        timeout 10 ssh trillium_gpu "source ~/.bashrc && scancel $JID" 2>/dev/null || true
        exit 0
      fi
    done
    echo "$ST,trillium,$JID,TIMEOUT,>14400" >> "$PROBE_CSV"
    timeout 10 ssh trillium_gpu "source ~/.bashrc && scancel $JID" 2>/dev/null || true
  ) &

  # Killarney
  (
    ST=$(date -u +%Y-%m-%dT%H:%M:%SZ); SE=$(date +%s)
    OUT=$(timeout 15 ssh killarney \
      "source /etc/profile.d/modules.sh && module load slurm/killarney/24.05.7 && sbatch \
        --account=aip-swasland --ntasks=1 --cpus-per-task=16 --mem=120gb \
        --time=11:59:00 --gres=gpu:l40s:4 \
        --job-name=queue_probe --output=/dev/null \
        --wrap='sleep 120'" 2>&1) || { echo "[probe:killarney] submit failed: $OUT" >&2; exit 1; }
    JID=$(awk '/Submitted batch job/{print $NF}' <<< "$OUT")
    [[ -z "$JID" ]] && { echo "[probe:killarney] no job ID: $OUT" >&2; exit 1; }
    echo "[probe:killarney] submitted $JID"
    elapsed=0
    while [[ $elapsed -lt 14400 ]]; do
      sleep 60; elapsed=$((elapsed + 60))
      STATE=$(timeout 10 ssh killarney \
        "source /etc/profile.d/modules.sh && module load slurm/killarney/24.05.7 && \
         squeue -j $JID --noheader -o '%T' 2>/dev/null" 2>/dev/null | tr -d ' ')
      if [[ "$STATE" == "RUNNING" || "$STATE" == "COMPLETING" || -z "$STATE" ]]; then
        QS=$(( $(date +%s) - SE ))
        echo "$ST,killarney,$JID,$(date -u +%Y-%m-%dT%H:%M:%SZ),$QS" >> "$PROBE_CSV"
        echo "[probe:killarney] $JID started: ${QS}s queue time"
        timeout 10 ssh killarney \
          "source /etc/profile.d/modules.sh && module load slurm/killarney/24.05.7 && scancel $JID" 2>/dev/null || true
        exit 0
      fi
    done
    echo "$ST,killarney,$JID,TIMEOUT,>14400" >> "$PROBE_CSV"
    timeout 10 ssh killarney \
      "source /etc/profile.d/modules.sh && module load slurm/killarney/24.05.7 && scancel $JID" 2>/dev/null || true
  ) &

  echo "Probe jobs submitted. Pollers running in background; results → $PROBE_CSV"
fi
