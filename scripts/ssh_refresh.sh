#!/usr/bin/env bash
# Establish/refresh SSH master sockets for MFA-gated clusters.
# Run once at the start of the day; each prompt is one MFA per host.
# Subsequent ssh/scp to these hosts reuse the master and skip MFA
# until the master dies (laptop sleep, network change, long idle).
set -u

hosts=(
  trail_turing trail_lovelace trail_dgx
  narval trillium trillium_gpu killarney rorqual fir tamia nibi vulcan
)

for h in "${hosts[@]}"; do
  printf '\n=== %s ===\n' "$h"
  if ssh -o ConnectTimeout=15 "$h" true; then
    echo "  [OK] master live"
  else
    echo "  [FAILED]"
  fi
done

echo
echo "Done. Live sockets:"
ls ~/.ssh/cm-* 2>/dev/null || echo "  (none)"
