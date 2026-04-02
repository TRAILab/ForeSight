---
name: connect-server
description: Ensure VPN is active and SSH connectivity works for a remote server. Use before any remote operation on DGX, Apollo, or Narval.
---

Ensure VPN is active and SSH connectivity works for a remote server.

## Arguments
Parse from: $ARGUMENTS
- `--server` (required): `dgx` | `apollo` | `narval` | `all`

## Behavior per server

### DGX
VPN: `nmcli` connection `utias-robotics` — should always stay on, never disconnect it.

1. Check if active:
```bash
nmcli con show --active | grep -q utias-robotics && echo "up" || echo "down"
```
2. If down, bring it up:
```bash
sudo nmcli con up id utias-robotics
```
3. Verify SSH:
```bash
ssh trail_dgx "echo ok"
```

### Apollo
VPN: UW `openconnect` — should always stay on, never disconnect it.

1. Check if active (test SSH directly):
```bash
ssh -o ConnectTimeout=5 apollo "echo ok" 2>/dev/null && echo "up" || echo "down"
```
2. If down: openconnect is interactive (prompts for password/2FA) — **do not run it automatically**. Instead, tell the user:
   > "Apollo VPN is down. Please run in a terminal: `sudo openconnect -v vpn.uwaterloo.ca -u s2papais`"
   > "Then re-run `/connect-server --server apollo` to verify."
   Stop here and wait for the user.
3. If up, confirm: "Apollo SSH OK."

### Narval
No VPN needed.

1. Verify SSH:
```bash
ssh -o ConnectTimeout=10 narval "echo ok"
```
2. Report success or failure.

### `--server all`
Run checks for DGX, Apollo, and Narval in sequence. Report a status summary:
```
DGX:    ✓ connected
Apollo: ✗ VPN down — action required (see above)
Narval: ✓ connected
```

## Notes
- VPNs are persistent — never add a disconnect step after SSH or rsync operations.
- If DGX nmcli fails (e.g. permission error), suggest: `sudo nmcli con up id utias-robotics`
- This skill is called automatically by `/sync-results` and `/submit-job` before any remote operation.
