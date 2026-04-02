---
name: sync-code
description: Push the current local git branch and pull it on one or all remote servers (DGX, Apollo, Narval).
---

Push the current local branch and pull it on a remote server.

## Arguments
Parse from: $ARGUMENTS
- `--server` (required): `dgx` | `apollo` | `narval` | `all`

## Steps

1. Run `/connect-server --server <server>` to ensure VPN and SSH are active. Stop if it fails.

2. Push local branch:
```bash
git push
```
If push fails (e.g. no upstream), set upstream:
```bash
git push -u origin $(git rev-parse --abbrev-ref HEAD)
```

3. Pull on each target server:
```bash
# DGX
ssh trail_dgx "cd /raid/home/spapais/ForeSight && git pull"

# Apollo
ssh apollo "cd /home/spapais/ForeSight && git pull"

# Narval
ssh narval "cd /home/spapais/ForeSight && git pull"
```

For `--server all`, run the pull on all three servers in sequence.

4. Report the branch name and which servers were updated. If a server's `git pull` outputs "Already up to date.", note it.

## Notes
- This skill syncs **code only** — for pulling result logs use `/sync-results`.
- Never rsync source files; git is the only code-sync mechanism for this project.
- If the remote is on a different branch, warn the user rather than switching branches automatically.
