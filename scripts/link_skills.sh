#!/usr/bin/env bash
# Re-run whenever skills are added, removed, or renamed.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ln -sf "$REPO/.claude/CLAUDE.md" "$REPO/AGENTS.md"

mkdir -p "$REPO/.agents/skills"
for skill_dir in "$REPO/.claude/skills"/*/; do
  skill=$(basename "$skill_dir")
  mkdir -p "$REPO/.agents/skills/$skill"
  for file in "$skill_dir"*; do
    [ -f "$file" ] && ln -sf "$file" "$REPO/.agents/skills/$skill/$(basename "$file")"
  done
done

echo "Done."
