#!/usr/bin/env bash
# Re-run whenever skills are added, removed, or renamed.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cp "$REPO/.claude/CLAUDE.md" "$REPO/AGENTS.md"

for dest in "$REPO/.agents/skills" "$HOME/.codex/skills" "$HOME/.agents/skills"; do
  mkdir -p "$dest"
  for skill_dir in "$REPO/.claude/skills"/*/; do
    skill=$(basename "$skill_dir")
    mkdir -p "$dest/$skill"
    for file in "$skill_dir"*; do
      [ -f "$file" ] && cp "$file" "$dest/$skill/$(basename "$file")"
    done
  done
done

echo "Done."
