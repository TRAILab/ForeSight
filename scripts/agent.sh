#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREES_DIR="$(dirname "$REPO_ROOT")/ForeSight-worktrees"
BASE_BRANCH="sd"

usage() {
    echo "Usage: $0 <spawn|kill> <slug>"
    echo "  spawn feat_a  — create worktree on branch sd_feat_a + tmux session feat_a"
    echo "  kill  feat_a  — remove worktree and kill tmux session feat_a"
    exit 1
}

[[ $# -ne 2 ]] && usage

CMD="$1"
SLUG="$2"
BRANCH="sd_${SLUG}"
SESSION="$SLUG"
WORKTREE_PATH="${WORKTREES_DIR}/${SLUG}"

case "$CMD" in
    spawn)
        # Create worktrees directory if needed
        mkdir -p "$WORKTREES_DIR"

        if git -C "$REPO_ROOT" worktree list | grep -q "$WORKTREE_PATH"; then
            echo "Worktree already exists at $WORKTREE_PATH"
        else
            echo "Creating worktree at $WORKTREE_PATH on branch $BRANCH (from $BASE_BRANCH)..."
            git -C "$REPO_ROOT" worktree add -b "$BRANCH" "$WORKTREE_PATH" "$BASE_BRANCH"
            echo "Worktree created."
        fi

        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "tmux session '$SESSION' already exists."
        else
            echo "Starting tmux session '$SESSION'..."
            tmux new-session -d -s "$SESSION" -c "$WORKTREE_PATH"
            echo "tmux session '$SESSION' started."
        fi

        echo "Done. Attach with: tmux attach -t $SESSION"
        ;;

    kill)
        # Kill tmux session
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "Killing tmux session '$SESSION'..."
            tmux kill-session -t "$SESSION"
        else
            echo "No tmux session named '$SESSION' found."
        fi

        # Remove git worktree
        if git -C "$REPO_ROOT" worktree list | grep -q "$WORKTREE_PATH"; then
            echo "Removing worktree at $WORKTREE_PATH..."
            git -C "$REPO_ROOT" worktree remove --force "$WORKTREE_PATH"
            echo "Worktree removed."
        else
            echo "No worktree found at $WORKTREE_PATH."
        fi
        ;;

    *)
        usage
        ;;
esac
