#!/usr/bin/env bash
# Report whether a branch has moved since a workflow's last successful scheduled run.
#
# Only a scheduled run compares; any other event always runs. A lookup that fails, or a first night
# with no green run before it, answers "moved", so the work runs rather than being skipped on a
# guess. Writes `unchanged=true|false` to $GITHUB_OUTPUT.
#
# Environment: EVENT_NAME, SHA, REPOSITORY, WORKFLOW, BRANCH, GITHUB_OUTPUT (and GH_TOKEN for gh).
set -euo pipefail

unchanged=false
if [ "$EVENT_NAME" = "schedule" ]; then
  last=$(gh run list --repo "$REPOSITORY" --workflow "$WORKFLOW" --branch "$BRANCH" \
    --event schedule --status success --limit 1 --json headSha --jq '.[0].headSha' \
    2>/dev/null || echo unknown)
  if [ "$last" = "$SHA" ]; then
    unchanged=true
  fi
  echo "last green scheduled run ${last:-none}, this run $SHA, unchanged: $unchanged"
else
  echo "a $EVENT_NAME run always runs"
fi
echo "unchanged=$unchanged" >> "$GITHUB_OUTPUT"
