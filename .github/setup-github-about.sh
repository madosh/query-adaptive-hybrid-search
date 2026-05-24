#!/usr/bin/env bash
# Apply GitHub About description + topics (requires GitHub CLI: gh auth login)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESCRIPTION="$(tr -d '\n' < "$SCRIPT_DIR/DESCRIPTION.txt" | sed 's/  */ /g')"
TOPICS="$(grep -v '^[[:space:]]*$' "$SCRIPT_DIR/TOPICS.txt" | paste -sd, -)"

echo "Setting repository description..."
gh repo edit --description "$DESCRIPTION"

echo "Adding repository topics..."
gh repo edit --add-topic "$TOPICS"

echo "Done. Verify in GitHub -> Settings -> General -> About."
