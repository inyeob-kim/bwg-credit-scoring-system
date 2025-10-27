#!/usr/bin/env bash
# Example cron job script placeholder for automated retraining.
# You'd replace the curl with your data source fetch.

set -euo pipefail
CSV_PATH=${1:-"data.csv"}
TARGET=${2:-"default_12m"}

curl -X POST "http://localhost:8000/train" \
  -F "file=@${CSV_PATH}" \
  -F "target=${TARGET}" \
  -F "test_size=0.2" \
  -F "random_state=42" \
  -F "feature_select_threshold=median"