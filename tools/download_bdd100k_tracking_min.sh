#!/usr/bin/env bash
set -euo pipefail

# Minimal download helper for BDD100K tracking (val).
# You must obtain the correct URLs from the BDD100K portal.

RAW_DIR="data/raw/bdd100k_tracking/raw"
OUT_DIR="data/raw/bdd100k_tracking"

mkdir -p "${RAW_DIR}"

if [[ -z "${BDD100K_TRACKING_IMAGES_URL:-}" || -z "${BDD100K_TRACKING_LABELS_URL:-}" ]]; then
  cat <<'EOF'
Set download URLs before running this script:

  export BDD100K_TRACKING_IMAGES_URL="YOUR_IMAGES_ZIP_URL"
  export BDD100K_TRACKING_LABELS_URL="YOUR_LABELS_ZIP_URL"

You can obtain the URLs from:
  https://bdd-data.berkeley.edu/portal.html#download
EOF
  exit 1
fi

echo "[INFO] Downloading tracking images..."
aria2c -c -x 8 -s 8 -d "${RAW_DIR}" "${BDD100K_TRACKING_IMAGES_URL}"

echo "[INFO] Downloading tracking labels..."
aria2c -c -x 8 -s 8 -d "${RAW_DIR}" "${BDD100K_TRACKING_LABELS_URL}"

echo "[INFO] Extracting..."
unzip -q "${RAW_DIR}"/*.zip -d "${OUT_DIR}"

echo "[OK] Done. Verify data under ${OUT_DIR}"
