#!/usr/bin/env bash
# On-demand code push: local repo -> cluster, via rsync (size+mtime compare, does
# NOT read file bytes, so it's seconds even on the slow Google Drive mount and puts
# no continuous load on the CephFS login node). One-way by design: you edit code
# locally and run it on the cluster; outputs stay remote and come back via
# ./pull_results.sh. No --delete, so it can never touch remote logs/outputs.
#
# Usage:  ./sync_now.sh
set -euo pipefail

REMOTE_HOST="login-gpu.ece.local.cmu.edu"
REMOTE_DIR="research/3DImage/lidar/K-radar/K-Radar/"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/"   # repo root = parent of mutagen/

rsync -az --stats \
  --exclude='.git/' \
  --exclude='.DS_Store' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.ipynb_checkpoints/' \
  --exclude='/logs/' \
  --exclude='/docs/' \
  --exclude='/resources/' \
  --exclude='/tools/revise_label/' \
  --exclude='/build/' \
  --exclude='/pretrained/' \
  --exclude='/spconv_debug/' \
  --exclude='*.so' --exclude='*.o' --exclude='*.egg-info' \
  --exclude='*.gif' --exclude='*.mp4' --exclude='*.mov' --exclude='*.avi' \
  --exclude='*.ply' --exclude='*.png' --exclude='*.jpg' --exclude='*.jpeg' \
  --exclude='*.svg' --exclude='*.tif' --exclude='*.tiff' \
  --exclude='*.pth' --exclude='*.pt' --exclude='*.ckpt' --exclude='*.pkl' \
  --exclude='*.onnx' --exclude='*.engine' \
  --exclude='*.npy' --exclude='*.npz' --exclude='*.bin' \
  --exclude='*.h5' --exclude='*.hdf5' --exclude='*.pcd' --exclude='*.mat' \
  --exclude='*.zip' --exclude='*.tar' --exclude='*.tar.gz' --exclude='*.tgz' \
  "$LOCAL_DIR" "${REMOTE_HOST}:${REMOTE_DIR}"

echo "* pushed local code -> ${REMOTE_HOST}:${REMOTE_DIR}"
