#!/usr/bin/env bash
set -euo pipefail
cd /home2/zhaoyx/scenario-dreamer
PY=/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python
ROOT=checkpoints/.scenario_dreamer_download_20260924
$PY -m gdown --folder --remaining-ok "https://drive.google.com/drive/folders/1kGvNQleNL_FAn956ngdx2Ga0iQDRkX_8?usp=sharing" -O "$ROOT/scenario_dreamer_autoencoder_waymo"
$PY -m gdown --folder --remaining-ok "https://drive.google.com/drive/folders/1xV35C5aEjbjbGsujzz134VvwgrZG4gWW?usp=sharing" -O "$ROOT/scenario_dreamer_ldm_large_waymo"
sha256sum "$ROOT/scenario_dreamer_autoencoder_waymo/last.ckpt" "$ROOT/scenario_dreamer_ldm_large_waymo/last.ckpt" > work/checkpoint_download_20260924/sha256.txt
