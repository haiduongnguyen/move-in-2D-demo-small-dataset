#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DATA="${PROJECT_ROOT}/project_data"

export TORCH_HOME="${PROJECT_DATA}/models/torch"
export HF_HOME="${PROJECT_DATA}/models/huggingface"
export XDG_CACHE_HOME="${PROJECT_DATA}/models/cache"

mkdir -p \
  "${TORCH_HOME}" \
  "${HF_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${PROJECT_DATA}/models/opencv" \
  "${PROJECT_DATA}/models/move_in_2d_aux"

echo "PROJECT_DATA=${PROJECT_DATA}"
echo "TORCH_HOME=${TORCH_HOME}"
echo "HF_HOME=${HF_HOME}"
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"

