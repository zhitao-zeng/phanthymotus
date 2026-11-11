#!/bin/sh
# Keep the CUDA 11.8 compatibility runtime private to the JP5 ASR worker.
set -eu
worker_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cuda_root="$worker_root/cuda-runtime/usr/local/cuda-11.8"
export LD_LIBRARY_PATH="$cuda_root/compat:$cuda_root/targets/aarch64-linux/lib:$worker_root/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$cuda_root/compat/libcuda.so.1 ${LD_PRELOAD:-/usr/lib/aarch64-linux-gnu/libgomp.so.1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
exec "$worker_root/bin/whisper_turbo_worker.native" "$@"
