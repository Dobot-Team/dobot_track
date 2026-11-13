#!/usr/bin/env bash
# Build the TrackVLA LLM norm + dynamic-quantization TensorRT plugin library.
#
# The library is self-contained (TensorRT + CUDA runtime only), so it builds in a few
# seconds.  Build it on Orin-local storage, not on the SSHFS mount.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
source_file="$script_dir/plugins/trackvla_llm_norm_quant_plugin.cu"
nvcc_bin=/usr/local/cuda/bin/nvcc
trt_root=/usr/src/tensorrt
if [[ -v NVCC ]]; then nvcc_bin="$NVCC"; fi
if [[ -v TENSORRT_ROOT ]]; then trt_root="$TENSORRT_ROOT"; fi
# Allow building straight from a scratch directory instead of the repo layout.
if [[ -v PLUGIN_SOURCE ]]; then source_file="$PLUGIN_SOURCE"; fi
output_dir="${OUTPUT_DIR:-$script_dir/build}"

test -f "$source_file"
test -f "$trt_root/include/NvInfer.h"
mkdir -p "$output_dir"

"$nvcc_bin" -O3 -std=c++17 -arch=sm_87 -shared -Xcompiler -fPIC \
  -I"$trt_root/include" "$source_file" \
  -L"$trt_root/lib" -lnvinfer -lcudart \
  -Xlinker -rpath -Xlinker "$trt_root/lib" \
  -o "$output_dir/libtrackvla_llm_norm_quant_plugin.so"

sha256sum "$output_dir/libtrackvla_llm_norm_quant_plugin.so"
