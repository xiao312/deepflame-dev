#!/bin/sh
set -u

case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
engine_path="$case_dir/h2_full_inference_wrapper_fp32.plan"
onnx_path="$case_dir/h2_full_inference_wrapper.onnx"

: "${TENSORRT_ROOT:?Please source DeepFlame bashrc or export TENSORRT_ROOT before running this script.}"

if [ ! -x "$TENSORRT_ROOT/bin/trtexec" ]; then
    echo "ERROR: trtexec not found under $TENSORRT_ROOT/bin"
    exit 1
fi

if [ ! -f "$onnx_path" ]; then
    echo "ERROR: ONNX wrapper not found: $onnx_path"
    exit 1
fi

cmd="$TENSORRT_ROOT/bin/trtexec \
  --onnx=$onnx_path \
  --minShapes=cell_inputs:1x12 \
  --optShapes=cell_inputs:880x12 \
  --maxShapes=cell_inputs:2048x12 \
  --saveEngine=$engine_path \
  --skipInference"

echo "Building TensorRT engine with command:"
echo "$cmd"

eval "$cmd"
