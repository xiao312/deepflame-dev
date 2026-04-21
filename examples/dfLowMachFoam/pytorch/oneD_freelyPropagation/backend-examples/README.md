# H2 backend examples

This directory contains three self-contained runnable variants of the H2 freely
propagating flame case using the new inference backend design.

## Cases

- `pytorchEmbedded-H2/` — embedded Python backend using `inference.py`
- `onnxRuntime-H2/` — ONNX Runtime backend case
- `tensorRt-H2/` — TensorRT backend case

Each case directory contains its own `0/`, `constant/`, `system/`, and required
artifacts so the subdirectories remain runnable after this directory is copied or
pushed independently.

All three cases are configured to run to:

```text
endTime = 0.01
```

and are meant to work for a new user via `./Allrun` once DeepFlame is built with
the required backend dependencies.

## Preparing DeepFlame for backend inference

Run all build steps from the repository root.

### 1. Activate the build environment

```bash
source /opt/openfoam7/etc/bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepflame
```

### 2. Prepare ONNX Runtime dependencies

#### Recommended install approach

For GPU-backed ONNX Runtime runs, the recommended approach is to install
ONNX Runtime and the matching NVIDIA CUDA runtime Python packages into the same
conda environment used to build and run DeepFlame.

Example:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepflame

pip install onnx onnxruntime-gpu==1.19.2 \
  nvidia-cublas-cu12 \
  nvidia-cuda-runtime-cu12 \
  nvidia-cudnn-cu12 \
  nvidia-cufft-cu12 \
  nvidia-curand-cu12 \
  nvidia-cuda-nvrtc-cu12 \
  nvidia-nvjitlink-cu12
```

For CPU-only ONNX Runtime usage, install:

```bash
pip install onnx onnxruntime
```

#### How to verify ONNX Runtime installation

Check that Python can import ONNX Runtime and report available providers:

```bash
python - <<'PY'
import onnxruntime as ort
print('onnxruntime:', ort.__version__)
print('providers:', ort.get_available_providers())
print('package_dir:', ort.__path__[0])
PY
```

Expected outcomes:

- CPU-only install should include `CPUExecutionProvider`
- GPU install should typically include `CUDAExecutionProvider`

You can also verify the C/C++ runtime library directory that DeepFlame will use:

```bash
python - <<'PY'
import pathlib, onnxruntime
capi = pathlib.Path(onnxruntime.__file__).resolve().parent / 'capi'
print(capi)
print((capi / 'libonnxruntime.so').exists())
PY
```

#### Option A: explicit ONNX Runtime path (recommended for DeepFlame configure)

Pass the ONNX Runtime `capi/` directory explicitly to `configure.sh`:

```bash
source ./configure.sh \
  --use_pytorch \
  --onnxruntime_dir "$CONDA_PREFIX/lib/python3.8/site-packages/onnxruntime/capi"
```

You can also set an environment variable instead:

```bash
export ONNXRUNTIME_ROOT="$CONDA_PREFIX/lib/python3.8/site-packages/onnxruntime/capi"
source ./configure.sh --use_pytorch
```

#### Option B: Python-package fallback

If you do not pass an explicit path, `configure.sh` can autodetect ONNX Runtime
from the active Python environment:

```bash
source ./configure.sh --use_pytorch
```

After running `configure.sh`, verify what DeepFlame detected. The configure
summary should print something like:

- `ONNXRUNTIME_SOURCE=cli`
- `ONNXRUNTIME_DIR=.../site-packages/onnxruntime/capi`
- `ONNXRUNTIME_LIB=.../libonnxruntime.so...`

ONNX Runtime precedence:

1. `--onnxruntime_dir`
2. `ONNXRUNTIME_ROOT` / `ONNXRUNTIME_DIR`
3. Python-package autodetect fallback

### 3. Prepare TensorRT dependencies

TensorRT support is optional and requires an explicit install path.

#### How to choose TensorRT / CUDA / ONNX Runtime versions

Before installing TensorRT, determine:

1. **GPU availability and architecture**

```bash
nvidia-smi
```

This tells you whether an NVIDIA GPU is available and what GPU model is present.

2. **CUDA major version available in your runtime stack**

Useful checks include:

```bash
nvidia-smi
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
PY
```

3. **cuDNN / provider compatibility expectations**

For ONNX Runtime GPU and TensorRT, version compatibility matters. In practice,
you should choose versions that are mutually compatible with:

- the installed NVIDIA driver
- the CUDA major version you intend to use
- the cuDNN major version available in your Python/runtime stack
- the ONNX Runtime GPU package you plan to use
- the TensorRT release you plan to install

General guidance:

- choose an ONNX Runtime GPU build whose CUDA/cuDNN requirements match your
  environment
- choose a TensorRT release built for the same CUDA major family
- if PyTorch is already installed and working with CUDA on the machine, prefer
  ONNX Runtime and TensorRT versions compatible with that same CUDA generation
- if you are unsure, start by identifying the working PyTorch CUDA version and
  matching ONNX Runtime GPU and TensorRT to that stack

#### Recommended TensorRT install approach

For C++ DeepFlame integration, prefer the official TensorRT **tar package** over
`pip`-only installs, because DeepFlame needs:

- headers such as `NvInfer.h`
- runtime libraries such as `libnvinfer.so`
- `trtexec`

A typical install pattern is:

```bash
mkdir -p ~/opt
cd ~/opt
# unpack official TensorRT tarball here
# example result:
#   ~/opt/TensorRT-10.16.1.11/
```

#### How to verify TensorRT installation

Check headers, libraries, and `trtexec`:

```bash
test -f "$HOME/opt/TensorRT-10.16.1.11/include/NvInfer.h" && echo OK: NvInfer.h
test -e "$HOME/opt/TensorRT-10.16.1.11/lib/libnvinfer.so" -o -e "$HOME/opt/TensorRT-10.16.1.11/lib/libnvinfer.so.10" && echo OK: libnvinfer
"$HOME/opt/TensorRT-10.16.1.11/bin/trtexec" --help | head
```

You should also verify that the chosen TensorRT install matches the CUDA family
available on the machine and in your active runtime stack.

#### Option A: explicit TensorRT path (recommended)

```bash
source ./configure.sh \
  --use_pytorch \
  --onnxruntime_dir "$CONDA_PREFIX/lib/python3.8/site-packages/onnxruntime/capi" \
  --tensorrt_dir "$HOME/opt/TensorRT-10.16.1.11"
```

#### Option B: environment variable

```bash
export TENSORRT_ROOT="$HOME/opt/TensorRT-10.16.1.11"
source ./configure.sh \
  --use_pytorch \
  --onnxruntime_dir "$CONDA_PREFIX/lib/python3.8/site-packages/onnxruntime/capi"
```

After running `configure.sh`, verify that DeepFlame detected TensorRT correctly.
The configure summary should print something like:

- `TENSORRT_SOURCE=cli`
- `TENSORRT_DIR=...`
- `TENSORRT_LIBDIR=...`

TensorRT precedence:

1. `--tensorrt_dir`
2. `TENSORRT_ROOT` / `TENSORRT_DIR`
3. no autodetect fallback

### 4. Build DeepFlame

```bash
source ./bashrc
wmake libso src/dfChemistryModel
wmake libso src/dfCombustionModels
wmake applications/solvers/dfLowMachFoam
```

## Running the cases

### Embedded Python

```bash
cd pytorchEmbedded-H2
./Allrun
```

### ONNX Runtime

```bash
cd onnxRuntime-H2
./Allrun
```

`onnxRuntime-H2/Allrun` will automatically build
`h2_full_inference_wrapper.onnx` with `onnxRuntime-H2/build_h2_onnx_wrapper.py`
if the ONNX wrapper file is missing.

### TensorRT

```bash
cd tensorRt-H2
./Allrun
```

`tensorRt-H2/Allrun` will automatically build:

- `h2_full_inference_wrapper.onnx` with `tensorRt-H2/build_h2_onnx_wrapper.py`
- `h2_full_inference_wrapper_fp32.plan` with `tensorRt-H2/build_h2_tensorrt_engine.sh`

if the required files are missing.

## Notes

- `TensorRT` currently supports centralized inference only.
- The TensorRT engine is specific to the exported H2 wrapper graph and the
  TensorRT/CUDA/GPU environment used to build it.
- All example configs use `log false;` by default to keep runs quiet.
- `tensorRt-H2/build_h2_tensorrt_engine.sh` documents the exact `trtexec`
  command used for the H2 TensorRT engine.
