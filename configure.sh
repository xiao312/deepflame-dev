#!/bin/sh

unset USE_LIBTORCH
unset USE_PYTORCH
unset LIBTORCH_DIR
unset LIBCANTERA_DIR
unset PYTORCH_INC
unset PYTORCH_LIB
unset USE_GPUSOLVER
unset AMGX_DIR
unset ODE_GPU_SOLVER
unset ONNXRUNTIME_DIR
unset ONNXRUNTIME_LIB
unset ONNXRUNTIME_SOURCE
unset TENSORRT_DIR
unset TENSORRT_LIBDIR
unset TENSORRT_SOURCE
unset CUDA_RUNTIME_INCLUDE
unset CUDA_NVCC_INCLUDE
unset CUDA_CCCL_INCLUDE
unset CUDA_RUNTIME_LIBDIR
unset NVIDIA_CUDA_LIB_DIRS

print_usage() {
    cat <<'EOF'
Usage: source ./configure.sh [options]

Core options:
  --libtorch_no                     Disable libtorch support (default)
  --libtorch_dir <path>             Use an explicit libtorch install
  --libtorch_autodownload           Download libtorch to ./thirdParty/libtorch
  --use_pytorch                     Enable embedded-Python / backend-based inference
  --libcantera_dir <path>           Use an explicit Cantera install (otherwise CONDA_PREFIX)
  --onnxruntime_dir <path>          Use an explicit ONNX Runtime library directory
  --tensorrt_dir <path>             Use an explicit TensorRT install root
  --amgx_dir <path>                 Enable AMGX support
  --use_ode_gpu_solver              Enable ODE GPU solver support
  -h, --help                        Show this message

Dependency precedence:
  ONNX Runtime : CLI flag --onnxruntime_dir
                 > env ONNXRUNTIME_ROOT / ONNXRUNTIME_DIR
                 > Python package autodetect (when --use_pytorch is enabled)
  TensorRT     : CLI flag --tensorrt_dir
                 > env TENSORRT_ROOT / TENSORRT_DIR
                 > no autodetect fallback
EOF
}

require_arg() {
    opt_name=$1
    opt_value=$2
    if [ -z "$opt_value" ]; then
        echo "ERROR: $opt_name requires a path argument."
        print_usage
        return 1
    fi
    return 0
}

python_path_probe() {
    python3 - "$1" <<'PY'
import pathlib
import site
import sys

suffix = sys.argv[1]
paths = []
try:
    paths.extend(site.getsitepackages())
except Exception:
    pass
try:
    user_site = site.getusersitepackages()
    if user_site:
        paths.append(user_site)
except Exception:
    pass
paths.append(str(pathlib.Path(sys.prefix) / 'lib'))

seen = set()
for root in paths:
    root_path = pathlib.Path(root)
    if not root_path.exists():
        continue
    for candidate in root_path.rglob(suffix):
        value = str(candidate.resolve())
        if value not in seen:
            seen.add(value)
            print(value)
            raise SystemExit(0)
print('')
PY
}

resolve_onnxruntime_from_dir() {
    candidate=$1
    if [ -z "$candidate" ]; then
        return 0
    fi
    if [ ! -d "$candidate" ]; then
        echo "ERROR: ONNX Runtime directory does not exist: $candidate"
        return 1
    fi

    lib_path=
    for pattern in "$candidate"/libonnxruntime.so.*; do
        case "$pattern" in
            *'.so.'*)
                if [ -e "$pattern" ]; then
                    lib_path=$pattern
                fi
                ;;
        esac
    done
    if [ -z "$lib_path" ] && [ -e "$candidate/libonnxruntime.so" ]; then
        lib_path="$candidate/libonnxruntime.so"
    fi

    if [ -z "$lib_path" ]; then
        echo "ERROR: Could not find libonnxruntime.so in $candidate"
        echo "       Pass --onnxruntime_dir <onnxruntime-capi-dir> or set ONNXRUNTIME_ROOT/ONNXRUNTIME_DIR."
        return 1
    fi

    ONNXRUNTIME_DIR=$candidate
    ONNXRUNTIME_LIB=$lib_path

    rm -f "$ONNXRUNTIME_DIR/libonnxruntime.so" "$ONNXRUNTIME_DIR/libonnxruntime.so.1"
    base_name=`basename "$ONNXRUNTIME_LIB"`
    ln -sf "$base_name" "$ONNXRUNTIME_DIR/libonnxruntime.so"
    ln -sf "$base_name" "$ONNXRUNTIME_DIR/libonnxruntime.so.1"
    return 0
}

resolve_tensorrt_from_root() {
    candidate=$1
    if [ -z "$candidate" ]; then
        return 0
    fi
    if [ ! -d "$candidate" ]; then
        echo "ERROR: TensorRT directory does not exist: $candidate"
        return 1
    fi
    if [ ! -f "$candidate/include/NvInfer.h" ]; then
        echo "ERROR: TensorRT headers not found under $candidate/include"
        echo "       Expected: $candidate/include/NvInfer.h"
        return 1
    fi

    libdir=
    if [ -d "$candidate/lib64" ]; then
        libdir=$candidate/lib64
    elif [ -d "$candidate/lib" ]; then
        libdir=$candidate/lib
    fi

    if [ -z "$libdir" ]; then
        echo "ERROR: TensorRT libraries not found under $candidate/lib or $candidate/lib64"
        return 1
    fi

    found_lib=
    for pattern in "$libdir"/libnvinfer.so.* "$libdir"/libnvinfer.so; do
        if [ -e "$pattern" ]; then
            found_lib=$pattern
            break
        fi
    done

    if [ -z "$found_lib" ]; then
        echo "ERROR: libnvinfer.so was not found in $libdir"
        return 1
    fi

    TENSORRT_DIR=$candidate
    TENSORRT_LIBDIR=$libdir
    return 0
}

# defaults
LIBTORCH_AUTO=false
USE_LIBTORCH=false
USE_PYTORCH=false
USE_GPUSOLVER=false
USE_ODE_GPU_SOLVER=false
CLI_ONNXRUNTIME_DIR=
CLI_TENSORRT_DIR=

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            print_usage
            return
            ;;
        --libtorch_dir)
            shift
            require_arg "--libtorch_dir" "$1" || return
            LIBTORCH_DIR=$1
            USE_LIBTORCH=true
            ;;
        --libtorch_autodownload)
            USE_LIBTORCH=true
            LIBTORCH_AUTO=true
            LIBTORCH_DIR="$PWD/thirdParty/libtorch"
            ;;
        --libtorch_no)
            USE_LIBTORCH=false
            LIBTORCH_DIR=
            ;;
        --use_pytorch)
            USE_PYTORCH=true
            ;;
        --libcantera_dir)
            shift
            require_arg "--libcantera_dir" "$1" || return
            LIBCANTERA_DIR=$1
            ;;
        --onnxruntime_dir)
            shift
            require_arg "--onnxruntime_dir" "$1" || return
            CLI_ONNXRUNTIME_DIR=$1
            ;;
        --tensorrt_dir)
            shift
            require_arg "--tensorrt_dir" "$1" || return
            CLI_TENSORRT_DIR=$1
            ;;
        --amgx_dir)
            shift
            require_arg "--amgx_dir" "$1" || return
            AMGX_DIR=$1
            USE_GPUSOLVER=true
            ;;
        --use_ode_gpu_solver)
            USE_ODE_GPU_SOLVER=true
            ;;
        *)
            echo "ERROR: $1 is not a recognized flag."
            print_usage
            return
            ;;
    esac
    shift
done

if [ -z "$LIBCANTERA_DIR" ] && [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: either provide --libcantera_dir or activate a conda environment containing libcantera-devel."
    return
fi

if [ -n "$LIBCANTERA_DIR" ] && [ -n "$CONDA_PREFIX" ]; then
    echo "duplicate libcantera dir from args and from conda!"
    echo "from args: $LIBCANTERA_DIR"
    echo "from conda: $CONDA_PREFIX"
    echo "using --libcantera_dir value: $LIBCANTERA_DIR"
fi

if [ -z "$LIBCANTERA_DIR" ] && [ -n "$CONDA_PREFIX" ]; then
    LIBCANTERA_DIR=$CONDA_PREFIX
fi

if [ "$USE_LIBTORCH" = true ] && [ "$USE_PYTORCH" = true ]; then
    echo "ERROR: choose either libtorch or pytorch, not both."
    return
fi

if [ "$LIBTORCH_AUTO" = true ]; then
    if [ -d "thirdParty/libtorch" ]; then
        echo "libtorch already exists."
    else
        if [ ! -e libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip ]; then
            wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip || return
        fi
        unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip -d thirdParty || return
    fi
fi

if [ "$USE_PYTORCH" = true ]; then
    PYTORCH_INC=`python3 -m pybind11 --includes 2>/dev/null`
    if [ -z "$PYTORCH_INC" ]; then
        echo "ERROR: failed to query pybind11 include flags. Ensure pybind11 is installed in the active Python environment."
        return
    fi

    PYTORCH_LIB=`pkg-config --libs python3-embed 2>/dev/null`
    if [ -z "$PYTORCH_LIB" ]; then
        echo "ERROR: failed to query python3-embed linker flags via pkg-config."
        return
    fi

    if [ -n "$CLI_ONNXRUNTIME_DIR" ]; then
        ONNXRUNTIME_SOURCE="cli"
        resolve_onnxruntime_from_dir "$CLI_ONNXRUNTIME_DIR" || return
    elif [ -n "$ONNXRUNTIME_ROOT" ]; then
        ONNXRUNTIME_SOURCE="env:ONNXRUNTIME_ROOT"
        resolve_onnxruntime_from_dir "$ONNXRUNTIME_ROOT" || return
    elif [ -n "$ONNXRUNTIME_DIR" ]; then
        ONNXRUNTIME_SOURCE="env:ONNXRUNTIME_DIR"
        resolve_onnxruntime_from_dir "$ONNXRUNTIME_DIR" || return
    else
        autodetect_ort_dir=`python3 - <<'PY'
import pathlib
try:
    import onnxruntime
    print((pathlib.Path(onnxruntime.__file__).resolve().parent / 'capi'))
except Exception:
    print('')
PY`
        if [ -n "$autodetect_ort_dir" ]; then
            ONNXRUNTIME_SOURCE="python-package-fallback"
            resolve_onnxruntime_from_dir "$autodetect_ort_dir" || return
        else
            ONNXRUNTIME_SOURCE="not-found"
        fi
    fi

    if [ -n "$CLI_TENSORRT_DIR" ]; then
        TENSORRT_SOURCE="cli"
        resolve_tensorrt_from_root "$CLI_TENSORRT_DIR" || return
    elif [ -n "$TENSORRT_ROOT" ]; then
        TENSORRT_SOURCE="env:TENSORRT_ROOT"
        resolve_tensorrt_from_root "$TENSORRT_ROOT" || return
    elif [ -n "$TENSORRT_DIR" ]; then
        TENSORRT_SOURCE="env:TENSORRT_DIR"
        resolve_tensorrt_from_root "$TENSORRT_DIR" || return
    else
        TENSORRT_SOURCE="not-found"
        TENSORRT_DIR=
        TENSORRT_LIBDIR=
    fi

    CUDA_RUNTIME_INCLUDE=`python_path_probe 'nvidia/cuda_runtime/include'`
    CUDA_NVCC_INCLUDE=`python_path_probe 'nvidia/cuda_nvcc/include'`
    CUDA_CCCL_INCLUDE=`python_path_probe 'nvidia/cuda_cccl/include'`
    CUDA_RUNTIME_LIBDIR=`python_path_probe 'nvidia/cuda_runtime/lib'`
    NVIDIA_CUDA_LIB_DIRS=`python3 - <<'PY'
import pathlib
import site
import sys
roots = []
try:
    roots.extend(site.getsitepackages())
except Exception:
    pass
try:
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(user_site)
except Exception:
    pass
wanted = ['cuda_runtime/lib','cublas/lib','cudnn/lib','cufft/lib','curand/lib','cuda_nvrtc/lib','nvjitlink/lib']
seen = []
for root in roots:
    base = pathlib.Path(root) / 'nvidia'
    if not base.exists():
        continue
    for rel in wanted:
        p = (base / rel)
        if p.exists():
            value = str(p.resolve())
            if value not in seen:
                seen.append(value)
print(':'.join(seen))
PY`
fi

echo "setup for deepflame bashrc:"
echo "LIBCANTERA_DIR=$LIBCANTERA_DIR"
if [ "$USE_LIBTORCH" = true ]; then
    echo "LIBTORCH_DIR=$LIBTORCH_DIR"
    echo 'PYTORCH_INC='
    echo 'PYTORCH_LIB='
fi
if [ "$USE_PYTORCH" = true ]; then
    echo "PYTORCH_INC=$PYTORCH_INC"
    echo "PYTORCH_LIB=$PYTORCH_LIB"
    echo "ONNXRUNTIME_SOURCE=$ONNXRUNTIME_SOURCE"
    echo "ONNXRUNTIME_DIR=$ONNXRUNTIME_DIR"
    echo "ONNXRUNTIME_LIB=$ONNXRUNTIME_LIB"
    echo "TENSORRT_SOURCE=$TENSORRT_SOURCE"
    echo "TENSORRT_DIR=$TENSORRT_DIR"
    echo "TENSORRT_LIBDIR=$TENSORRT_LIBDIR"
    echo "CUDA_RUNTIME_INCLUDE=$CUDA_RUNTIME_INCLUDE"
    echo "CUDA_NVCC_INCLUDE=$CUDA_NVCC_INCLUDE"
    echo "CUDA_CCCL_INCLUDE=$CUDA_CCCL_INCLUDE"
    echo "CUDA_RUNTIME_LIBDIR=$CUDA_RUNTIME_LIBDIR"
    echo "NVIDIA_CUDA_LIB_DIRS=$NVIDIA_CUDA_LIB_DIRS"
    echo 'LIBTORCH_DIR='
fi
if [ "$USE_GPUSOLVER" = true ]; then
    echo "AMGX_DIR=$AMGX_DIR"
fi
if [ "$USE_ODE_GPU_SOLVER" = true ]; then
    echo "ODE_GPU_SOLVER=$OPENCC_PATH"
fi

cp bashrc.in bashrc
sed -i "s#pwd#$PWD#g" ./bashrc
sed -i "s#LIBTORCH_DIR#$LIBTORCH_DIR#g" ./bashrc
sed -i "s#PYTORCH_INC#$PYTORCH_INC#g" ./bashrc
sed -i "s#PYTORCH_LIB#$PYTORCH_LIB#g" ./bashrc
sed -i "s#LIBCANTERA_DIR#$LIBCANTERA_DIR#g" ./bashrc
sed -i "s#@ONNXRUNTIME_DIR@#$ONNXRUNTIME_DIR#g" ./bashrc
sed -i "s#@ONNXRUNTIME_LIB@#$ONNXRUNTIME_LIB#g" ./bashrc
sed -i "s#@ONNXRUNTIME_SOURCE@#$ONNXRUNTIME_SOURCE#g" ./bashrc
sed -i "s#@TENSORRT_DIR@#$TENSORRT_DIR#g" ./bashrc
sed -i "s#@TENSORRT_LIBDIR@#$TENSORRT_LIBDIR#g" ./bashrc
sed -i "s#@TENSORRT_SOURCE@#$TENSORRT_SOURCE#g" ./bashrc
sed -i "s#@CUDA_RUNTIME_INCLUDE@#$CUDA_RUNTIME_INCLUDE#g" ./bashrc
sed -i "s#@CUDA_NVCC_INCLUDE@#$CUDA_NVCC_INCLUDE#g" ./bashrc
sed -i "s#@CUDA_CCCL_INCLUDE@#$CUDA_CCCL_INCLUDE#g" ./bashrc
sed -i "s#@CUDA_RUNTIME_LIBDIR@#$CUDA_RUNTIME_LIBDIR#g" ./bashrc
sed -i "s#@NVIDIA_CUDA_LIB_DIRS@#$NVIDIA_CUDA_LIB_DIRS#g" ./bashrc
sed -i "s#@AMGX_DIR@#$AMGX_DIR#g" ./bashrc
sed -i "s#@ODE_GPU_SOLVER@#$OPENCC_PATH#g" ./bashrc

if [ -d "src_orig" ]; then
    echo "src_orig exist."
else
    mkdir -p src_orig/TurbulenceModels
    mkdir -p src_orig/thermophysicalModels
    mkdir -p src_orig/lagrangian
    mkdir -p src_orig/regionModels
    mkdir -p src_orig/functionObjects
    cp -r $FOAM_SRC/TurbulenceModels/compressible src_orig/TurbulenceModels
    cp -r $FOAM_SRC/thermophysicalModels/basic src_orig/thermophysicalModels
    cp -r $FOAM_SRC/thermophysicalModels/thermophysicalProperties src_orig/thermophysicalModels
    cp -r $FOAM_SRC/lagrangian/intermediate src_orig/lagrangian
    cp -r $FOAM_SRC/lagrangian/turbulence src_orig/lagrangian
    cp -r $FOAM_SRC/regionModels/surfaceFilmModels src_orig/regionModels
    cp -r $FOAM_SRC/functionObjects/field src_orig/functionObjects
fi
