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
unset TENSORRT_DIR
unset TENSORRT_LIBDIR
unset CUDA_RUNTIME_INCLUDE
unset CUDA_NVCC_INCLUDE
unset CUDA_CCCL_INCLUDE
unset CUDA_RUNTIME_LIBDIR
unset NVIDIA_CUDA_LIB_DIRS

print_usage() {
    echo "Usage: . install.sh --libtorch_no (default) | --libtorch_dir _path_to_libtorch | --libtorch_autodownload | --use_pytorch | --libcantera_dir _path_to_libcantera
        | --tensorrt_dir _path_to_tensorrt | --amgx_dir _path_to_amgx | --use_ode_gpu_solver"
}

# default
LIBTORCH_AUTO=false
USE_LIBTORCH=false
USE_PYTORCH=false
USE_GPUSOLVER=false
USE_ODE_GPU_SOLVER=false

while test $# -gt 0; do
    case "$1" in
        -h|--help)
            print_usage
            return
            ;;
        --libtorch_dir)
            shift
            if test $# -gt 0; then
                LIBTORCH_DIR=$1
                USE_LIBTORCH=true
            else
                print_usage
            return
            fi
            shift
            ;;
        --libtorch_autodownload)
            USE_LIBTORCH=true
            LIBTORCH_AUTO=true
            LIBTORCH_DIR="$PWD/thirdParty/libtorch"
            shift
            ;;
        --libtorch_no)
            shift
            USE_LIBTORCH=false
            shift
            ;;
        --use_pytorch)
            shift
            USE_PYTORCH=true
            shift
            ;;
        --libcantera_dir)
            shift
            if test $# -gt 0; then
                LIBCANTERA_DIR=$1
            else
                print_usage
            return
            fi
            shift
            ;;
        --tensorrt_dir)
            shift
            if test $# -gt 0; then
                TENSORRT_DIR=$1
            else
                print_usage
            return
            fi
            shift
            ;;
        --amgx_dir)
            shift
            if test $# -gt 0; then
                AMGX_DIR=$1
                USE_GPUSOLVER=true
            else
                print_usage
            return
            fi
            shift
            ;;
        --use_ode_gpu_solver)
            shift
            USE_ODE_GPU_SOLVER=true
            ;;
        -h|--help)
            shift
            print_usage
            shift
            ;;
        *)
            echo "$1 is not a recognized flag!"
            print_usage
            return
            ;;
    esac
done



# if LIBCANTERA_DIR empty and CONDA_PREFIX empty
if [ -z "$LIBCANTERA_DIR" ] && [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR! either offer libcantera dir or ensure in the conda enviorment including libcantera-devel!"
    return
fi


# if LIBCANTERA_DIR not empty and CONDA_PREFIX not empty
# --libcantera_dir _path_to_libcantera has a higher priority than the path from conda enviornment
if [ ! -z "$LIBCANTERA_DIR" ] && [ ! -z "$CONDA_PREFIX" ]; then
    echo "duplicate libcantera dir from args and from conda!"
    echo "from args: "$LIBCANTERA_DIR
    echo "from args: "$CONDA_PREFIX
    echo "we are going to use the dir from args: "$LIBCANTERA_DIR
fi


# if LIBCANTERA_DIR empty and CONDA_PREFIX not empty
if [ -z "$LIBCANTERA_DIR" ] && [ ! -z "$CONDA_PREFIX" ]; then
    LIBCANTERA_DIR=$CONDA_PREFIX
fi


# if LIBCANTERA_DIR not empty and CONDA_PREFIX empty
# just use LIBCANTERA_DIR


if [ $USE_LIBTORCH = true ] && [ $USE_PYTORCH = true ]; then
    echo "ERROR! either use libtorch or pytorch!"
    return
fi


if [ $LIBTORCH_AUTO = true ]; then
    if [ -d "thirdParty/libtorch" ]; then
        echo "libtorch already exist."
    else
        if [ -e libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip ]
        then
            echo "libtorch.zip exist."
        else
            wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip
        fi
        unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip -d thirdParty
    fi
fi


if [ -z "$TENSORRT_DIR" ] && [ ! -z "$TENSORRT_ROOT" ]; then
    TENSORRT_DIR=$TENSORRT_ROOT
fi

if [ ! -z "$TENSORRT_DIR" ]; then
    if [ -d "$TENSORRT_DIR/lib64" ]; then
        TENSORRT_LIBDIR=$TENSORRT_DIR/lib64
    elif [ -d "$TENSORRT_DIR/lib" ]; then
        TENSORRT_LIBDIR=$TENSORRT_DIR/lib
    fi
fi

if [ $USE_PYTORCH = true ]; then
    PYTORCH_INC=`python3 -m pybind11 --includes`
    if [ -z "$PYTORCH_INC" ]; then
        return
    fi
    PYTORCH_LIB=`pkg-config --libs python3-embed`
    ONNXRUNTIME_DIR=`python3 - <<'PY'
import pathlib
try:
    import onnxruntime
    print(pathlib.Path(onnxruntime.__file__).resolve().parent / 'capi')
except Exception:
    print('')
PY`
    if [ ! -z "$ONNXRUNTIME_DIR" ]; then
        ONNXRUNTIME_LIB=`python3 - <<'PY'
import pathlib
try:
    import onnxruntime
    capi = pathlib.Path(onnxruntime.__file__).resolve().parent / 'capi'
    libs = sorted(capi.glob('libonnxruntime.so.*'))
    print(libs[-1] if libs else '')
except Exception:
    print('')
PY`
        if [ ! -z "$ONNXRUNTIME_LIB" ]; then
            rm -f "$ONNXRUNTIME_DIR/libonnxruntime.so"
            rm -f "$ONNXRUNTIME_DIR/libonnxruntime.so.1"
            ln -sf `basename "$ONNXRUNTIME_LIB"` "$ONNXRUNTIME_DIR/libonnxruntime.so"
            ln -sf `basename "$ONNXRUNTIME_LIB"` "$ONNXRUNTIME_DIR/libonnxruntime.so.1"
        fi
    fi

    CUDA_RUNTIME_INCLUDE=`python3 - <<'PY'
import pathlib,sys
p = pathlib.Path(sys.prefix) / 'lib/python3.8/site-packages/nvidia/cuda_runtime/include'
print(p if p.exists() else '')
PY`
    CUDA_NVCC_INCLUDE=`python3 - <<'PY'
import pathlib,sys
p = pathlib.Path(sys.prefix) / 'lib/python3.8/site-packages/nvidia/cuda_nvcc/include'
print(p if p.exists() else '')
PY`
    CUDA_CCCL_INCLUDE=`python3 - <<'PY'
import pathlib,sys
p = pathlib.Path(sys.prefix) / 'lib/python3.8/site-packages/nvidia/cuda_cccl/include'
print(p if p.exists() else '')
PY`
    CUDA_RUNTIME_LIBDIR=`python3 - <<'PY'
import pathlib,sys
p = pathlib.Path(sys.prefix) / 'lib/python3.8/site-packages/nvidia/cuda_runtime/lib'
print(p if p.exists() else '')
PY`
    NVIDIA_CUDA_LIB_DIRS=`python3 - <<'PY'
import pathlib,sys
base = pathlib.Path(sys.prefix) / 'lib/python3.8/site-packages/nvidia'
parts=[]
for name in ['cuda_runtime/lib','cublas/lib','cudnn/lib','cufft/lib','curand/lib','cuda_nvrtc/lib','nvjitlink/lib']:
    p = base / name
    if p.exists():
        parts.append(str(p))
print(':'.join(parts))
PY`
fi



echo "setup for deepflame bashrc:"
echo LIBCANTERA_DIR=$LIBCANTERA_DIR
if [ $USE_LIBTORCH = true ]; then
    echo LIBTORCH_DIR=$LIBTORCH_DIR
    echo PYTORCH_INC=""
    echo PYTORCH_LIB=""
fi
if [ $USE_PYTORCH = true ]; then
    echo PYTORCH_INC=$PYTORCH_INC
    echo PYTORCH_LIB=$PYTORCH_LIB
    echo ONNXRUNTIME_DIR=$ONNXRUNTIME_DIR
    echo ONNXRUNTIME_LIB=$ONNXRUNTIME_LIB
    echo TENSORRT_DIR=$TENSORRT_DIR
    echo TENSORRT_LIBDIR=$TENSORRT_LIBDIR
    echo CUDA_RUNTIME_INCLUDE=$CUDA_RUNTIME_INCLUDE
    echo CUDA_NVCC_INCLUDE=$CUDA_NVCC_INCLUDE
    echo CUDA_CCCL_INCLUDE=$CUDA_CCCL_INCLUDE
    echo CUDA_RUNTIME_LIBDIR=$CUDA_RUNTIME_LIBDIR
    echo NVIDIA_CUDA_LIB_DIRS=$NVIDIA_CUDA_LIB_DIRS
    echo LIBTORCH_DIR=""
fi
if [ $USE_GPUSOLVER = true ]; then
    echo AMGX_DIR=$AMGX_DIR
fi
if [ $USE_ODE_GPU_SOLVER = true ]; then
    echo ODE_GPU_SOLVER=$OPENCC_PATH
fi

cp bashrc.in bashrc
sed -i "s#pwd#$PWD#g" ./bashrc
sed -i "s#LIBTORCH_DIR#$LIBTORCH_DIR#g" ./bashrc
sed -i "s#PYTORCH_INC#$PYTORCH_INC#g" ./bashrc
sed -i "s#PYTORCH_LIB#$PYTORCH_LIB#g" ./bashrc
sed -i "s#LIBCANTERA_DIR#$LIBCANTERA_DIR#g" ./bashrc
sed -i "s#@ONNXRUNTIME_DIR@#$ONNXRUNTIME_DIR#g" ./bashrc
sed -i "s#@ONNXRUNTIME_LIB@#$ONNXRUNTIME_LIB#g" ./bashrc
sed -i "s#@TENSORRT_DIR@#$TENSORRT_DIR#g" ./bashrc
sed -i "s#@TENSORRT_LIBDIR@#$TENSORRT_LIBDIR#g" ./bashrc
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
