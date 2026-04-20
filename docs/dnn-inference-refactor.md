# DNN inference refactor notes

This branch starts a non-breaking refactor of the DNN chemistry inference path.

## Goals

- preserve the current external behavior of `dfChemistryModel`
- make future ONNX Runtime and TensorRT backends easy to add
- reduce cognitive load by separating:
  - backend dispatch
  - preprocessing
  - model execution
  - postprocessing
- keep the legacy public API so the surrounding solver code does not change all at once

## What changed in this step

- `DNNInferencer` now uses a small pImpl/facade design
- backend-specific TorchScript logic is hidden in `DNNInferencer.cpp`
- preprocessing and postprocessing are factored into helper functions
- the public `Inference(...)` and `Inference_multiDNNs(...)` APIs are preserved

## What did not change

- `dfChemistryModel` call sites
- MPI gather/scatter behavior in `libtorchFunctions.H`
- model normalization constants
- expected input packing order
- output reconstruction semantics

## Recommended next steps

1. Introduce a backend interface at the `dfChemistryModel` level:
   - `PyTorchPythonBackend`
   - `LibTorchBackend`
   - `OnnxRuntimeBackend`
   - `TensorRTBackend`
2. Move shared preprocessing/postprocessing out of TorchScript-only code.
3. Add offline regression tests for:
   - feature packing
   - normalization
   - output reconstruction
   - backend equivalence on frozen batches
4. Only after equivalence is established:
   - add persistent device buffers
   - add ONNX Runtime I/O binding
   - add TensorRT engine execution
   - revisit MPI batching topology
