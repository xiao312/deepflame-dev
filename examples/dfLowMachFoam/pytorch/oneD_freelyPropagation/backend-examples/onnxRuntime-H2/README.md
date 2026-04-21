# onnxRuntime-H2

Self-contained runnable H2 case using the ONNX Runtime backend.

This case is configured to run to `endTime = 0.01`.

If `h2_full_inference_wrapper.onnx` is missing, `./Allrun` rebuilds it with:

- `build_h2_onnx_wrapper.py`

Run with:

```bash
./Allrun
```
