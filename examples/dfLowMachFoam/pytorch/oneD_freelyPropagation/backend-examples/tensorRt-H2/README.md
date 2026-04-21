# tensorRt-H2

Self-contained runnable H2 case using the TensorRT backend.

This case is configured to run to `endTime = 0.01`.

If files are missing, `./Allrun` rebuilds them using:

- `build_h2_onnx_wrapper.py`
- `build_h2_tensorrt_engine.sh`

Run with:

```bash
./Allrun
```
