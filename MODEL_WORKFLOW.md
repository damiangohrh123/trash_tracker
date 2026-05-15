# Model Workflow

This project uses one runtime model path:

- `assets/best_float32.tflite`
- `assets/labels.txt`

## Rules

1. Do not append raw bytes/JSON/FlatBuffer blobs to a `.tflite` file.
2. Replace `assets/best_float32.tflite` only with a clean export from your training/conversion pipeline.
3. Keep only one active runtime model in `assets/` to avoid confusion.
4. If you generate metadata, use official TensorFlow Lite metadata tooling only.

## Pre-run checklist

1. Confirm model file exists:
   - `assets/best_float32.tflite`
2. Confirm labels file exists:
   - `assets/labels.txt`
3. Optional metadata check:
   - `python verify_model_metadata.py`
   - or `python verify_model_metadata.py assets/best_float32.tflite`

## Notes

- The app uses **`tflite_flutter`** (not ML Kit) so it can run **Ultralytics YOLO** exports. ML Kit custom object detection expects a different model contract than typical YOLO TFLite outputs.
- The app copies `assets/best_float32.tflite` and `assets/labels.txt` at runtime before loading the interpreter.
- Any experimental scripts should live outside the runtime path and should never mutate the production model file in place.

## Export from Jupyter (Ultralytics YOLO)

After training, export a clean TFLite from your best weights (adjust path and `imgsz` to match training):

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
model.export(format="tflite", imgsz=640)
```

Copy the generated `.tflite` into `assets/best_float32.tflite` (or change the filename in `pubspec.yaml` and `lib/main.dart` to match).

### Input layout

The app detects **NHWC** `[1, H, W, 3]` vs **NCHW** `[1, 3, H, W]` from the model input tensor shape and preprocesses accordingly. If your export uses different normalization (e.g. letterbox, mean/std), update `_buildInputTensor` in `lib/main.dart`.

### Output decoding

Inference runs and shows a **debug summary** of raw output values. Next step is to implement YOLO-specific decode + NMS for your exact output tensor shape (Ultralytics version dependent).
