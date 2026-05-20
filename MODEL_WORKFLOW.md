# Model Workflow

Runtime assets:

- `assets/best_float32.tflite`
- `assets/labels.txt`

## Rules

1. Do not append raw bytes or JSON to `.tflite` files.
2. Replace `assets/best_float32.tflite` only with a clean Ultralytics export.
3. Keep one active runtime model in `assets/`.
4. Keep `labels.txt` class order aligned with training `data.yaml`.

## Export from Jupyter

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
model.export(format="tflite", imgsz=800, int8=False)
```

Copy the exported `.tflite` to `assets/best_float32.tflite`.

## App decoding contract

Current model contract:

- Input: `[1, 800, 800, 3]`, float RGB normalized to `[0, 1]`
- Output: `[1, 300, 6]`
- Row format: `x1, y1, x2, y2, confidence, class_id` (normalized xyxy)
- Confidence threshold: `0.25` (see `lib/detection_parser.dart`)

Next planned feature: pause camera preview and render bounding boxes on captured image.
