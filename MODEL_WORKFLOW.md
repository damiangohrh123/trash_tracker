# Model Workflow

Runtime assets:

- `assets/best_float32.tflite`
- `assets/labels.txt`

## Training notebook

Use `train_trash_tracker.ipynb` (repo root) for a full retrain pipeline:

1. Configure Roboflow workspace / project / version
2. Audit class balance and negative images
3. Train YOLO26 at 800px
4. Validate and test on `ml_output/test_images/`
5. Export TFLite and copy to `assets/`

Note: the notebook switches its working directory to `ml_output/` early on, so the Roboflow dataset download, training runs, and test images all land there instead of cluttering the repo root.

Setup (one-time): run `setup_env.bat` from the repo root to create the `trash_tracker` conda env and register it as a Jupyter kernel. Then, each session:

```bash
conda activate trash_tracker
set ROBOFLOW_API_KEY=your_key_here
jupyter notebook --notebook-dir="%cd%"
```

Select the "Python (trash_tracker)" kernel before running cells.

Note: TFLite export (`model.export(format="tflite", ...)`) only works on Linux x86 or macOS as of Ultralytics' move to the LiteRT export path — it will fail on Windows. Export via WSL2, a cloud notebook, or the Ultralytics Platform instead.

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
- Confidence thresholds: see `lib/detection_parser.dart` (`candidateConfidenceThreshold`, `confirmedConfidenceThreshold`)

The app pauses the camera preview after capture and renders bounding boxes on the captured image. Tap **Scan again** to resume live preview.
