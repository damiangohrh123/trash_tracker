# Trash Tracker: Iterative Development of a Mobile-Optimized Waste Classifier

## 1. Introduction

Recycling rules for materials like glass, plastic, and paper confuse people, and one wrong item can ruin a whole batch. This project is a mobile app that solves that: point the camera at an item, and it names the category right away. The model recognizes six types of waste — biodegradable, cardboard, glass, metal, paper, and plastic.

The app runs entirely on the phone, with no cloud calls, so it stays fast and private. The main challenge is keeping the model small enough for a phone while still accurate enough to sort trash correctly.

## 2. System Architecture

The project has two parts: the YOLO (You Only Look Once) detection model and the Flutter app. Training started on the smallest YOLO variant for speed, then moved to a larger one in Phase 5 for more capacity to separate similar materials. Training runs on an NVIDIA RTX 5080 GPU; the finished model exports to TFLite for on-device use.

Flutter lets the app run on both Android and iOS from one codebase. It captures a photo, runs the TFLite model locally, and shows the result — no server round-trip.

## 3. Model Training

### 3.1 Phase 1: Baseline Training and Results

A first run trained a small model on ~10,000 images across seven categories, 100 epochs (~4 hours), as a baseline.

<div align="center">
    <img src="ml_output/document_images/phase1_metrics.png" width="600">
    <p align="center"><strong>Fig. 1. </strong>Precision, recall, and mAP scores for the seven categories in Phase 1, showing Glass's 88% precision against Paper's 15%.<p>
</div>

mAP50 landed at 0.60, but accuracy varied sharply by class — Glass hit 88% precision, Paper only 15%. The cause: a skewed dataset, 13,000+ Biodegradable instances against just 33 for Paper, so the model leaned hard on whatever it saw most.

Test images exposed two more issues: look-alike materials confused each other (a clear glass bottle read as PLASTIC at 80% confidence), and dense objects got split into multiple overlapping boxes instead of one. A cardboard box was also read as PAPER at 86% confidence. Inference speed was still strong at 1.3ms per image.

<div align="center">
<table><tr>
<td align="center"><img src="ml_output/document_images/phase1_glass.jpg" width="190"><br><sub><strong>Fig. 2.</strong> Glass misread as plastic (80%).</sub></td>
<td align="center"><img src="ml_output/document_images/phase1_tomatoes.jpg" width="190"><br><sub><strong>Fig. 3.</strong> Overlapping boxes on tomatoes.</sub></td>
<td align="center"><img src="ml_output/document_images/phase1_cardboard.jpg" width="190"><br><sub><strong>Fig. 4.</strong> Duplicate boxes on cardboard.</sub></td>
<td align="center"><img src="ml_output/document_images/phase1_cardboard2.jpg" width="190"><br><sub><strong>Fig. 5.</strong> Cardboard read as Paper (86%).</sub></td>
</tr></table>
</div>

### 3.2 Phase 2: Data Augmentation and Model Optimization

Phase 2 targeted the Paper bias and the duplicate-box clutter through the data, not longer training: 3x augmentation (rotation, flips, brightness) grew the dataset from 10,000 to 25,000+ images, and Paper/Plastic were rebalanced against the 13,000+ Biodegradable instances. Trained 100 epochs (~9.5 hours). Overall mAP50 held at 0.60, but the model got noticeably more robust underneath that number.

<div align="center">
    <img src="ml_output/document_images/phase2_metrics.png" width="600">
    <p align="center"><strong>Fig. 6. </strong>Per-class precision, recall, and mAP scores, Phase 2.<p>
</div>

<div align="center">
    <img src="ml_output/document_images/phase2_confusion_matrix_normalized.png" width="600">
    <p align="center"><strong>Fig. 7. </strong>73% of Paper instances missed as background.<p>
</div>

<div align="center">
    <img src="ml_output/document_images/phase2_BoxF1_curve.png" width="600">
    <p align="center"><strong>Fig. 8. </strong>F1-confidence curve, optimal threshold 0.259.<p>
</div>

The confusion matrix explains why: Glass (78%) and Biodegradable (61%) held up fine, but 73% of Paper instances got no box at all — only 33 examples to learn from — and Plastic was confused with Metal 16% of the time. The F1 curve's best threshold, 0.259, matters later for filtering the app's false detections.

<div align="center">
<table><tr>
<td align="center"><img src="ml_output/document_images/phase2_val_batch1_pred.jpg" width="380"><br><sub><strong>Fig. 9.</strong> Metal cans now separated correctly.</sub></td>
<td align="center"><img src="ml_output/document_images/phase2_val_batch2_pred.jpg" width="380"><br><sub><strong>Fig. 10.</strong> Over-detection on a biodegradable pile.</sub></td>
</tr></table>
</div>

Box separation improved (Fig. 9), but dense piles of organic matter still exploded into dozens of overlapping boxes (Fig. 10) — the model draws a box at every texture change instead of treating a pile as one object. The Phase 3 plan: relabel large piles as single objects during annotation, and merge boxes overlapping more than ~45% IoU in the app.

### 3.3 Phase 3: Dataset Rebalancing and Training Extension

Phase 3 added 8,000+ new Paper instances and cleaned up redundant Biodegradable annotations, then trained 150 epochs at batch size 32. mAP50 reached 0.613.

<div align="center">
    <img src="ml_output/document_images/phase3_metrics.png" width="600">
    <p align="center"><strong>Fig. 11. </strong>Per-class precision, recall, and mAP scores, Phase 3.<p>
</div>

<div align="center">
    <img src="ml_output/document_images/phase3_confusion_matrix_normalized.png" width="600">
    <p align="center"><strong>Fig. 12. </strong>Paper now confused with Biodegradable (61%) instead of missed entirely.<p>
</div>

<div align="center">
    <img src="ml_output/document_images/phase3_BoxF1_curve.png" width="600">
    <p align="center"><strong>Fig. 13. </strong>F1-confidence curve, threshold shifted to 0.295.<p>
</div>

Paper's 73% background-miss from Phase 2 is gone — the model now proposes a box — but 61% of those get misclassified as Biodegradable, since the validation set still has 7,490 Biodegradable instances against only 31 Paper. Glass (76%) and Metal (67%) stayed reliable, and the F1 threshold rose to 0.295 (peak F1 0.61), meaning fewer false positives.

<div align="center">
<table><tr>
<td align="center"><img src="ml_output/document_images/phase3_val_batch0_pred.jpg" width="250"><br><sub><strong>Fig. 14.</strong> Strong confidence on metallic shapes.</sub></td>
<td align="center"><img src="ml_output/document_images/phase3_val_batch1_pred.jpg" width="250"><br><sub><strong>Fig. 15.</strong> Cardboard read as Biodegradable.</sub></td>
<td align="center"><img src="ml_output/document_images/phase3_val_batch2_pred.jpg" width="250"><br><sub><strong>Fig. 16.</strong> Organic piles mostly separated.</sub></td>
</tr></table>
</div>

Spatial separation on dense piles improved further, though not fully solved, and a new bias appeared: Cardboard items often read as Biodegradable. With Paper's background-miss problem resolved, Phase 4 shifted focus to adding negative (no-object) examples instead of more data.

### 3.4 Phase 4: Precision Engineering and Background Calibration

Phase 4 added 800 negative examples to calibrate the background, raised training resolution to 800px, and applied adaptive equalization for pale-on-pale objects. mAP50 rose to 0.689, a 12.4% gain over Phase 3.

<div align="center">
    <img src="ml_output/document_images/phase4_metrics.png" width="600">
    <p align="center"><strong>Fig. 17. </strong>Paper stabilizes at 0.758 mAP50; Glass stays strong at 0.819.<p>
</div>

<div align="center">
    <img src="ml_output/document_images/phase4_confusion_matrix_normalized.png" width="600">
    <p align="center"><strong>Fig. 18. </strong>The model is now more conservative: fewer false positives.<p>
</div>

<div align="center">
    <img src="ml_output/document_images/phase4_BoxF1_curve.png" width="600">
    <p align="center"><strong>Fig. 19. </strong>Every class peaks higher on the F1 scale (avg. 0.67).<p>
</div>

Background calibration worked: false positives dropped across every category and Paper reached a 74% true-positive rate. The tradeoff is that 61% of Biodegradable items now read as background instead of being detected — the model got more conservative, missing vague shapes rather than flagging clean surfaces, which suits real-world use.

<div align="center">
<table><tr>
<td align="center"><img src="ml_output/document_images/phase4_val_batch0_pred.jpg" width="250"><br><sub><strong>Fig. 20.</strong> Plastic textures at 0.9-1.0 confidence.</sub></td>
<td align="center"><img src="ml_output/document_images/phase4_val_batch1_pred.jpg" width="250"><br><sub><strong>Fig. 21.</strong> Complex backgrounds, zero false positives.</sub></td>
<td align="center"><img src="ml_output/document_images/phase4_biodegradable.jpg" width="250"><br><sub><strong>Fig. 22.</strong> Organic piles still over-detected.</sub></td>
</tr></table>
</div>

Higher resolution paid off on plastic's crinkled, transparent textures, and the negative examples eliminated false positives on complex backgrounds. Organic piles still over-detect and need more NMS tuning, but the model was otherwise ready to export into the app.

### 3.5 Phase 5: Split Diagnosis and Model Capacity Upgrade

Phase 5 started by checking the dataset split, not the model: Biodegradable, Cardboard, and Glass had as few as 3-6 images each in the test set despite being well represented in training. Fixing this — moving images from training into test in Roboflow — grew the test set from 220 to 370 images, using the same preprocessing as Phase 4.

<div align="center">
    <img src="ml_output/document_images/phase5_confusion_matrix_normalized.png" width="600">
    <p align="center"><strong>Fig. 23. </strong>With a balanced test split, Biodegradable reaches 82% accuracy.<p>
</div>

<div align="center">
    <img src="ml_output/document_images/phase5_BoxPR_curve.png" width="600">
    <p align="center"><strong>Fig. 24. </strong>Glass leads the precision-recall curve; Plastic trails at every threshold.<p>
</div>

Retraining on the fixed split gave mAP50 0.699. Glass was strongest (0.849); Plastic was weakest (0.536) despite having the most training examples of any class — the PR curve confirms this is a real weakness, not a threshold artifact.

<div align="center">
    <img src="ml_output/document_images/phase5_val_batch1_pred.jpg" width="420">
    <p align="center"><strong>Fig. 25. </strong>A glass measuring cup predicted as Metal at 40% confidence.<p>
</div>

Validation images pointed to look-alike materials: a glass measuring cup read as Metal (shiny surface), a frosted plastic bottle read as Glass, and a grey egg carton read as Metal (slightly metallic texture). Training curves showed no overfitting, plateauing around epoch 25-30 before stopping at epoch 63 — the model had converged, so more epochs wouldn't fix the Glass/Metal/Plastic confusion. Phase 5 moved to a larger model instead, trading file size for the extra capacity needed to tell these materials apart.

### 3.6 Phase 6: Evaluating the Larger Model

Phase 6 tested the larger model from Phase 5's capacity upgrade, training on the same fixed dataset split. It ran 135 epochs (~2.5 hours) before early stopping, with the best weights saved at epoch 105.

mAP50 rose to 0.723, up from 0.699 in Phase 5 — a real but modest gain. Glass drove most of it, climbing to 0.897 and no longer meaningfully confused with Metal. Plastic barely moved (0.536 → 0.559) and still trails every other class on the precision-recall curve, suggesting its weakness is visual ambiguity rather than model capacity.

<div align="center">
    <img src="ml_output/document_images/phase6_confusion_matrix_normalized.png" width="600">
    <p align="center"><strong>Fig. 26. </strong>Glass is no longer confused with Metal, but Cardboard now has the biggest background-miss rate (25%).<p>
</div>

The bigger surprise was Cardboard, not flagged as a problem before: only 57% of true Cardboard instances are classified correctly, 25% are missed as background — the worst rate of any class — and 13% are confused with Paper.

<div align="center">
    <img src="ml_output/document_images/phase6_BoxPR_curve.png" width="600">
    <p align="center"><strong>Fig. 27. </strong>Glass leads the pack; Plastic still trails every class at every threshold.<p>
</div>

Training curves showed healthy convergence — validation loss tracked training loss with no divergence, and metrics plateaued around epoch 30, well before the epoch-135 stop. Validation images matched the numbers: a twisted plastic bottle still split into two boxes (0.9 and 0.3 confidence), and a cardboard egg carton was misread as PLASTIC.

<div align="center">
    <img src="ml_output/document_images/phase6_val_batch0_pred.jpg" width="700">
    <p align="center"><strong>Fig. 28. </strong>Correct calls at low confidence, a duplicate box on one bottle, and an egg carton misread as Plastic.<p>
</div>

The larger model paid off for Glass but not Plastic, pointing away from model capacity as the fix there. Cardboard's new background-miss rate is the next thing to dig into — likely needs more negative examples with cardboard-like textures, or a rebalance now that some of Paper's old confusion has shifted onto it.

## 4. Model Workflow and Maintenance

The app ships with two runtime files: `assets/best_float32.tflite` (the trained model) and `assets/labels.txt` (the class names, in the same order the model outputs them).

To retrain, run `notebooks/train_trash_tracker.ipynb`. It downloads the dataset from Roboflow, audits class balance, trains YOLO26 at 800px, validates and tests on `ml_output/test_images/`, and exports a TFLite file to copy into `assets/`. Dataset downloads and training runs are all written to `ml_output/` at the repo root, so they stay out of the `notebooks/` folder and out of the repo root's way.

Setup (one time): create a Python environment with the packages in `notebooks/requirements.txt` and register it as a Jupyter kernel. Also copy `notebooks/.env.example` to `notebooks/.env` and fill in your Roboflow API key — the notebook loads it from there automatically. Each session after that, from inside `notebooks/`:

```bash
conda activate trash_tracker
jupyter notebook --notebook-dir="%cd%"
```

Select the "Python (trash_tracker)" kernel before running cells.

TFLite export only works on Linux x86 or macOS, since Ultralytics moved to the LiteRT export path. It fails on Windows — export through WSL2, a cloud notebook, or the Ultralytics Platform instead.

A few rules keep the runtime model consistent: do not append raw bytes or JSON to `.tflite` files, replace `assets/best_float32.tflite` only with a clean Ultralytics export, keep one active runtime model in `assets/`, and keep `labels.txt`'s class order aligned with the training `data.yaml`.

The app expects a fixed model contract: input `[1, 800, 800, 3]` (float RGB, normalized to `[0, 1]`), output `[1, 300, 6]`, where each row is `x1, y1, x2, y2, confidence, class_id` in normalized xyxy format. Confidence thresholds live in `lib/detection_parser.dart` (`candidateConfidenceThreshold`, `confirmedConfidenceThreshold`). After a scan, the app pauses the camera preview and draws boxes on the captured image; tapping **Scan again** resumes the live preview.

## References
