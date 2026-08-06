# Trash Tracker: Iterative Development of a Mobile-Optimized Waste Classifier

## 1. Introduction

Recycling helps the environment, but it is harder than it seems. Many people want to recycle correctly, but the rules for materials like glass, plastic, and paper are confusing. Simple mistakes, like putting the wrong item in the bin, can ruin an entire batch of recycling. There are not many easy tools that tell someone exactly what they are holding and where it should go.

This project builds a mobile app to solve that problem using artificial intelligence. The app looks at an object through the phone's camera and tells the user its category right away. The model recognizes six types of waste: biodegradable, cardboard, glass, metal, paper, and plastic.

The app must work directly on the phone. It should not need to send data to the cloud or rely on a fast internet connection. Running the model locally keeps the app fast and keeps the user's data private. The main challenge is keeping the model small enough for a phone, but accurate enough to classify trash correctly.

## 2. System Architecture

The project has two main parts: the machine learning model and the mobile app.

For the model, I use YOLO (You Only Look Once), an object detection architecture. I started with the smallest version because it is lightweight and phones have limited processing power. In Phase 5, I moved to a larger version in the same family. This gives the model more capacity to tell similar materials apart, while still staying small enough to run on a phone. Training runs on a computer with an NVIDIA RTX 5080 GPU, which makes it possible to test different settings quickly. Once training is done, the model is converted to TFLite format, which is built for running efficiently on mobile devices.

The app itself is built with Flutter, so it runs on both Android and iOS. The app uses the phone's camera to capture an image, then the TFLite model classifies it. Everything runs locally, so the user only needs to point the camera at an item to see its category.

## 3. Model Training

### 3.1 Phase 1: Baseline Training and Results

The first step was an initial training run to see how a small model would handle the dataset. This used about 10,000 images across seven categories, trained for 100 epochs over roughly four hours. This gave a baseline model to measure future progress against.

The model's mAP50 (mean Average Precision) score was 0.60, meaning it identified objects correctly about 60% of the time. Results varied a lot by category. Glass had 88% precision, so when the model predicted glass, it was usually right. Paper had only 15% precision. The cause was an imbalanced dataset: over 13,000 biodegradable instances versus just 33 for paper. With so few paper examples, the model became biased toward the categories it saw most often.

<div align="center">
    <img src="ml_output/document_images/phase1_metrics.png" width="600">
    <p align="center"><strong>Fig. 1. </strong>Precision, recall, and mAP scores for the seven categories in Phase 1, showing Glass's 88% precision against Paper's 15%.<p>
</div>

Running the model on test images showed how it was thinking. Each prediction draws a box with a label and a confidence score. In one case, a clear glass bottle was labeled PLASTIC at 80% confidence — glass and plastic look alike when both are clear, so the model needs more varied glass examples.

<div align="center">
    <img src="ml_output/document_images/phase1_glass.jpg" width="300">
    <p align="center"><strong>Fig. 2. </strong>A clear glass object misidentified as plastic with 80% confidence.<p>
</div>

On images like the tomatoes and cardboard box, the model drew far too many boxes. It found the object, but split it into several overlapping boxes, which would look messy on screen.

<div align="center">
    <img src="ml_output/document_images/phase1_tomatoes.jpg" width="300">
    <p align="center"><strong>Fig. 3. </strong>Multiple overlapping boxes on a single set of tomatoes.<p>
</div>

<div align="center">
    <img src="ml_output/document_images/phase1_cardboard.jpg" width="300">
    <p align="center"><strong>Fig. 4. </strong>Duplicate boxes on one cardboard item.<p>
</div>

A cardboard box was also labeled PAPER at 86% confidence, likely because cardboard and paper share a similar texture, and the model has too few paper examples to tell them apart.

<div align="center">
    <img src="ml_output/document_images/phase1_cardboard2.jpg" width="300">
    <p align="center"><strong>Fig. 5. </strong>A cardboard box labeled Paper at 86% confidence.<p>
</div>

Despite the accuracy issues, speed was promising: processing one image took about 1.3 milliseconds. Even a slower phone processor should keep up in real time.

### 3.2 Phase 2: Data Augmentation and Model Optimization

Phase 1 showed two clear problems: bias against Paper, and cluttered multi-box detections. Phase 2 focused on the data itself rather than just training longer. A 3x augmentation strategy in Roboflow expanded the dataset from 10,000 to over 25,000 images, using random rotations, flips, and brightness changes so the model would learn general features instead of memorizing specific photos. The dataset was also rebalanced so Paper and Plastic appeared more often during training, instead of being overwhelmed by the 13,000+ Biodegradable instances from Phase 1. Training ran for 100 epochs over about 9.5 hours. The overall mAP50 stayed at 0.60, but the model became noticeably more robust.

<div align="center">
    <img src="ml_output/document_images/phase2_metrics.png" width="600">
    <p align="center"><strong>Fig. 6. </strong>Precision, recall, and mAP scores for the seven categories in Phase 2.<p>
</div>

The overall score stayed the same, but a closer look at the diagnostics tells a more detailed story.

<div align="center">
    <img src="ml_output/document_images/phase2_confusion_matrix_normalized.png" width="600">
    <p align="center"><strong>Fig. 7. </strong>Normalized confusion matrix for Phase 2, showing a 73% background-miss rate for Paper.<p>
</div>

The confusion matrix shows the core problem. Glass (78%) and Biodegradable (61%) are fairly reliable, but 73% of Paper instances are missed as background entirely — the model isn't even proposing a box for them, likely because there are only 33 paper examples. Plastic is also confused with Metal 16% of the time, suggesting the model needs clearer examples of shiny, reflective surfaces.

<div align="center">
    <img src="ml_output/document_images/phase2_BoxF1_curve.png" width="600">
    <p align="center"><strong>Fig. 8. </strong>F1-confidence curve showing an optimal threshold of 0.259.<p>
</div>

The F1 curve shows the best balance of precision and recall sits at a 0.259 confidence threshold. This number matters for the app later, since it sets the cutoff for filtering out false detections while still catching real items.

Validation images show a real cleanup in detection logic compared to Phase 1.

<div align="center">
    <img src="ml_output/document_images/phase2_val_batch1_pred.jpg" width="500">
    <p align="center"><strong>Fig. 9. </strong>Metal cans correctly separated into individual boxes.<p>
</div>

The model is now better at separating individual items in a cluster. Overlapping cans get distinct boxes instead of duplicates, a clear improvement over Phase 1 — though some duplication still shows up elsewhere.

<div align="center">
    <img src="ml_output/document_images/phase2_val_batch2_pred.jpg" width="500">
    <p align="center"><strong>Fig. 10. </strong>Severe over-detection on a pile of biodegradable items, split into dozens of overlapping boxes.<p>
</div>

Phase 2 also revealed a limitation that would resurface in later phases: over-detection on piles of organic matter. When items form a large, messy pile, the model tries to draw a box around every edge and texture change, creating a cluttered "explosion" of overlapping boxes. This creates a real tradeoff. Two apples sitting side by side should get two separate boxes, but a large, indistinguishable pile is better shown as one single box. Solving this means teaching the model when to treat items individually versus as a group.

The plan for Phase 3 has two parts. First, relabel large piles as a single object during annotation, instead of labeling every visible piece. Second, tune the IoU (Intersection over Union) threshold in the app so that boxes overlapping more than about 45% get merged into one, while separate side-by-side items stay distinct.

### 3.3 Phase 3: Dataset Rebalancing and Training Extension

Phase 3 targeted the Paper background-miss problem and the Biodegradable bias through rebalancing: over 8,000 new Paper instances were added, and redundant Biodegradable annotations were cleaned up. Training extended to 150 epochs with a batch size of 32 so the model could fully absorb the new data. The mAP50 reached 0.613, a stable result despite the added complexity.

<div align="center">
    <img src="ml_output/document_images/phase3_metrics.png" width="600">
    <p align="center"><strong>Fig. 11. </strong>Precision, recall, and mAP scores for the seven categories in Phase 3.<p>
</div>

The confusion matrix shows a breakthrough for Paper: the 73% background-miss rate from Phase 2 is gone, since the model now proposes a box for paper items. But 61% of those get misclassified as Biodegradable instead, likely because the validation set still has 7,490 Biodegradable instances against only 31 for Paper — the model leans toward the majority class for light, matte textures.

<div align="center">
    <img src="ml_output/document_images/phase3_confusion_matrix_normalized.png" width="600">
    <p align="center"><strong>Fig. 12. </strong>Glass (76%) and Metal (67%) stay reliable, but 61% of Paper instances are now confused with Biodegradable instead of missed entirely.<p>
</div>

The F1 curve shows the optimal threshold shifted up to 0.295, meaning the model is more confident in its detections. A higher cutoff means fewer false positives without losing accuracy.

<div align="center">
    <img src="ml_output/document_images/phase3_BoxF1_curve.png" width="600">
    <p align="center"><strong>Fig. 13. </strong>Optimal threshold shifted to 0.295, with a peak F1 score of 0.61.<p>
</div>

Validation images show real improvement in spatial logic — items in dense piles are isolated much better than in earlier phases, though the over-detection issue from Phase 2 is still present in some clusters, just less severe. A new label bias also appears: ground-truth Cardboard items are often predicted as Biodegradable.

<div align="center">
    <img src="ml_output/document_images/phase3_val_batch0_pred.jpg" width="500">
    <p align="center"><strong>Fig. 14. </strong>Strong feature extraction on metallic, cylindrical shapes, with confidence scores between 0.8 and 1.0.<p>
</div>

<div align="center">
    <img src="ml_output/document_images/phase3_val_batch1_pred.jpg" width="500">
    <p align="center"><strong>Fig. 15. </strong>A Cardboard item incorrectly predicted as Biodegradable.<p>
</div>

<div align="center">
    <img src="ml_output/document_images/phase3_val_batch2_pred.jpg" width="500">
    <p align="center"><strong>Fig. 16. </strong>Organic piles mostly separated into individual items, though some overlapping boxes remain.<p>
</div>

Phase 3 reached 0.613 mAP50, with Paper's background-miss problem resolved. The main remaining issue is confusion between light-colored processed materials and organic matter, driven by the Biodegradable-to-Paper instance imbalance. Phase 4 will shift focus from adding more data to adding negative (no-object) examples.

### 3.4 Phase 4: Precision Engineering and Background Calibration

Phase 4 focused on stopping false detections and fixing the mix-up between processed materials and organic matter. This meant adding 800 negative (no-object) examples to calibrate the background, raising the training resolution to 800px for finer texture detail, and applying adaptive equalization to fix errors with pale objects on pale backgrounds. Together these changes raised mAP50 to 0.689, a 12.4% gain over Phase 3.

<div align="center">
    <img src="ml_output/document_images/phase4_metrics.png" width="600">
    <p align="center"><strong>Fig. 17. </strong>Paper stabilizes at 0.758 mAP50; Glass stays strong at 0.819.<p>
</div>

The confusion matrix confirms the background calibration worked. False positives on empty surfaces dropped across every category, and Paper now has a 74% true positive rate. One tradeoff remains: 61% of Biodegradable items are now misclassified as background instead of detected. This makes the model more conservative — it stays quiet until an item is clearly visible, which is better for real-world use even though it means missing some true positives.

<div align="center">
    <img src="ml_output/document_images/phase4_confusion_matrix_normalized.png" width="600">
    <p align="center"><strong>Fig. 18. </strong>The model is now more conservative, preferring to miss a vague shape over falsely flagging a clean surface.<p>
</div>

The F1 curve's optimal threshold shifted to 0.272, with a peak score of 0.67. Paper's curve is now noticeably higher and broader than before, showing the model recognizes processed textures confidently across a wider range of thresholds.

<div align="center">
    <img src="ml_output/document_images/phase4_BoxF1_curve.png" width="600">
    <p align="center"><strong>Fig. 19. </strong>Every category peaks higher on the F1 scale, averaging 0.67 overall.<p>
</div>

Validation images show the benefit of the higher resolution: plastic items with crinkled textures and transparent layers are now isolated with 0.9-1.0 confidence, and the negative examples pay off, with complex backgrounds producing zero false positives. The over-detection issue on dense organic piles is still present at this point and still needs further NMS (Non-Maximum Suppression) tuning.

<div align="center">
    <img src="ml_output/document_images/phase4_val_batch0_pred.jpg" width="500">
    <p align="center"><strong>Fig. 20. </strong>Clear and stacked plastic items detected with high confidence, even in a messy scene.<p>
</div>

<div align="center">
    <img src="ml_output/document_images/phase4_val_batch1_pred.jpg" width="500">
    <p align="center"><strong>Fig. 21. </strong>Complex non-waste backgrounds correctly produce no detections.<p>
</div>

<div align="center">
    <img src="ml_output/document_images/phase4_biodegradable.jpg" width="500">
    <p align="center"><strong>Fig. 22. </strong>Organic piles still generate multiple overlapping boxes for single items.<p>
</div>

Phase 4 reached 0.689 mAP50 and removed most ghost detections, with Paper now performing reliably. Overlapping boxes on organic piles remain the main visual issue, but the model was considered ready for export into the Flutter app.

### 3.5 Phase 5: Split Diagnosis and Model Capacity Upgrade

Phase 5 started by checking the dataset split instead of the model. Biodegradable, Cardboard, and Glass were barely showing up in the test set — as few as 3-6 images each — even though they were well represented in training. This was fixed by manually moving batches of images from training into the test split in Roboflow, then regenerating the dataset. The test set grew from 220 to 370 images, using the same preprocessing and augmentation as Phase 4 (auto-orient, horizontal flip, ±15° rotation, ±25% brightness, up to 2.5px blur).

<div align="center">
    <img src="ml_output/document_images/phase5_confusion_matrix_normalized.png" width="600">
    <p align="center"><strong>Fig. 23. </strong>With a balanced test split, Biodegradable reaches 82% accuracy. Glass and Metal are still confused with each other, and Plastic has the highest false-positive rate of any class.<p>
</div>

Retraining on the fixed split gave an overall mAP50 of 0.699. Glass was the strongest class at 0.849, while Plastic was the weakest at 0.536, despite having the most training examples of any category. The precision-recall curve confirms this is a real problem, not just a threshold issue: Plastic's curve stays below every other class across the board.

<div align="center">
    <img src="ml_output/document_images/phase5_BoxPR_curve.png" width="600">
    <p align="center"><strong>Fig. 24. </strong>Glass leads the precision-recall curve; Plastic falls behind at every threshold.<p>
</div>

Checking the validation images explained the two main confusions. A glass measuring cup was predicted as Metal, likely due to its shiny, reflective surface. A frosted plastic bottle was predicted as Glass, since frosted plastic and frosted glass look almost the same in a photo. A grey egg carton (true label Cardboard) was also predicted as Metal, likely due to its slightly metallic texture.

<div align="center">
    <img src="ml_output/document_images/phase5_val_batch1_pred.jpg" width="500">
    <p align="center"><strong>Fig. 25. </strong>A glass measuring cup predicted as Metal at 40% confidence.<p>
</div>

Training curves showed no overfitting, and metrics leveled off around epoch 25-30 before early stopping at epoch 63. Since the model had fully converged, more epochs would not fix the Glass/Metal/Plastic confusion. Phase 5 instead moved to a larger model for the next round, trading a bigger file size for more capacity to tell similar materials apart.

## 4. Model Workflow and Maintenance

The app ships with two runtime files: `assets/best_float32.tflite` (the trained model) and `assets/labels.txt` (the class names, in the same order the model outputs them).

To retrain, run `notebooks/train_trash_tracker.ipynb`. It downloads the dataset from Roboflow, audits class balance, trains YOLO26 at 800px, validates and tests on `ml_output/test_images/`, and exports a TFLite file to copy into `assets/`. Dataset downloads and training runs are all written to `ml_output/` at the repo root, so they stay out of the `notebooks/` folder and out of the repo root's way.

Setup (one time): from inside `notebooks/`, run `setup_env.bat` to create the `trash_tracker` conda environment and register it as a Jupyter kernel. Each session after that, from inside `notebooks/`:

```bash
conda activate trash_tracker
set ROBOFLOW_API_KEY=your_key_here
jupyter notebook --notebook-dir="%cd%"
```

Select the "Python (trash_tracker)" kernel before running cells.

TFLite export only works on Linux x86 or macOS, since Ultralytics moved to the LiteRT export path. It fails on Windows — export through WSL2, a cloud notebook, or the Ultralytics Platform instead.

A few rules keep the runtime model consistent: do not append raw bytes or JSON to `.tflite` files, replace `assets/best_float32.tflite` only with a clean Ultralytics export, keep one active runtime model in `assets/`, and keep `labels.txt`'s class order aligned with the training `data.yaml`.

The app expects a fixed model contract: input `[1, 800, 800, 3]` (float RGB, normalized to `[0, 1]`), output `[1, 300, 6]`, where each row is `x1, y1, x2, y2, confidence, class_id` in normalized xyxy format. Confidence thresholds live in `lib/detection_parser.dart` (`candidateConfidenceThreshold`, `confirmedConfidenceThreshold`). After a scan, the app pauses the camera preview and draws boxes on the captured image; tapping **Scan again** resumes the live preview.

## References
