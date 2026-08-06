import 'dart:math' show max, min;

/// Minimum score to read a row from model output (Phase 4 val optimum ~0.27).
const double candidateConfidenceThreshold = 0.25;

/// Minimum score to treat a detection as a confident result in the banner.
const double confirmedConfidenceThreshold = 0.50;

const double nmsIouThreshold = 0.45;
const int maxDisplayedDetections = 20;

/// Large, low-confidence boxes are usually background objects (screens, desks).
const double largeBoxAreaRatio = 0.22;
const double largeBoxConfidenceCap = 0.50;

/// Parses Ultralytics YOLO TFLite output shaped `[1, N, 6]`.
/// Each row is: x1, y1, x2, y2, confidence, classId (normalized xyxy).
List<Detection> parseDetections(dynamic output, List<String> labels) {
  if (output is! List || output.isEmpty) return const [];

  final batch = output.first;
  if (batch is! List) return const [];

  final detections = <Detection>[];

  for (final row in batch) {
    if (row is! List || row.length < 6) continue;

    final x1 = _asDouble(row[0]).clamp(0.0, 1.0);
    final y1 = _asDouble(row[1]).clamp(0.0, 1.0);
    final x2 = _asDouble(row[2]).clamp(0.0, 1.0);
    final y2 = _asDouble(row[3]).clamp(0.0, 1.0);
    final confidence = _asDouble(row[4]);
    final classId = _asDouble(row[5]).round();

    if (confidence < candidateConfidenceThreshold) continue;
    if (x2 <= x1 || y2 <= y1) continue;
    if (classId < 0 || classId >= labels.length) continue;

    detections.add(
      Detection(
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2,
        confidence: confidence,
        label: labels[classId],
      ),
    );
  }

  detections.sort((a, b) => b.confidence.compareTo(a.confidence));
  final deduped = _applyNms(detections);
  if (deduped.length > maxDisplayedDetections) {
    return deduped.sublist(0, maxDisplayedDetections);
  }
  return deduped;
}

List<Detection> filterSceneFalsePositives(List<Detection> detections) {
  return detections.where((detection) => !detection.isSuspiciousSceneMatch).toList();
}

List<Detection> mapDetectionsToOriginalImage(
  List<Detection> detections,
  LetterboxMapping mapping,
) {
  return detections
      .map((detection) => detection.mapToOriginalImage(mapping))
      .where((detection) => detection.isValid)
      .toList();
}

DetectionSummary summarizeDetections(List<Detection> detections) {
  final filtered = filterSceneFalsePositives(detections);
  final confirmed = filtered.where((detection) => detection.isConfirmed).toList();

  if (confirmed.isEmpty) {
    if (filtered.isNotEmpty) {
      return DetectionSummary.uncertain(filtered.length);
    }
    return const DetectionSummary.empty();
  }

  final classCounts = <String, int>{};
  final topConfidenceByLabel = <String, double>{};

  for (final detection in confirmed) {
    classCounts.update(detection.label, (count) => count + 1, ifAbsent: () => 1);
    final existing = topConfidenceByLabel[detection.label];
    if (existing == null || detection.confidence > existing) {
      topConfidenceByLabel[detection.label] = detection.confidence;
    }
  }

  final top = confirmed.first;
  return DetectionSummary(
    kind: SummaryKind.detected,
    topLabel: top.label,
    topConfidence: top.confidence,
    detectionCount: confirmed.length,
    classCounts: classCounts,
    topConfidenceByLabel: topConfidenceByLabel,
    uncertainCount: filtered.length - confirmed.length,
  );
}

List<Detection> _applyNms(List<Detection> detections) {
  final kept = <Detection>[];
  for (final detection in detections) {
    final overlapsSameClass = kept.any(
      (existing) =>
          existing.label == detection.label &&
          _iou(existing, detection) >= nmsIouThreshold,
    );
    if (!overlapsSameClass) {
      kept.add(detection);
    }
  }
  return kept;
}

double _iou(Detection a, Detection b) {
  final x1 = max(a.x1, b.x1);
  final y1 = max(a.y1, b.y1);
  final x2 = min(a.x2, b.x2);
  final y2 = min(a.y2, b.y2);

  final intersectionWidth = max(0.0, x2 - x1);
  final intersectionHeight = max(0.0, y2 - y1);
  final intersection = intersectionWidth * intersectionHeight;
  if (intersection <= 0) return 0.0;

  final areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  final areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return intersection / (areaA + areaB - intersection);
}

double _asDouble(dynamic value) {
  if (value is num) return value.toDouble();
  return 0.0;
}

class LetterboxMapping {
  const LetterboxMapping({
    required this.scale,
    required this.padX,
    required this.padY,
    required this.originalWidth,
    required this.originalHeight,
    required this.inputWidth,
    required this.inputHeight,
  });

  final double scale;
  final double padX;
  final double padY;
  final int originalWidth;
  final int originalHeight;
  final int inputWidth;
  final int inputHeight;
}

class Detection {
  const Detection({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    required this.confidence,
    required this.label,
  });

  final double x1;
  final double y1;
  final double x2;
  final double y2;
  final double confidence;
  final String label;

  bool get isValid => x2 > x1 && y2 > y1;

  double get areaRatio => (x2 - x1) * (y2 - y1);

  bool get isConfirmed => confidence >= confirmedConfidenceThreshold;

  bool get isSuspiciousSceneMatch =>
      areaRatio >= largeBoxAreaRatio && confidence < largeBoxConfidenceCap;

  Detection mapToOriginalImage(LetterboxMapping mapping) {
    double toOriginalX(double normalizedX) {
      final pixel = normalizedX * mapping.inputWidth;
      final unpadded = (pixel - mapping.padX) / mapping.scale;
      return (unpadded / mapping.originalWidth).clamp(0.0, 1.0);
    }

    double toOriginalY(double normalizedY) {
      final pixel = normalizedY * mapping.inputHeight;
      final unpadded = (pixel - mapping.padY) / mapping.scale;
      return (unpadded / mapping.originalHeight).clamp(0.0, 1.0);
    }

    return Detection(
      x1: toOriginalX(x1),
      y1: toOriginalY(y1),
      x2: toOriginalX(x2),
      y2: toOriginalY(y2),
      confidence: confidence,
      label: label,
    );
  }
}

enum SummaryKind { empty, uncertain, detected }

class DetectionSummary {
  const DetectionSummary({
    required this.kind,
    required this.topLabel,
    required this.topConfidence,
    required this.detectionCount,
    required this.classCounts,
    required this.topConfidenceByLabel,
    this.uncertainCount = 0,
  });

  const DetectionSummary.empty()
      : kind = SummaryKind.empty,
        topLabel = null,
        topConfidence = 0,
        detectionCount = 0,
        classCounts = const {},
        topConfidenceByLabel = const {},
        uncertainCount = 0;

  DetectionSummary.uncertain(int weakDetectionCount)
      : kind = SummaryKind.uncertain,
        topLabel = null,
        topConfidence = 0,
        detectionCount = 0,
        classCounts = const {},
        topConfidenceByLabel = const {},
        uncertainCount = weakDetectionCount;

  final SummaryKind kind;
  final String? topLabel;
  final double topConfidence;
  final int detectionCount;
  final Map<String, int> classCounts;
  final Map<String, double> topConfidenceByLabel;
  final int uncertainCount;

  bool get hasDetections => detectionCount > 0;

  String toDisplayText() {
    if (kind == SummaryKind.empty) {
      return 'No trash detected.\nFill the frame with waste items.';
    }

    if (kind == SummaryKind.uncertain) {
      return 'Nothing confirmed.\nMove closer to individual waste items.';
    }

    if (detectionCount == 1 && topLabel != null) {
      final percent = (topConfidence * 100).toStringAsFixed(1);
      return '$topLabel ($percent%)';
    }

    final labels = topConfidenceByLabel.keys.toList()
      ..sort((a, b) => topConfidenceByLabel[b]!.compareTo(topConfidenceByLabel[a]!));

    final breakdown = labels.map((label) {
      final count = classCounts[label]!;
      final percent = (topConfidenceByLabel[label]! * 100).toStringAsFixed(0);
      if (count > 1) {
        return '$label x$count ($percent%)';
      }
      return '$label ($percent%)';
    }).join(' · ');

    final header = '$detectionCount items detected';
    if (uncertainCount > 0) {
      return '$header\n$breakdown\n+$uncertainCount low-confidence';
    }
    return '$header\n$breakdown';
  }
}
