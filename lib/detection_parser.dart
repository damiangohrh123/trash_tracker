const double confidenceThreshold = 0.25;

/// Parses Ultralytics YOLO TFLite output shaped `[1, N, 6]`.
/// Each row is: x1, y1, x2, y2, confidence, classId (normalized xyxy).
DetectionSummary summarizeDetections(
  dynamic output,
  List<String> labels,
) {
  if (output is! List || output.isEmpty) {
    return const DetectionSummary.empty();
  }

  final batch = output.first;
  if (batch is! List) {
    return const DetectionSummary.empty();
  }

  var count = 0;
  var bestConfidence = 0.0;
  var bestClassId = -1;

  for (final row in batch) {
    if (row is! List || row.length < 6) continue;

    final x1 = _asDouble(row[0]);
    final y1 = _asDouble(row[1]);
    final x2 = _asDouble(row[2]);
    final y2 = _asDouble(row[3]);
    final confidence = _asDouble(row[4]);
    final classId = _asDouble(row[5]).round();

    if (confidence < confidenceThreshold) continue;
    if (x2 <= x1 || y2 <= y1) continue;
    if (classId < 0 || classId >= labels.length) continue;

    count++;
    if (confidence > bestConfidence) {
      bestConfidence = confidence;
      bestClassId = classId;
    }
  }

  if (count == 0) {
    return const DetectionSummary.empty();
  }

  return DetectionSummary(
    topLabel: labels[bestClassId],
    topConfidence: bestConfidence,
    detectionCount: count,
  );
}

double _asDouble(dynamic value) {
  if (value is num) return value.toDouble();
  return 0.0;
}

class DetectionSummary {
  const DetectionSummary({
    required this.topLabel,
    required this.topConfidence,
    required this.detectionCount,
  });

  const DetectionSummary.empty()
      : topLabel = null,
        topConfidence = 0,
        detectionCount = 0;

  final String? topLabel;
  final double topConfidence;
  final int detectionCount;

  bool get hasDetections => detectionCount > 0;

  String toDisplayText() {
    if (!hasDetections || topLabel == null) {
      return 'No trash detected.';
    }

    final percent = (topConfidence * 100).toStringAsFixed(1);
    return '$topLabel ($percent%)\nDetections: $detectionCount';
  }
}
