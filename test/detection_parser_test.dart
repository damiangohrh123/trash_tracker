import 'package:flutter_test/flutter_test.dart';
import 'package:trash_tracker/detection_parser.dart';

void main() {
  const labels = [
    'BIODEGRADABLE',
    'CARDBOARD',
    'GLASS',
    'METAL',
    'PAPER',
    'PLASTIC',
  ];

  test('summarizeDetections returns top label and count', () {
    final output = [
      [
        [0.1, 0.1, 0.2, 0.2, 0.3, 1.0],
        [0.55, 0.11, 0.66, 0.29, 0.87, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      ],
    ];

    final summary = summarizeDetections(output, labels);

    expect(summary.hasDetections, isTrue);
    expect(summary.topLabel, 'BIODEGRADABLE');
    expect(summary.topConfidence, closeTo(0.87, 0.001));
    expect(summary.detectionCount, 2);
    expect(summary.toDisplayText(), contains('BIODEGRADABLE'));
  });

  test('summarizeDetections returns empty summary when no valid rows', () {
    final output = [
      [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      ],
    ];

    final summary = summarizeDetections(output, labels);

    expect(summary.hasDetections, isFalse);
    expect(summary.toDisplayText(), 'No trash detected.');
  });
}
