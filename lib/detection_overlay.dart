import 'package:flutter/material.dart';

import 'detection_parser.dart';

class DetectionOverlayPainter extends CustomPainter {
  DetectionOverlayPainter(this.detections);

  final List<Detection> detections;

  @override
  void paint(Canvas canvas, Size size) {
    final confirmedPaint = Paint()
      ..color = Colors.greenAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5;

    final uncertainPaint = Paint()
      ..color = Colors.amberAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    final fillPaint = Paint()
      ..color = Colors.black.withValues(alpha: 0.65);

    for (final detection in detections) {
      final rect = Rect.fromLTRB(
        detection.x1 * size.width,
        detection.y1 * size.height,
        detection.x2 * size.width,
        detection.y2 * size.height,
      );

      final boxPaint =
          detection.isConfirmed ? confirmedPaint : uncertainPaint;
      canvas.drawRect(rect, boxPaint);

      final labelPrefix = detection.isConfirmed ? '' : '? ';
      final label =
          '$labelPrefix${detection.label} ${(detection.confidence * 100).toStringAsFixed(0)}%';
      final textPainter = TextPainter(
        text: TextSpan(
          text: label,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 12,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final labelWidth = textPainter.width + 8;
      final labelHeight = textPainter.height + 4;
      final labelTop = rect.top - labelHeight < 0
          ? rect.top
          : rect.top - labelHeight;

      final labelRect = Rect.fromLTWH(
        rect.left,
        labelTop,
        labelWidth,
        labelHeight,
      );

      canvas.drawRect(labelRect, fillPaint);
      textPainter.paint(canvas, Offset(rect.left + 4, labelTop + 2));
    }
  }

  @override
  bool shouldRepaint(covariant DetectionOverlayPainter oldDelegate) {
    return oldDelegate.detections != detections;
  }
}
