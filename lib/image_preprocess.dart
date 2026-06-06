import 'dart:math';

import 'package:image/image.dart' as img;

/// Ultralytics-style letterbox metadata for mapping model boxes back to the
/// original photo coordinates.
class LetterboxParams {
  const LetterboxParams({
    required this.scale,
    required this.padX,
    required this.padY,
    required this.originalWidth,
    required this.originalHeight,
    required this.inputWidth,
    required this.inputHeight,
  });

  final double scale;
  final int padX;
  final int padY;
  final int originalWidth;
  final int originalHeight;
  final int inputWidth;
  final int inputHeight;
}

class LetterboxResult {
  const LetterboxResult(this.image, this.params);

  final img.Image image;
  final LetterboxParams params;
}

LetterboxResult letterboxImage(
  img.Image source,
  int targetWidth,
  int targetHeight,
) {
  final scale = min(
    targetWidth / source.width,
    targetHeight / source.height,
  );
  final resizedWidth = (source.width * scale).round();
  final resizedHeight = (source.height * scale).round();
  final resized = img.copyResize(
    source,
    width: resizedWidth,
    height: resizedHeight,
  );

  final canvas = img.Image(width: targetWidth, height: targetHeight);
  img.fill(canvas, color: img.ColorRgb8(114, 114, 114));

  final padX = (targetWidth - resizedWidth) ~/ 2;
  final padY = (targetHeight - resizedHeight) ~/ 2;
  img.compositeImage(canvas, resized, dstX: padX, dstY: padY);

  return LetterboxResult(
    canvas,
    LetterboxParams(
      scale: scale,
      padX: padX,
      padY: padY,
      originalWidth: source.width,
      originalHeight: source.height,
      inputWidth: targetWidth,
      inputHeight: targetHeight,
    ),
  );
}

List<List<List<List<double>>>> imageToInputTensor(img.Image letterboxed) {
  final height = letterboxed.height;
  final width = letterboxed.width;

  return [
    List.generate(height, (y) {
      return List.generate(width, (x) {
        final pixel = letterboxed.getPixel(x, y);
        return [
          pixel.r / 255.0,
          pixel.g / 255.0,
          pixel.b / 255.0,
        ];
      });
    }),
  ];
}
