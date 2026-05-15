import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

late List<CameraDescription> _cameras;

/// In `flutter test`, call before `pumpWidget` so the camera plugin is not required.
void debugSetCamerasForTest(List<CameraDescription> cameras) {
  _cameras = cameras;
}

const String _modelAssetName = 'best_float32.tflite';
const String _labelsAssetName = 'labels.txt';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  _cameras = await availableCameras();
  runApp(const TrashTrackerApp());
}

class TrashTrackerApp extends StatelessWidget {
  const TrashTrackerApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.green),
      home: const CameraScreen(),
    );
  }
}

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});
  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? controller;
  Interpreter? _interpreter;
  List<String> _labels = [];
  List<int> _inputShape = const [1, 640, 640, 3];
  String _modelDebug = '';
  bool _isProcessing = false;
  bool _isModelReady = false;
  String _resultText = "Ready to scan";

  @override
  void initState() {
    super.initState();
    _initializeAll();
  }

  Future<void> _initializeAll() async {
    await _initializeModel();
    await _initializeCamera();
  }

  Future<void> _initializeModel() async {
    try {
      final modelPath = await _copyAssetToLocal(_modelAssetName);
      final labelsPath = await _copyAssetToLocal(_labelsAssetName);
      _labels = await File(labelsPath).readAsLines();

      _interpreter = Interpreter.fromFile(File(modelPath));
      _inputShape = _interpreter!.getInputTensor(0).shape;
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      _modelDebug = 'in=$_inputShape out=$outputShape';
      _isModelReady = true;
      debugPrint("Model initialized: $_modelDebug");
    } catch (e, stackTrace) {
      debugPrint("Model init error: $e");
      debugPrint("$stackTrace");
      if (mounted) setState(() => _resultText = "Model Error: $e");
    }
  }

  Future<void> _initializeCamera() async {
    if (_cameras.isEmpty) return;

    controller = CameraController(
      _cameras[0],
      ResolutionPreset.medium, // Medium is safer for memory on A-series phones
      enableAudio: false,
    );

    try {
      await controller!.initialize();
      if (mounted) setState(() {});
    } catch (e) {
      setState(() => _resultText = "Camera Error: $e");
    }
  }

  Future<String> _copyAssetToLocal(String assetName) async {
    final directory = await getApplicationSupportDirectory();
    final path = join(directory.path, assetName);
    final file = File(path);

    // To be safe, we always re-copy the asset during development
    // to ensure the file isn't corrupted or 0-bytes
    final data = await rootBundle.load('assets/$assetName');
    final bytes = data.buffer.asUint8List(
      data.offsetInBytes,
      data.lengthInBytes,
    );

    await Directory(dirname(path)).create(recursive: true);
    await file.writeAsBytes(bytes, flush: true);

    return file.path;
  }

  Future<void> _captureAndDetect() async {
    if (controller == null ||
        !controller!.value.isInitialized ||
        !_isModelReady ||
        _isProcessing) {
      debugPrint(
        "Capture skipped: Model/camera not ready or already processing.",
      );
      return;
    }

    if (mounted) {
      setState(() {
        _isProcessing = true;
        _resultText = "Analyzing...";
      });
    }

    try {
      // 1. Capture the photo
      final XFile photo = await controller!.takePicture();
      debugPrint("Photo captured: ${photo.path}");

      // 2. Build model input tensor from captured image
      final input = await _buildInputTensor(photo.path);

      // 3. Run TFLite inference
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      final output = _createOutputBuffer(outputShape);
      _interpreter!.run(input, output);

      // 4. Placeholder parse for YOLO output (replace with full decode+NMS next)
      final summary = _summarizeRawOutput(output);
      if (mounted) setState(() => _resultText = summary);

      // Delete the temp photo to save space
      await File(photo.path).delete();
      debugPrint("Temporary photo deleted.");
    } catch (e, stackTrace) {
      debugPrint("Error during detection: $e");
      debugPrint("Stack trace: $stackTrace");
      if (mounted) setState(() => _resultText = "Processing Error: $e");
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  @override
  void dispose() {
    controller?.dispose();
    _interpreter?.close();
    super.dispose();
  }

  /// Builds a shaped input tensor object for common YOLO TFLite layouts:
  /// - NHWC: `[1, H, W, 3]`
  /// - NCHW: `[1, 3, H, W]`
  Future<Object> _buildInputTensor(String imagePath) async {
    final bytes = await File(imagePath).readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw StateError('Unable to decode captured image.');
    }

    final (height, width, layout) = _resolveInputLayout(_inputShape);
    final resized = img.copyResize(decoded, width: width, height: height);
    if (layout == _InputLayout.nhwc) {
      return [
        List.generate(height, (y) {
          return List.generate(width, (x) {
            final pixel = resized.getPixel(x, y);
            return [
              pixel.r / 255.0,
              pixel.g / 255.0,
              pixel.b / 255.0,
            ];
          });
        }),
      ];
    }

    // NCHW
    final rChannel = List.generate(
      height,
      (_) => List<double>.filled(width, 0.0),
    );
    final gChannel = List.generate(
      height,
      (_) => List<double>.filled(width, 0.0),
    );
    final bChannel = List.generate(
      height,
      (_) => List<double>.filled(width, 0.0),
    );

    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final pixel = resized.getPixel(x, y);
        rChannel[y][x] = pixel.r / 255.0;
        gChannel[y][x] = pixel.g / 255.0;
        bChannel[y][x] = pixel.b / 255.0;
      }
    }

    return [
      [rChannel, gChannel, bChannel],
    ];
  }

  /// Resolves spatial size and memory layout from the model's input tensor shape.
  (int height, int width, _InputLayout layout) _resolveInputLayout(
    List<int> shape,
  ) {
    if (shape.length == 4) {
      final a = shape[1];
      final b = shape[2];
      final c = shape[3];
      if (a == 3 && c != 3) {
        return (b, c, _InputLayout.nchw);
      }
      if (c == 3) {
        return (a, b, _InputLayout.nhwc);
      }
    }
    return (640, 640, _InputLayout.nhwc);
  }

  dynamic _createOutputBuffer(List<int> shape) {
    if (shape.isEmpty) return 0.0;
    if (shape.length == 1) {
      return List<double>.filled(max(shape[0], 1), 0.0);
    }
    return List.generate(
      shape[0],
      (_) => _createOutputBuffer(shape.sublist(1)),
    );
  }

  String _summarizeRawOutput(dynamic output) {
    final values = _flatten(output);
    if (values.isEmpty) {
      return "Inference finished, but output was empty.";
    }

    final top = values.reduce(max);
    final topPercent = (top * 100).clamp(0, 100).toStringAsFixed(1);
    final labelInfo = _labels.isEmpty ? "labels unavailable" : "${_labels.length} labels loaded";
    return "Inference OK ($labelInfo)\nTop raw score: $topPercent%\n$_modelDebug";
  }

  List<double> _flatten(dynamic value) {
    if (value is double) return [value];
    if (value is int) return [value.toDouble()];
    if (value is List) {
      final flattened = <double>[];
      for (final element in value) {
        flattened.addAll(_flatten(element));
      }
      return flattened;
    }
    return const [];
  }

  @override
  Widget build(BuildContext context) {
    if (controller == null || !controller!.value.isInitialized) {
      return Scaffold(
        appBar: AppBar(title: const Text("Trash Tracker")),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text("Trash Tracker"),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => setState(() => _resultText = "Ready to scan"),
          ),
        ],
      ),
      body: Stack(
        children: [
          // Using an AspectRatio to prevent preview stretching
          ClipRect(
            child: AspectRatio(
              aspectRatio: controller!.value.aspectRatio,
              child: CameraPreview(controller!),
            ),
          ),

          // Result Box
          Positioned(
            top: 20,
            left: 20,
            right: 20,
            child: Container(
              padding: const EdgeInsets.all(15),
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.7),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                _resultText,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),

          // Capture Button
          Positioned(
            bottom: 50,
            left: 0,
            right: 0,
            child: Center(
              child: GestureDetector(
                onTap: _isProcessing ? null : _captureAndDetect,
                child: Container(
                  height: 80,
                  width: 80,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white, width: 4),
                    color: _isProcessing
                        ? Colors.grey
                        : Colors.green.withValues(alpha: 0.8),
                  ),
                  child: _isProcessing
                      ? const Padding(
                          padding: EdgeInsets.all(20.0),
                          child: CircularProgressIndicator(color: Colors.white),
                        )
                      : const Icon(
                          Icons.camera_alt,
                          color: Colors.white,
                          size: 40,
                        ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

enum _InputLayout { nhwc, nchw }
