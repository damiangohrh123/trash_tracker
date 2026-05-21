import 'dart:io';
import 'dart:math';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'detection_parser.dart';

late List<CameraDescription> _cameras;

const _modelAssetName = 'best_float32.tflite';
const _labelsAssetName = 'labels.txt';

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
  CameraController? _controller;
  Interpreter? _interpreter;
  List<String> _labels = [];
  int _inputHeight = 800;
  int _inputWidth = 800;
  bool _isProcessing = false;
  bool _isModelReady = false;
  String _resultText = 'Ready to scan';

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
      _labels = (await File(labelsPath).readAsLines())
          .map((line) => line.trim())
          .where((line) => line.isNotEmpty)
          .toList();

      _interpreter = Interpreter.fromFile(File(modelPath));
      final inputShape = _interpreter!.getInputTensor(0).shape;
      if (inputShape.length == 4 && inputShape[3] == 3) {
        _inputHeight = inputShape[1];
        _inputWidth = inputShape[2];
      }

      _isModelReady = true;
      debugPrint('Model ready: ${inputShape.join('x')}');
    } catch (e, stackTrace) {
      debugPrint('Model init error: $e\n$stackTrace');
      if (mounted) setState(() => _resultText = 'Model Error: $e');
    }
  }

  Future<void> _initializeCamera() async {
    if (_cameras.isEmpty) return;

    _controller = CameraController(
      _cameras[0],
      ResolutionPreset.medium,
      enableAudio: false,
    );

    try {
      await _controller!.initialize();
      if (mounted) setState(() {});
    } catch (e) {
      if (mounted) setState(() => _resultText = 'Camera Error: $e');
    }
  }

  Future<String> _copyAssetToLocal(String assetName) async {
    final directory = await getApplicationSupportDirectory();
    final filePath = join(directory.path, assetName);
    final data = await rootBundle.load('assets/$assetName');
    final bytes = data.buffer.asUint8List(
      data.offsetInBytes,
      data.lengthInBytes,
    );

    await Directory(dirname(filePath)).create(recursive: true);
    await File(filePath).writeAsBytes(bytes, flush: true);
    return filePath;
  }

  Future<void> _captureAndDetect() async {
    if (_controller == null ||
        !_controller!.value.isInitialized ||
        !_isModelReady ||
        _isProcessing) {
      return;
    }

    if (mounted) {
      setState(() {
        _isProcessing = true;
        _resultText = 'Analyzing...';
      });
    }

    XFile? photo;
    try {
      photo = await _controller!.takePicture();
      final input = await _buildInputTensor(photo.path);
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      final output = _createOutputBuffer(outputShape);
      _interpreter!.run(input, output);

      final summary = summarizeDetections(output, _labels);
      if (mounted) setState(() => _resultText = summary.toDisplayText());
    } catch (e, stackTrace) {
      debugPrint('Error during detection: $e\n$stackTrace');
      if (mounted) setState(() => _resultText = 'Processing Error: $e');
    } finally {
      if (photo != null) {
        try {
          await File(photo.path).delete();
        } catch (deleteError) {
          debugPrint('Temp photo cleanup failed: $deleteError');
        }
      }
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  Future<List<List<List<List<double>>>>> _buildInputTensor(
    String imagePath,
  ) async {
    final bytes = await File(imagePath).readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw StateError('Unable to decode captured image.');
    }

    final resized = img.copyResize(
      decoded,
      width: _inputWidth,
      height: _inputHeight,
    );

    return [
      List.generate(_inputHeight, (y) {
        return List.generate(_inputWidth, (x) {
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

  void _resetResult() {
    setState(() => _resultText = 'Ready to scan');
  }

  @override
  void dispose() {
    _controller?.dispose();
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return Scaffold(
        appBar: AppBar(title: const Text('Trash Tracker')),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Trash Tracker'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _resetResult,
          ),
        ],
      ),
      body: Stack(
        children: [
          ClipRect(
            child: AspectRatio(
              aspectRatio: _controller!.value.aspectRatio,
              child: CameraPreview(_controller!),
            ),
          ),
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
                          padding: EdgeInsets.all(20),
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
