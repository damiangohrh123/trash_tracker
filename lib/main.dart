import 'dart:io';
import 'dart:math';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'detection_overlay.dart';
import 'detection_parser.dart';
import 'image_preprocess.dart';

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
  String _resultText = 'Point at waste items';

  String? _resultImagePath;
  double _resultAspectRatio = 3 / 4;
  List<Detection> _detections = const [];

  bool get _showingResult => _resultImagePath != null;

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
      ResolutionPreset.high,
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
        _isProcessing ||
        _showingResult) {
      return;
    }

    if (mounted) {
      setState(() {
        _isProcessing = true;
        _resultText = 'Analyzing...';
      });
    }

    try {
      final photo = await _controller!.takePicture();
      final decoded = await _decodeImage(photo.path);
      final letterbox = letterboxImage(decoded, _inputWidth, _inputHeight);
      final input = imageToInputTensor(letterbox.image);
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      final output = _createOutputBuffer(outputShape);
      _interpreter!.run(input, output);

      final mapping = LetterboxMapping(
        scale: letterbox.params.scale,
        padX: letterbox.params.padX.toDouble(),
        padY: letterbox.params.padY.toDouble(),
        originalWidth: letterbox.params.originalWidth,
        originalHeight: letterbox.params.originalHeight,
        inputWidth: letterbox.params.inputWidth,
        inputHeight: letterbox.params.inputHeight,
      );
      final modelDetections = parseDetections(output, _labels);
      final detections = mapDetectionsToOriginalImage(modelDetections, mapping);
      final summary = summarizeDetections(detections);

      await _controller!.pausePreview();

      if (mounted) {
        setState(() {
          _resultImagePath = photo.path;
          _resultAspectRatio = decoded.width / decoded.height;
          _detections = detections;
          _resultText = summary.toDisplayText();
        });
      }
    } catch (e, stackTrace) {
      debugPrint('Error during detection: $e\n$stackTrace');
      if (mounted) setState(() => _resultText = 'Processing Error: $e');
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  Future<void> _scanAgain() async {
    final previousPath = _resultImagePath;
    if (_controller != null && _controller!.value.isInitialized) {
      await _controller!.resumePreview();
    }

    if (previousPath != null) {
      try {
        await File(previousPath).delete();
      } catch (deleteError) {
        debugPrint('Result image cleanup failed: $deleteError');
      }
    }

    if (mounted) {
      setState(() {
        _resultImagePath = null;
        _detections = const [];
        _resultText = 'Point at waste items';
      });
    }
  }

  Future<img.Image> _decodeImage(String imagePath) async {
    final bytes = await File(imagePath).readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw StateError('Unable to decode captured image.');
    }
    return decoded;
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

  @override
  void dispose() {
    if (_resultImagePath != null) {
      File(_resultImagePath!).delete().ignore();
    }
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
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Trash Tracker'),
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          if (_showingResult)
            _buildResultView()
          else
            _buildLiveCameraView(),
          _buildResultBanner(),
          _buildBottomAction(),
        ],
      ),
    );
  }

  Widget _buildLiveCameraView() {
    return ClipRect(
      child: AspectRatio(
        aspectRatio: _controller!.value.aspectRatio,
        child: CameraPreview(_controller!),
      ),
    );
  }

  Widget _buildResultView() {
    return Center(
      child: AspectRatio(
        aspectRatio: _resultAspectRatio,
        child: Stack(
          fit: StackFit.expand,
          children: [
            Image.file(
              File(_resultImagePath!),
              fit: BoxFit.fill,
            ),
            CustomPaint(
              painter: DetectionOverlayPainter(_detections),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultBanner() {
    return Positioned(
      top: 12,
      left: 16,
      right: 16,
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.75),
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
    );
  }

  Widget _buildBottomAction() {
    return Positioned(
      bottom: 36,
      left: 0,
      right: 0,
      child: Center(
        child: _showingResult ? _buildScanAgainButton() : _buildCaptureButton(),
      ),
    );
  }

  Widget _buildCaptureButton() {
    return GestureDetector(
      onTap: _isProcessing ? null : _captureAndDetect,
      child: Container(
        height: 80,
        width: 80,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white, width: 4),
          color: _isProcessing
              ? Colors.grey
              : Colors.green.withValues(alpha: 0.85),
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
    );
  }

  Widget _buildScanAgainButton() {
    return FilledButton.icon(
      onPressed: _scanAgain,
      icon: const Icon(Icons.camera_front),
      label: const Text('Scan again'),
      style: FilledButton.styleFrom(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
      ),
    );
  }
}
