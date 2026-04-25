import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_object_detection/google_mlkit_object_detection.dart';
import 'package:flutter/services.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';

late List<CameraDescription> _cameras;

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
  ObjectDetector? _objectDetector;
  bool _isBusy = false;
  String _resultText = "Initializing AI...";

  @override
  void initState() {
    super.initState();
    _setupEverything();
  }

  /// Sequential setup to prevent race-condition crashes
  Future<void> _setupEverything() async {
    await _initializeDetector();
    await _initializeCamera();
  }

  Future<void> _initializeDetector() async {
    try {
      final modelPath = await _getModelPath('assets/best_float32.tflite');

      final options = LocalObjectDetectorOptions(
        mode: DetectionMode.stream,
        modelPath: modelPath,
        classifyObjects: true,
        multipleObjects: true,
      );

      _objectDetector = ObjectDetector(options: options);
      setState(() => _resultText = "AI Ready. Starting Camera...");
    } catch (e) {
      setState(() => _resultText = "Model Load Error: $e");
    }
  }

  Future<void> _initializeCamera() async {
    if (_cameras.isEmpty) return;

    controller = CameraController(
      _cameras[0],
      ResolutionPreset.medium,
      enableAudio: false,
    );

    try {
      await controller!.initialize();
      if (!mounted) return;

      // Start the stream only after controller and detector are ready
      await controller!.startImageStream(_processCameraImage);
      setState(() => _resultText = "Scanning for trash...");
    } catch (e) {
      setState(() => _resultText = "Camera Error: $e");
    }
  }

  void _processCameraImage(CameraImage image) async {
    // Safety check: Don't process if busy, detector isn't ready, or UI is gone
    if (_objectDetector == null || _isBusy || !mounted) return;

    _isBusy = true;

    try {
      // 1. Efficiently combine image planes
      final WriteBuffer allBytes = WriteBuffer();
      for (final Plane plane in image.planes) {
        allBytes.putUint8List(plane.bytes);
      }
      final bytes = allBytes.done().buffer.asUint8List();

      // 2. Build metadata using the device's native format
      final imageInput = InputImage.fromBytes(
        bytes: bytes,
        metadata: InputImageMetadata(
          size: Size(image.width.toDouble(), image.height.toDouble()),
          rotation: InputImageRotation.rotation90deg,
          format:
              InputImageFormatValue.fromRawValue(image.format.raw) ??
              InputImageFormat.nv21,
          bytesPerRow: image.planes[0].bytesPerRow,
        ),
      );

      // 3. Inference
      final objects = await _objectDetector!.processImage(imageInput);

      if (objects.isNotEmpty && mounted) {
        final firstObj = objects.first;
        if (firstObj.labels.isNotEmpty) {
          final label = firstObj.labels.first;
          setState(() {
            _resultText =
                "${label.text} (${(label.confidence * 100).toStringAsFixed(0)}%)";
          });
        } else {
          setState(() => _resultText = "Object detected (unlabeled)");
        }
      }
    } catch (e) {
      debugPrint("AI Processing Error: $e");
    } finally {
      _isBusy = false;
    }
  }

  Future<String> _getModelPath(String asset) async {
    final path = join((await getApplicationSupportDirectory()).path, asset);
    final file = File(path);

    // Only copy if it doesn't exist to save startup time
    if (!await file.exists()) {
      await Directory(dirname(path)).create(recursive: true);
      final data = await rootBundle.load(asset);
      final bytes = data.buffer.asUint8List(
        data.offsetInBytes,
        data.lengthInBytes,
      );
      await file.writeAsBytes(bytes);
    }
    return path;
  }

  @override
  void dispose() {
    controller?.dispose();
    _objectDetector?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (controller == null || !controller!.value.isInitialized) {
      return Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const CircularProgressIndicator(),
              const SizedBox(height: 20),
              Text(_resultText),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text("Trash Tracker AI")),
      body: Stack(
        children: [
          CameraPreview(controller!),
          Align(
            alignment: Alignment.bottomCenter,
            child: Container(
              margin: const EdgeInsets.all(30),
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
              decoration: BoxDecoration(
                color: Colors.black87,
                borderRadius: BorderRadius.circular(15),
              ),
              child: Text(
                _resultText,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
