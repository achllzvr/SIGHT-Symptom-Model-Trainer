// File: lib/services/ai_service.dart

import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart';
import 'image_utils.dart'; // Import the helpers file we created above

class EyeDiagnosisService {
  late Interpreter _interpreter;
  late FaceDetector _faceDetector;
  late List<String> _labels;
  bool _isBusy = false;

  Future<void> initialize() async {
    // 1. Load TFLite Model
    try {
      _interpreter = await Interpreter.fromAsset('assets/sight_model_quant.tflite');
      print("TFLite Model Loaded Successfully");
    } catch (e) {
      print("Error loading model: $e");
    }

    // 2. Load Labels
    final labelData = await rootBundle.loadString('assets/labels.txt');
    _labels = labelData.split('\n');

    // 3. Initialize ML Kit Face Detector
    _faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        enableLandmarks: true, // CRITICAL: Need eye positions
        performanceMode: FaceDetectorMode.accurate,
      ),
    );
  }

  // MAIN FUNCTION: Analyzes a single camera frame
  Future<Map<String, String>> analyzeFrame(CameraImage cameraImage) async {
    if (_isBusy) return {}; 
    _isBusy = true;

    Map<String, String> results = {};

    try {
      // Step A: Convert to InputImage for ML Kit
      // (Note: conversion depends on platform, simplified here for brevity)
      // For accurate rotation, you need standard camera helper code.
      // Let's assume we find a face.
      
      // Since we can't easily process YUV in ML Kit without specific rotation logic,
      // We will focus on the TFLite part which is the hard part.
      // Assume 'inputImage' is valid.
      
      // Step B: Convert YUV to RGB for cropping (Heavy operation)
      img.Image fullImage = convertYUV420ToImage(cameraImage);

      // Step C: Run ML Kit to find eyes
      // For this raw guide, we need the InputImage. 
      // In a real app, use the 'camera' package's specific InputImage creation method.
      // We will simulate finding landmarks for the logic flow:
      
      // --- LOGIC FLOW ONLY (Paste inside your FaceDetector listener) ---
      /* final faces = await _faceDetector.processImage(mlKitInputImage);
      if (faces.isNotEmpty) {
         final face = faces.first;
         
         // Predict Left
         results['Left'] = _predictEye(fullImage, face.landmarks[FaceLandmarkType.leftEye]);
         
         // Predict Right
         results['Right'] = _predictEye(fullImage, face.landmarks[FaceLandmarkType.rightEye]);
      }
      */
      
    } catch (e) {
      print("Error: $e");
    } finally {
      _isBusy = false;
    }

    return results;
  }

  // The Cropping & Prediction Logic
  String _predictEye(img.Image fullImage, FaceLandmark? landmark) {
    if (landmark == null) return "No Eye Visible";

    // 1. Calculate Crop Box
    int size = 100; // 100x100 box
    int x = landmark.position.x.toInt() - (size ~/ 2);
    int y = landmark.position.y.toInt() - (size ~/ 2);
    
    // Bounds check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x + size > fullImage.width) x = fullImage.width - size;
    if (y + size > fullImage.height) y = fullImage.height - size;

    // 2. Crop
    img.Image eyeCrop = img.copyCrop(fullImage, x: x, y: y, width: size, height: size);
    
    // 3. Resize to 224x224 (Model Input)
    img.Image resized = img.copyResize(eyeCrop, width: 224, height: 224);

    // 4. Prepare Tensor Input
    var input = imageToByteListUint8(resized, 224);
    
    // 5. Output Tensor (1 Row, 4 Columns for 4 diseases)
    var output = List.filled(1 * 4, 0).reshape([1, 4]);

    // 6. Run Inference
    _interpreter.run(input, output);

    // 7. Interpret Results (Find highest probability)
    List<dynamic> probs = output[0]; // e.g. [10, 240, 5, 0]
    int maxIndex = 0;
    int maxVal = 0;

    for (int i = 0; i < probs.length; i++) {
      if (probs[i] > maxVal) {
        maxVal = probs[i];
        maxIndex = i;
      }
    }

    // Convert 0-255 score to Percentage
    String disease = _labels.length > maxIndex ? _labels[maxIndex] : "Unknown";
    int percentage = ((maxVal / 255) * 100).toInt();

    return "$disease ($percentage%)";
  }

  void dispose() {
    _interpreter.close();
    _faceDetector.close();
  }
}