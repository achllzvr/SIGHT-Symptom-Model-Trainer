// 1. Get the face object from ML Kit
final face = await faceDetector.processImage(inputImage);

// 2. Define a map to store results
Map<String, List<double>> eyeResults = {};

// 3. Process Left Eye
if (face.landmarks[FaceLandmarkType.leftEye] != null) {
  final leftPos = face.landmarks[FaceLandmarkType.leftEye]!.position;
  // ... Code to Crop Left Eye ...
  // ... Code to Run AI ...
  eyeResults['Left Eye'] = output; // e.g., [Healthy%, Uveitis%, Cataract%]
}

// 4. Process Right Eye
if (face.landmarks[FaceLandmarkType.rightEye] != null) {
  final rightPos = face.landmarks[FaceLandmarkType.rightEye]!.position;
  // ... Code to Crop Right Eye ...
  // ... Code to Run AI ...
  eyeResults['Right Eye'] = output;
}