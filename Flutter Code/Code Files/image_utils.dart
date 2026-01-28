import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'dart:typed_data';

/// Converts a [CameraImage] in YUV420 format to [img.Image] in RGB format
img.Image convertYUV420ToImage(CameraImage cameraImage) {
  final int width = cameraImage.width;
  final int height = cameraImage.height;

  final int uvRowStride = cameraImage.planes[1].bytesPerRow;
  final int uvPixelStride = cameraImage.planes[1].bytesPerPixel!;

  // Create Image buffer
  var image = img.Image(width: width, height: height);

  for (int w = 0; w < width; w++) {
    for (int h = 0; h < height; h++) {
      final int uvIndex =
          uvPixelStride * (w / 2).floor() + uvRowStride * (h / 2).floor();
      final int index = h * width + w;

      final y = cameraImage.planes[0].bytes[index];
      final u = cameraImage.planes[1].bytes[uvIndex];
      final v = cameraImage.planes[2].bytes[uvIndex];

      // Convert YUV to RGB
      int r = (y + v * 1.402 - 0.701 * 255).toInt();
      int g = (y - u * 0.34414 - v * 0.71414 + 0.529 * 255).toInt();
      int b = (y + u * 1.772 - 0.886 * 255).toInt();

      // Clamp to 0-255
      r = r.clamp(0, 255);
      g = g.clamp(0, 255);
      b = b.clamp(0, 255);

      // Set pixel (img package uses different setPixel based on version)
      // For image ^4.0.0:
      image.setPixelRgb(w, h, r, g, b);
    }
  }
  return image;
}

/// Converts an [img.Image] to the specific List<int> the TFLite model needs
/// For Quantized models, this returns values 0-255 (Uint8)
Uint8List imageToByteListUint8(img.Image image, int inputSize) {
  var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
  var buffer = Float32List.view(convertedBytes.buffer);
  int pixelIndex = 0;

  for (var i = 0; i < inputSize; i++) {
    for (var j = 0; j < inputSize; j++) {
      var pixel = image.getPixel(j, i);
      // Extract RGB channels
      buffer[pixelIndex++] = pixel.r.toDouble(); // Red
      buffer[pixelIndex++] = pixel.g.toDouble(); // Green
      buffer[pixelIndex++] = pixel.b.toDouble(); // Blue
    }
  }
  
  // Cast to Uint8 for Quantized TFLite
  return Float32List.fromList(buffer).buffer.asUint8List();
}