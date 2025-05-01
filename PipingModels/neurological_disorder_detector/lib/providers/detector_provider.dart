import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:camera/camera.dart';
import 'package:path_provider/path_provider.dart';
import 'package:image_picker/image_picker.dart';
import '../models/detection_result.dart';
import '../services/api_service.dart';

class DetectorProvider extends ChangeNotifier {
  DetectionResult? _result;
  bool _isProcessing = false;
  bool _isApiAvailable = false;
  String _errorMessage = '';
  File? _imageFile;

  // Getters
  DetectionResult? get result => _result;
  bool get isProcessing => _isProcessing;
  bool get isApiAvailable => _isApiAvailable;
  String get errorMessage => _errorMessage;
  File? get imageFile => _imageFile;

  // Constructor checks API availability
  DetectorProvider() {
    _checkApiAvailability();
  }

  // Check if API is available
  Future<void> _checkApiAvailability() async {
    _isApiAvailable = await ApiService.checkApiHealth();
    notifyListeners();
  }

  // Process image from camera
  Future<void> processImageFromCamera(XFile file) async {
    _setProcessing(true);
    
    try {
      // Convert XFile to File
      final File imageFile = File(file.path);
      _imageFile = imageFile;
      
      // Process the image
      await _processImage(imageFile);
    } catch (e) {
      _setError('Failed to process camera image: $e');
    } finally {
      _setProcessing(false);
    }
  }

  // Process image from gallery
  Future<void> pickAndProcessImage() async {
    final ImagePicker picker = ImagePicker();
    
    try {
      final XFile? pickedFile = await picker.pickImage(
        source: ImageSource.gallery,
      );
      
      if (pickedFile == null) {
        return; // User canceled the picker
      }
      
      _setProcessing(true);
      final File imageFile = File(pickedFile.path);
      _imageFile = imageFile;
      
      await _processImage(imageFile);
    } catch (e) {
      _setError('Failed to pick or process image: $e');
    } finally {
      _setProcessing(false);
    }
  }

  // Process image from camera frame
  Future<void> processFrame(CameraImage cameraImage, CameraDescription camera) async {
    if (_isProcessing) return; // Prevent multiple simultaneous processing
    
    _setProcessing(true);
    
    try {
      // Convert camera image to a format we can use
      final File? imageFile = await _convertCameraImageToFile(cameraImage);
      if (imageFile == null) {
        _setError('Failed to convert camera image');
        return;
      }
      
      _imageFile = imageFile;
      await _processImage(imageFile);
    } catch (e) {
      _setError('Failed to process camera frame: $e');
    } finally {
      _setProcessing(false);
    }
  }

  // Main image processing method
  Future<void> _processImage(File imageFile) async {
    if (!_isApiAvailable) {
      await _checkApiAvailability();
      if (!_isApiAvailable) {
        _setError('API is not available. Please check your connection.');
        return;
      }
    }
    
    try {
      final bytes = await imageFile.readAsBytes();
      final result = await ApiService.detectDisorders(bytes);
      
      if (result == null) {
        _setError('Failed to get detection results');
        return;
      }
      
      // Set result and clear any previous errors
      _result = result;
      _errorMessage = '';
      notifyListeners();
    } catch (e) {
      _setError('Error during image processing: $e');
    }
  }

  // Helper to convert camera image to file
  Future<File?> _convertCameraImageToFile(CameraImage cameraImage) async {
    try {
      // This conversion is simplified and may need to be adjusted based on your camera format
      // Convert YUV420 to RGB (common for Android cameras)
      img.Image? image;
      
      if (cameraImage.format.group == ImageFormatGroup.yuv420) {
        image = _convertYUV420ToImage(cameraImage);
      } else if (cameraImage.format.group == ImageFormatGroup.bgra8888) {
        image = _convertBGRA8888ToImage(cameraImage);
      }
      
      if (image == null) {
        return null;
      }
      
      // Create temporary file
      final tempDir = await getTemporaryDirectory();
      final tempPath = '${tempDir.path}/temp_frame.jpg';
      final File file = File(tempPath);
      
      // Encode as JPEG and save
      final jpegBytes = img.encodeJpg(image, quality: 85);
      await file.writeAsBytes(jpegBytes);
      
      return file;
    } catch (e) {
      debugPrint('Error converting camera image: $e');
      return null;
    }
  }

  // Convert YUV420 format to Image
  img.Image? _convertYUV420ToImage(CameraImage cameraImage) {
    // This is a simplified implementation
    // A complete implementation would need to handle the YUV to RGB conversion properly
    // For accurate conversion, consider using platform-specific code or a dedicated package
    
    final width = cameraImage.width;
    final height = cameraImage.height;
    
    // Create a new image
    final image = img.Image(width: width, height: height);
    
    // We're simplifying by just using the Y plane for grayscale
    // A real implementation would convert YUV to RGB properly
    final yPlane = cameraImage.planes[0].bytes;
    
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final index = y * width + x;
        final yValue = yPlane[index];
        // Set grayscale value for all RGB channels
        image.setPixel(x, y, img.ColorRgb8(yValue, yValue, yValue));
      }
    }
    
    return image;
  }

  // Convert BGRA8888 format to Image
  img.Image _convertBGRA8888ToImage(CameraImage cameraImage) {
    final bytes = cameraImage.planes[0].bytes;
    return img.Image.fromBytes(
      width: cameraImage.width,
      height: cameraImage.height,
      bytes: bytes.buffer,
      numChannels: 4,
    );
  }

  // Helper to set processing state
  void _setProcessing(bool isProcessing) {
    _isProcessing = isProcessing;
    notifyListeners();
  }

  // Helper to set error message
  void _setError(String message) {
    _errorMessage = message;
    notifyListeners();
  }

  // Clear current results
  void clearResults() {
    _result = null;
    _imageFile = null;
    _errorMessage = '';
    notifyListeners();
  }
}