// Add this to your camera_screen.dart
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';
import 'package:image_picker/image_picker.dart';
import '../providers/detector_provider.dart';
import 'results_screen.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({Key? key}) : super(key: key);

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  bool _isLoading = true;
  String _errorMessage = '';
  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _checkPermissions();
  }

  Future<void> _checkPermissions() async {
    try {
      final status = await Permission.camera.status;
      
      if (status.isPermanentlyDenied) {
        setState(() {
          _isLoading = false;
          _errorMessage = 'Camera permission permanently denied. Please enable in settings.';
        });
      } else {
        // For any other status, just try to use the camera
        setState(() {
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
        _errorMessage = 'Error checking permissions: $e';
      });
    }
  }

  Future<void> _takePicture() async {
    try {
      // Try to pick an image from camera
      final XFile? photo = await _picker.pickImage(source: ImageSource.camera);
      
      if (photo != null) {
        // Process the image
        final detectorProvider = Provider.of<DetectorProvider>(context, listen: false);
        await detectorProvider.processImageFromCamera(photo);
        
        // Navigate to results
        if (!mounted) return;
        Navigator.push(
          context, 
          MaterialPageRoute(builder: (context) => const ResultsScreen())
        );
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Error taking picture: $e';
      });
    }
  }

  Future<void> _pickFromGallery() async {
    try {
      final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
      
      if (image != null) {
        final detectorProvider = Provider.of<DetectorProvider>(context, listen: false);
        await detectorProvider.processImageFromCamera(image);
        
        if (!mounted) return;
        Navigator.push(
          context, 
          MaterialPageRoute(builder: (context) => const ResultsScreen())
        );
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Error picking image: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Capture Image'),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _errorMessage.isNotEmpty
              ? _buildErrorView()
              : _buildCameraOptions(),
    );
  }

  Widget _buildErrorView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, color: Colors.red, size: 60),
            const SizedBox(height: 16),
            Text(
              _errorMessage,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.red),
            ),
            const SizedBox(height: 24),
            ElevatedButton(
              onPressed: openAppSettings,
              child: const Text('Open Settings'),
            ),
            const SizedBox(height: 12),
            TextButton(
              onPressed: _pickFromGallery,
              child: const Text('Use Gallery Instead'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCameraOptions() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(
            Icons.camera_alt,
            size: 80,
            color: Colors.blue,
          ),
          const SizedBox(height: 24),
          const Text(
            'Take a picture for analysis',
            style: TextStyle(fontSize: 18),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 32),
          ElevatedButton.icon(
            icon: const Icon(Icons.camera_alt),
            label: const Text('Open Camera'),
            onPressed: _takePicture,
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.symmetric(
                horizontal: 24,
                vertical: 12,
              ),
            ),
          ),
          const SizedBox(height: 16),
          TextButton.icon(
            icon: const Icon(Icons.photo_library),
            label: const Text('Select from Gallery'),
            onPressed: _pickFromGallery,
          ),
        ],
      ),
    );
  }
}