// lib/services/api_service.dart
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import '../models/detection_result.dart';

class ApiService {
  static final String baseUrl = 'https://191e-129-110-241-55.ngrok-free.app';


  static void _log(String message) {
    debugPrint('API SERVICE: $message');
  }

  static Future<bool> checkApiHealth() async {
    _log('Checking Flask API health...');
    try {
      final response = await http.get(Uri.parse('$baseUrl/health'));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['status'] == 'healthy';
      }
    } catch (e) {
      _log('Health check failed: $e');
    }
    return false;
  }


  static Future<DetectionResult?> detectDisorders(Uint8List imageBytes) async {
    try {
      final String base64Image = base64Encode(imageBytes);
      final response = await http.post(
        Uri.parse('$baseUrl/detect'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'image': base64Image}),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return DetectionResult.fromJson(data);
      } else {
        _log('Detection failed with status: ${response.statusCode}');
      }
    } catch (e) {
      _log('Detection failed: $e');
    }

    return null;
  }

}
