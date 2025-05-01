import 'package:flutter/material.dart';

class DetectionResult {
  final bool faceDetected;
  final DisorderProbability pd; // Parkinson's Disease
  final DisorderProbability ms; // Multiple Sclerosis
  final DisorderProbability ad; // Alzheimer's Disease

  DetectionResult({
    required this.faceDetected,
    required this.pd,
    required this.ms,
    required this.ad,
  });

  factory DetectionResult.fromJson(Map<String, dynamic> json) {
    return DetectionResult(
      faceDetected: json['face_detected'] ?? false,
      pd: DisorderProbability.fromJson(json['pd'] ?? {}),
      ms: DisorderProbability.fromJson(json['ms'] ?? {}),
      ad: DisorderProbability.fromJson(json['ad'] ?? {}),
    );
  }

  String getHighestProbabilityDisorder() {
    if (!faceDetected) return 'No face detected';
    
    // Find disorder with highest probability
    final List<MapEntry<String, double>> probabilities = [
      MapEntry('Parkinson\'s Disease', pd.probability),
      MapEntry('Multiple Sclerosis', ms.probability),
      MapEntry('Alzheimer\'s Disease', ad.probability),
    ];
    
    probabilities.sort((a, b) => b.value.compareTo(a.value));
    return probabilities.first.key;
  }
  
  // Helper method to get color based on likelihood
  static Color getLikelihoodColor(String likelihood) {
    switch (likelihood.toLowerCase()) {
      case 'high':
        return Colors.red;
      case 'medium':
        return Colors.orange;
      case 'low':
        return Colors.yellow;
      default:
        return Colors.green;
    }
  }
}

class DisorderProbability {
  final double probability;
  final String likelihood;
  final Map<String, dynamic>? features;

  DisorderProbability({
    required this.probability,
    required this.likelihood,
    this.features,
  });

  factory DisorderProbability.fromJson(Map<String, dynamic> json) {
    return DisorderProbability(
      probability: (json['probability'] ?? 0.0).toDouble(),
      likelihood: json['likelihood'] ?? 'Unknown',
      features: json['features'],
    );
  }
}



