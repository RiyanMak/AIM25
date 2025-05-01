import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/detector_provider.dart';
import '../models/detection_result.dart';

class ResultsScreen extends StatelessWidget {
  const ResultsScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Detection Results'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () {
              Provider.of<DetectorProvider>(context, listen: false).clearResults();
              Navigator.pop(context);
            },
            tooltip: 'New Detection',
          ),
        ],
      ),
      body: Consumer<DetectorProvider>(
        builder: (context, detectorProvider, child) {
          if (detectorProvider.isProcessing) {
            return const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 16),
                  Text('Processing image...'),
                ],
              ),
            );
          }

          if (detectorProvider.errorMessage.isNotEmpty) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(
                    Icons.error_outline,
                    color: Colors.red,
                    size: 60,
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Error: ${detectorProvider.errorMessage}',
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: Colors.red),
                  ),
                  const SizedBox(height: 16),
                  ElevatedButton(
                    onPressed: () {
                      detectorProvider.clearResults();
                      Navigator.pop(context);
                    },
                    child: const Text('Try Again'),
                  ),
                ],
              ),
            );
          }

          final result = detectorProvider.result;
          if (result == null) {
            return const Center(
              child: Text('No results available'),
            );
          }

          if (!result.faceDetected) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(
                    Icons.face,
                    color: Colors.orange,
                    size: 60,
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    'No face detected in the image.\nPlease try again with a clearer face image.',
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 16),
                  ElevatedButton(
                    onPressed: () {
                      detectorProvider.clearResults();
                      Navigator.pop(context);
                    },
                    child: const Text('Try Again'),
                  ),
                ],
              ),
            );
          }

          // Show actual detection results
          return _buildResultsView(context, result, detectorProvider);
        },
      ),
    );
  }

  Widget _buildResultsView(
    BuildContext context, 
    DetectionResult result,
    DetectorProvider detectorProvider,
  ) {
    // Find disorder with highest probability
    final highestDisorder = result.getHighestProbabilityDisorder();

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Display processed image if available
          if (detectorProvider.imageFile != null)
            Container(
              height: 200,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
                borderRadius: BorderRadius.circular(8),
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.file(
                  detectorProvider.imageFile!,
                  fit: BoxFit.contain,
                ),
              ),
            ),
          const SizedBox(height: 24),
          
          // Summary card
          Card(
            elevation: 4,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Summary',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Highest probability: $highestDisorder',
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    'Note: This is not a medical diagnosis. Please consult a healthcare professional for proper evaluation.',
                    style: TextStyle(
                      fontStyle: FontStyle.italic,
                      color: Colors.red,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          
          // Individual disorder cards
          _buildDisorderCard(
            'Parkinson\'s Disease', 
            result.pd,
            Icons.accessibility_new,
          ),
          const SizedBox(height: 12),
          _buildDisorderCard(
            'Multiple Sclerosis', 
            result.ms,
            Icons.medical_services,
          ),
          const SizedBox(height: 12),
          _buildDisorderCard(
            'Alzheimer\'s Disease', 
            result.ad,
            Icons.psychology,
          ),
          const SizedBox(height: 24),
          
          // Action buttons
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              ElevatedButton.icon(
                onPressed: () {
                  detectorProvider.clearResults();
                  Navigator.pop(context);
                },
                icon: const Icon(Icons.camera_alt),
                label: const Text('New Detection'),
              ),
              ElevatedButton.icon(
                onPressed: () {
                  // Implement save functionality if needed
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text('Results saved'),
                    ),
                  );
                },
                icon: const Icon(Icons.save),
                label: const Text('Save Results'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
        ],
      ),
    );
  }

  Widget _buildDisorderCard(
    String name, 
    DisorderProbability disorder,
    IconData icon,
  ) {
    final Color indicatorColor = 
        DetectionResult.getLikelihoodColor(disorder.likelihood);
        
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, size: 28),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    name,
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: indicatorColor,
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    disorder.likelihood,
                    style: TextStyle(
                      color: indicatorColor == Colors.yellow ? 
                          Colors.black : Colors.white,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            LinearProgressIndicator(
              value: disorder.probability,
              backgroundColor: Colors.grey[300],
              valueColor: AlwaysStoppedAnimation<Color>(indicatorColor),
              minHeight: 10,
              borderRadius: BorderRadius.circular(5),
            ),
            const SizedBox(height: 8),
            Text(
              'Probability: ${(disorder.probability * 100).toStringAsFixed(1)}%',
              style: const TextStyle(fontWeight: FontWeight.w500),
            ),
            
            // Show features if available
            if (disorder.features != null && disorder.features!.isNotEmpty) ...[
              const SizedBox(height: 12),
              const Text(
                'Key Features:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 4),
              ...disorder.features!.entries
                  .where((e) => e.value is num && e.key != 'mask_face_score') // Filter numeric features
                  .map((e) => Padding(
                        padding: const EdgeInsets.only(bottom: 4),
                        child: Row(
                          children: [
                            Expanded(
                              child: Text(
                                _formatFeatureName(e.key),
                                style: const TextStyle(fontSize: 13),
                              ),
                            ),
                            Text(
                              e.value is double 
                                  ? (e.value as double).toStringAsFixed(2)
                                  : e.value.toString(),
                              style: const TextStyle(fontWeight: FontWeight.w500),
                            ),
                          ],
                        ),
                      ))
                  .toList(),
            ],
          ],
        ),
      ),
    );
  }

  String _formatFeatureName(String key) {
    // Convert snake_case to Title Case with spaces
    return key
        .split('_')
        .map((word) => word.isNotEmpty 
            ? word[0].toUpperCase() + word.substring(1) 
            : '')
        .join(' ');
  }
}