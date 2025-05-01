import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:provider/provider.dart';
import 'package:camera/camera.dart';
import 'providers/detector_provider.dart';
import 'screens/camera_screen.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  // Ensure Flutter is initialized FIRST
  WidgetsFlutterBinding.ensureInitialized();
  
  // Now it's safe to print debug info
  debugPrint("==== App Starting ====");
  
  // Load environment variables
  await dotenv.load(fileName: ".env").catchError((e) {
    debugPrint("Warning: .env file not found. Using default API URL.");
  });
  
  // Get available cameras
  try {
    cameras = await availableCameras();
  } on CameraException catch (e) {
    debugPrint('Error getting cameras: ${e.description}');
  }
  
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (context) => DetectorProvider(),
      child: MaterialApp(
        title: 'Neurological Disorder Detector',
        theme: ThemeData(
          primarySwatch: Colors.blue,
          visualDensity: VisualDensity.adaptivePlatformDensity,
          useMaterial3: true,
        ),
        darkTheme: ThemeData(
          brightness: Brightness.dark,
          primarySwatch: Colors.blue,
          visualDensity: VisualDensity.adaptivePlatformDensity,
          useMaterial3: true,
        ),
        themeMode: ThemeMode.system,
        home: const HomeScreen(),
      ),
    );
  }
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Neurological Disorder Detector'),
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(
                Icons.health_and_safety,
                size: 100,
                color: Colors.blue,
              ),
              const SizedBox(height: 24),
              const Text(
                'Welcome to Neurological Disorder Detector',
                style: TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              const Text(
                'This app can help detect signs of neurological disorders using computer vision and AI.',
                style: TextStyle(fontSize: 16),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 8),
              const Text(
                'Note: This is not a medical diagnosis. Always consult a healthcare professional.',
                style: TextStyle(
                  fontSize: 14,
                  fontStyle: FontStyle.italic,
                  color: Colors.red,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 32),
              ElevatedButton.icon(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const CameraScreen(),
                    ),
                  );
                },
                icon: const Icon(Icons.camera_alt),
                label: const Text('Start Detection'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 32,
                    vertical: 16,
                  ),
                ),
              ),
              const SizedBox(height: 16),
              TextButton.icon(
                onPressed: () {
                  // Pick from gallery directly
                  final detectorProvider = Provider.of<DetectorProvider>(context, listen: false);
                  detectorProvider.pickAndProcessImage();
                },
                icon: const Icon(Icons.photo_library),
                label: const Text('Select from Gallery'),
              ),
              const SizedBox(height: 24),
              // API status indicator
              Consumer<DetectorProvider>(
                builder: (context, provider, child) {
                  return Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                        width: 12,
                        height: 12,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: provider.isApiAvailable ? Colors.green : Colors.red,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'API Status: ${provider.isApiAvailable ? 'Online' : 'Offline'}',
                        style: TextStyle(
                          color: provider.isApiAvailable ? Colors.green : Colors.red,
                        ),
                      ),
                    ],
                  );
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}