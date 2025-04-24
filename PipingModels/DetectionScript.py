import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import mediapipe as mp
import time
import os

# Step 1: Define the Emotion Recognition Model
class FacialExpressionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(FacialExpressionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(18432, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        # No print statement for production
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 2: Initialize MediaPipe for Facial Landmark Detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.3)  # Lower threshold for better detection

# Step 3: Define Facial Feature Extractor based on the research paper
class FacialFeatureExtractor:
    def __init__(self):
        # Key features mentioned in the paper
        self.key_features = [
            'smile_amplitude',       # Reduced in PD
            'smile_symmetry',        # Asymmetrical in PD
            'facial_mobility',       # Reduced in PD
            'mask_face_score'        # Higher in PD
        ]
    
    def extract_features(self, image):
        """Extract facial features relevant to PD detection from the paper"""
        # Convert image to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb_image.shape[:2]
        
        # Process image with MediaPipe
        results = face_mesh.process(rgb_image)
        
        # Initialize features dictionary
        features = {
            'smile_amplitude': 0,
            'smile_symmetry': 0,
            'facial_mobility': 0,
            'mask_face_score': 0
        }
        
        # Check if face was detected
        if not results.multi_face_landmarks:
            return features, False
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        try:
            # Convert landmarks to numpy array
            points = np.array([(lm.x * w, lm.y * h, lm.z) for lm in landmarks])
            
            # Extract features mentioned in the research paper
            
            # 1. Smile amplitude - measure of facial expression magnitude
            # Research shows PD patients have smaller smile amplitude
            mouth_left = points[61]
            mouth_right = points[291]
            mouth_top = points[13]
            mouth_bottom = points[14]
            mouth_center = (mouth_top + mouth_bottom) / 2
            smile_measure = abs(mouth_center[1] - (mouth_left[1] + mouth_right[1])/2)
            features['smile_amplitude'] = smile_measure
            
            # 2. Smile symmetry - PD often has asymmetrical facial expressions
            left_smile = np.linalg.norm(mouth_left - mouth_top)
            right_smile = np.linalg.norm(mouth_right - mouth_top)
            epsilon = 1e-6  # Prevent division by zero
            smile_asymmetry = abs(left_smile - right_smile) / (max(left_smile, right_smile) + epsilon)
            features['smile_symmetry'] = smile_asymmetry
            
            # 3. Facial mobility - measure of overall facial movement potential
            # The paper mentioned PD patients have restricted facial movement
            cheek_raise_left = np.linalg.norm(points[117] - points[123])
            cheek_raise_right = np.linalg.norm(points[346] - points[352])
            brow_movement = np.linalg.norm(points[9] - points[337])
            # Add after calculating facial_mobility
            features['facial_mobility'] = min(1.0, features['facial_mobility'])
            
            # 4. Mask face score - combined measure based on research findings
            # Higher value indicates more PD-like features
            features['mask_face_score'] = (
                smile_asymmetry * 0.4 + 
                (1.0 - min(1.0, smile_measure * 5)) * 0.3 + 
                (1.0 - min(1.0, features['facial_mobility'] * 10)) * 0.3
            )
            
            return features, True
        
        except Exception as e:
            print(f"Error extracting facial features: {e}")
            return features, False

# Step 4: Define Jitter Calculator for Tremor Detection (from paper)
class JitterCalculator:
    def __init__(self, buffer_size=30):
        self.buffer_size = buffer_size
        self.landmark_buffers = {}
        
    def add_frame(self, landmarks):
        """Add a frame of landmarks to the buffer"""
        for idx, point in enumerate(landmarks):
            if idx not in self.landmark_buffers:
                self.landmark_buffers[idx] = []
            
            self.landmark_buffers[idx].append(point)
            if len(self.landmark_buffers[idx]) > self.buffer_size:
                self.landmark_buffers[idx].pop(0)
    
    def calculate_positional_jitter(self):
        """Calculate positional jitter as defined in the paper"""
        # Key points: facial landmarks that are important for PD detection
        key_points = [61, 291, 13, 14, 10, 338, 117, 346]  # Mouth, brows, cheeks
        jitter_values = []
        
        for idx in key_points:
            if idx in self.landmark_buffers and len(self.landmark_buffers[idx]) >= 2:
                points = np.array(self.landmark_buffers[idx])
                
                # Jitter_abs: average absolute difference between consecutive positions
                diffs = np.abs(np.diff(points, axis=0))
                jitter_abs = np.mean(diffs)
                
                jitter_values.append(jitter_abs)
        
        # Average jitter across key points
        if jitter_values:
            avg_jitter = np.mean(jitter_values)
            # Scale to 0-1 range for easier interpretation
            scaled_jitter = min(1.0, avg_jitter * 10.0)  # Even more aggressive reduction
            return scaled_jitter
        else:
            return 0.0

# Step 5: PD Probability Calculator from Emotions
def pd_from_emotions(emotion_probs):
    """Calculate PD probability from emotion distribution (based on paper)"""
    # Paper mentions PD patients have:
    # - Reduced expressivity (especially happiness)
    # - Increased neutral expression ("mask face")
    
    # Weights based on clinical findings
    weights = np.array([
        0.1,    # Angry - not strongly associated with PD
        0.1,    # Disgust - not strongly associated with PD
        0.1,    # Fear - not strongly associated with PD
        -0.5,   # Happy - REDUCED in PD (negative weight)
        0.1,    # Sad - not strongly associated with PD
        -0.2,   # Surprise - reduced in PD
        0.6     # Neutral - INCREASED in PD ("mask face")
    ])
    
    # Calculate weighted emotion score
    emotion_score = np.sum(weights * emotion_probs)
    
    # Normalize to 0-1 range with threshold adjustment
    pd_prob = (emotion_score + 0.5) / 1.0
    pd_prob = max(0.0, min(1.0, (pd_prob - 0.3) * 1.4))
    
    return pd_prob

# Step 6: Complete PD Detection System
class PDDetectionSystem:
    def __init__(self, model_path=None):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                "mps" if torch.backends.mps.is_available() else 
                                "cpu")
        print(f"Using device: {self.device}")
        
        # Load emotion recognition model
        self.emotion_model = None
        if model_path and os.path.exists(model_path):
            try:
                self.emotion_model = FacialExpressionCNN(num_classes=7).to(self.device)
                self.emotion_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.emotion_model.eval()
                print(f"Emotion model loaded from: {model_path}")
            except Exception as e:
                print(f"Error loading emotion model: {e}")
                print("Running without emotion recognition")
        else:
            print("No emotion model provided or file not found.")
            print("Running with facial features only.")
        
        # Initialize components
        self.feature_extractor = FacialFeatureExtractor()
        self.jitter_calculator = JitterCalculator(buffer_size=30)
        
        # Define emotion classes
        self.emotions = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        
        # Transform for images
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ])
    
    def process_frame(self, frame):
        """Process a single frame for PD detection"""
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Process with MediaPipe for landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        # Initialize result dictionary
        result = {
            'face_detected': False,
            'emotion': None,
            'emotion_probs': None,
            'features': None,
            'pd_probability': 0.0,
            'pd_likelihood': 'Unknown'
        }
        
        # Check if face was detected
        if not results.multi_face_landmarks:
            return result
        
        # Face was detected
        result['face_detected'] = True
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Convert landmarks to numpy array
        landmark_points = np.array([(lm.x * w, lm.y * h, lm.z) for lm in landmarks])
        
        # Add landmarks to jitter calculator
        self.jitter_calculator.add_frame(landmark_points)
        
        # Calculate jitter (tremor measure)
        jitter = self.jitter_calculator.calculate_positional_jitter()
        
        # Extract facial features
        features, _ = self.feature_extractor.extract_features(frame)
        result['features'] = features
        
        # Predict emotion if model is available
        emotion_pd_prob = 0.0
        if self.emotion_model:
            # Prepare image for emotion model
            emotion_input = self.transform(rgb_frame).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                emotion_outputs = self.emotion_model(emotion_input)
                emotion_probs = torch.softmax(emotion_outputs, dim=1)[0].cpu().numpy()
                _, emotion_class = torch.max(emotion_outputs, 1)
                
                # Get emotion name
                emotion = self.emotions[emotion_class.item()]
                
                # Store emotion results
                result['emotion'] = emotion
                result['emotion_probs'] = emotion_probs
                
                # Calculate PD probability from emotion distribution
                emotion_pd_prob = pd_from_emotions(emotion_probs)
        
        # Calculate overall PD probability (combining all features)
        # Weight features based on the research paper's findings
        feature_pd_prob = features['mask_face_score']
        
        # Weight distribution:
        # - 40% from facial features (mask face, asymmetry)
        # - 40% from emotion pattern (if available)
        # - 20% from tremor (jitter)
        if self.emotion_model:
            pd_probability = (
                feature_pd_prob * 0.4 + 
                emotion_pd_prob * 0.4 + 
                jitter * 0.2
            )
        else:
            pd_probability = (
                feature_pd_prob * 0.7 + 
                jitter * 0.3
            )
        
        # Apply calibration to reduce false positives
        pd_probability = max(0.0, min(1.0, (pd_probability - 0.3) * 1.3))
        
        # Determine PD likelihood category
        if pd_probability > 0.7:
            pd_likelihood = "High"
        elif pd_probability > 0.4:
            pd_likelihood = "Medium"
        else:
            pd_likelihood = "Low"
        
        # Store final results
        result['pd_probability'] = pd_probability
        result['pd_likelihood'] = pd_likelihood
        result['jitter'] = jitter
        
        return result

# Step 7: Run Live PD Detection
def run_pd_detection(model_path=None):
    """Run real-time PD detection using webcam"""
    # Create PD detection system
    pd_system = PDDetectionSystem(model_path)
    
    # Open webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    print("Parkinson's Disease Detection started. Press 'q' to quit.")
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Process frame
        result = pd_system.process_frame(frame)
        
        # Display results on frame
        if result['face_detected']:
            # Display PD probability
            color = (0, 255, 0)  # Green for Low
            if result['pd_likelihood'] == "High":
                color = (0, 0, 255)  # Red
            elif result['pd_likelihood'] == "Medium":
                color = (0, 165, 255)  # Orange
            
            # Display basic info
            cv2.putText(frame, f"PD Probability: {result['pd_probability']:.2f} ({result['pd_likelihood']})", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display emotion if available
            if result['emotion']:
                cv2.putText(frame, f"Emotion: {result['emotion']}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Display key facial features
            y_pos = 90
            features = result['features']
            if features:
                key_features = {
                    'Smile Symmetry': features['smile_symmetry'],
                    'Facial Mobility': features['facial_mobility'],
                    'Mask Face Score': features['mask_face_score'],
                    'Tremor (Jitter)': result.get('jitter', 0.0)
                }
                
                for name, value in key_features.items():
                    # Normalize value for visualization
                    normalized = min(1.0, value)
                    feature_text = f"{name}: {value:.2f}"
                    cv2.putText(frame, feature_text, (10, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Draw bar representation
                    bar_length = int(150 * normalized)
                    cv2.rectangle(frame, (200, y_pos-15), (200+bar_length, y_pos-5), color, -1)
                    
                    y_pos += 25
            
            # Display PD indicators from research paper
            indicators = []
            if features and features['smile_symmetry'] > 0.3:
                indicators.append("Asymmetrical smile")
            if features and features['mask_face_score'] > 0.5:
                indicators.append("Reduced expressivity (mask face)")
            if result.get('jitter', 0.0) > 0.3:
                indicators.append("Facial tremor detected")
            if features and features['facial_mobility'] < 0.1:
                indicators.append("Reduced facial mobility")
            
            if indicators:
                cv2.putText(frame, "PD Indicators:", (10, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_pos += 20
                
                for indicator in indicators:
                    cv2.putText(frame, f"- {indicator}", (20, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_pos += 20
            
            # Calculate and display FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
                cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-120, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # No face detected
            cv2.putText(frame, "No face detected", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow('Parkinson\'s Disease Detection', frame)
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run PD detection with trained model
if __name__ == "__main__":
    # Path to your trained emotion recognition model
    # This will be created by running the training script
    model_path = "./PipingModels/best_emotion_model.pth"
    
    # You can uncomment this if you want to run detection without training first
    # (will use facial features only, without emotion recognition)
    # run_pd_detection()
    
    # Or run with your trained model
    run_pd_detection(model_path)