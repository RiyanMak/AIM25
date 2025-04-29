import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import mediapipe as mp
import time
import os
from scipy import signal

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
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 2: Initialize MediaPipe for Facial Landmark Detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.3)

# Step 3: Define Enhanced Facial Feature Extractor
class FacialFeatureExtractor:
    def __init__(self, history_size=30):
        self.key_features = ['smile_amplitude', 'smile_symmetry', 'facial_mobility', 'mask_face_score']
        self.feature_history = {feature: [] for feature in self.key_features}
        self.history_size = history_size
        self.neutral_landmarks = None
        self.neutral_established = False
        self.mouth_corners = [61, 291]
        self.mouth_vertical = [13, 14]
        self.eyebrows = [105, 334]
        self.eyes = [159, 386]
        self.cheeks = [117, 346]
        self.forehead = [10]

    def extract_features(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb_image.shape[:2]
        results = face_mesh.process(rgb_image)
        features = {f: 0 for f in self.key_features}

        if not results.multi_face_landmarks:
            return features, False

        landmarks = results.multi_face_landmarks[0].landmark

        try:
            points = np.array([(lm.x * w, lm.y * h, lm.z) for lm in landmarks])

            if not self.neutral_established:
                self.neutral_landmarks = points.copy()
                self.neutral_established = True

            mouth_left = points[self.mouth_corners[0]]
            mouth_right = points[self.mouth_corners[1]]
            mouth_top = points[self.mouth_vertical[0]]
            mouth_bottom = points[self.mouth_vertical[1]]

            mouth_center = (mouth_top + mouth_bottom) / 2
            mouth_width = np.linalg.norm(mouth_right - mouth_left)
            mouth_height = np.linalg.norm(mouth_top - mouth_bottom)

            smile_measure = mouth_width / (np.linalg.norm(points[self.cheeks[0]] - points[self.cheeks[1]]))
            features['smile_amplitude'] = smile_measure

            nose_tip = points[1]
            left_nose_distance = np.linalg.norm(mouth_left - nose_tip)
            right_nose_distance = np.linalg.norm(mouth_right - nose_tip)
            epsilon = 1e-6
            smile_asymmetry = abs(left_nose_distance - right_nose_distance) / (max(left_nose_distance, right_nose_distance) + epsilon)
            features['smile_symmetry'] = smile_asymmetry

            brow_movement = np.linalg.norm(points[self.eyebrows[0]] - points[self.eyebrows[1]])
            cheek_movement = np.linalg.norm(points[self.cheeks[0]] - points[self.cheeks[1]])
            mouth_movement = mouth_height / mouth_width

            if self.neutral_established:
                neutral_brow = np.linalg.norm(self.neutral_landmarks[self.eyebrows[0]] - self.neutral_landmarks[self.eyebrows[1]])
                neutral_cheek = np.linalg.norm(self.neutral_landmarks[self.cheeks[0]] - self.neutral_landmarks[self.cheeks[1]])
                neutral_mouth_top = self.neutral_landmarks[self.mouth_vertical[0]]
                neutral_mouth_bottom = self.neutral_landmarks[self.mouth_vertical[1]]
                neutral_mouth_height = np.linalg.norm(neutral_mouth_top - neutral_mouth_bottom)
                neutral_mouth_left = self.neutral_landmarks[self.mouth_corners[0]]
                neutral_mouth_right = self.neutral_landmarks[self.mouth_corners[1]]
                neutral_mouth_width = np.linalg.norm(neutral_mouth_right - neutral_mouth_left)
                neutral_mouth_ratio = neutral_mouth_height / (neutral_mouth_width + epsilon)

                brow_mobility = abs(brow_movement - neutral_brow) / (neutral_brow + epsilon)
                cheek_mobility = abs(cheek_movement - neutral_cheek) / (neutral_cheek + epsilon)
                mouth_mobility = abs(mouth_movement - neutral_mouth_ratio) / (neutral_mouth_ratio + epsilon)

                raw_mobility = (brow_mobility * 0.3) + (cheek_mobility * 0.3) + (mouth_mobility * 0.4)
                features['facial_mobility'] = min(1.0, raw_mobility * 10)
            else:
                features['facial_mobility'] = min(1.0, (brow_movement + cheek_movement + mouth_movement) / 3.0)

            for feature in self.key_features:
                self.feature_history[feature].append(features[feature])
                if len(self.feature_history[feature]) > self.history_size:
                    self.feature_history[feature].pop(0)

            expression_ranges = {}
            if all(len(self.feature_history[f]) > 5 for f in ['smile_amplitude', 'facial_mobility']):
                for feature in ['smile_amplitude', 'facial_mobility']:
                    history = np.array(self.feature_history[feature])
                    expression_ranges[feature] = np.std(history) / (np.mean(history) + epsilon)

            expression_range_score = (expression_ranges.get('smile_amplitude', 0) * 0.5 +
                                      expression_ranges.get('facial_mobility', 0) * 0.5)

            features['mask_face_score'] = (
                features['smile_symmetry'] * 0.2 +
                (1.0 - min(1.0, features['smile_amplitude'] * 3)) * 0.3 +
                (1.0 - features['facial_mobility']) * 0.3 +
                (1.0 - min(1.0, expression_range_score * 10)) * 0.2
            )

            for feature in features:
                features[feature] = max(0.0, min(1.0, features[feature]))

            return features, True

        except Exception as e:
            print(f"Error extracting facial features: {e}")
            return features, False
# Step 4: Define Enhanced Jitter Calculator for Micro-Tremor Detection
class EnhancedJitterCalculator:
    def __init__(self, buffer_size=90):  # Increased buffer for frequency analysis
        self.buffer_size = buffer_size
        self.landmark_buffers = {}
        self.face_size_history = []
        self.max_face_size_history = 10
        self.fps = 30  # Estimate, will be refined during operation
        self.last_frame_time = None
        self.frame_times = []
        
    def add_frame(self, landmarks, timestamp=None):
        """Add a frame of landmarks with improved time tracking"""
        # Track frame timing for accurate frequency analysis
        current_time = timestamp if timestamp is not None else time.time()
        if self.last_frame_time is not None:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 30:  # Keep recent frame times
                self.frame_times.pop(0)
            # Update FPS estimate
            if self.frame_times:
                self.fps = 1.0 / np.mean(self.frame_times)
        self.last_frame_time = current_time
        
        # Calculate face size for normalization
        face_size = self._calculate_face_size(landmarks)
        self.face_size_history.append(face_size)
        if len(self.face_size_history) > self.max_face_size_history:
            self.face_size_history.pop(0)
        
        # Use median for stable normalization
        stable_face_size = np.median(self.face_size_history) if self.face_size_history else 1.0
        
        # Store landmarks with timestamp
        for idx, point in enumerate(landmarks):
            if idx not in self.landmark_buffers:
                self.landmark_buffers[idx] = []
            
            # Store point, face size, and timestamp
            self.landmark_buffers[idx].append((point, stable_face_size, current_time))
            if len(self.landmark_buffers[idx]) > self.buffer_size:
                self.landmark_buffers[idx].pop(0)
    
    def _calculate_face_size(self, landmarks):
        """Calculate face size from landmarks for normalization"""
        try:
            # Find bounding box of face
            x_coords = [point[0] for point in landmarks]
            y_coords = [point[1] for point in landmarks]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Use diagonal of bounding box as face size
            face_size = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
            return max(face_size, 1.0)  # Ensure non-zero size
        except Exception:
            return 1.0
    
    def calculate_positional_jitter(self):
        """Enhanced jitter calculation focusing on PD-specific tremors"""
        # PD tremors are typically 4-6 Hz frequency with small amplitude
        # We'll use frequency domain analysis to detect these
        
        # Key points most likely to show PD tremor
        # Focus on perioral and periorbital regions which show earliest signs
        key_points = [
            61, 291,        # Mouth corners (left, right)
            37, 267,        # Eye corners (left, right) 
            9, 8,           # Forehead
            206, 426,       # Cheek (left, right)
            17, 0, 13, 14,  # Lips region
            4, 5            # Chin
        ]
        
        # Metrics to detect PD tremor
        tremor_scores = []
        pd_frequency_power = []
        
        # Current estimated FPS (needed for frequency analysis)
        current_fps = max(15.0, self.fps)  # Ensure reasonable minimum 
        
        for idx in key_points:
            if idx in self.landmark_buffers and len(self.landmark_buffers[idx]) >= 60:  # Need enough data for frequency analysis
                # Extract points, normalized by face size
                buffer_data = self.landmark_buffers[idx]
                points = np.array([data[0] for data in buffer_data])
                face_sizes = np.array([data[1] for data in buffer_data])
                
                # Normalize by face size to account for distance/movement
                for i in range(len(points)):
                    if face_sizes[i] > 0:
                        points[i] = points[i] / face_sizes[i]
                
                # Calculate displacement for each axis
                # Get x, y, z components separately
                x_positions = points[:, 0]
                y_positions = points[:, 1]
                z_positions = points[:, 2] if points.shape[1] > 2 else np.zeros(len(points))
                
                # Detrend data to remove general movement and focus on oscillations
                x_detrended = signal.detrend(x_positions)
                y_detrended = signal.detrend(y_positions)
                z_detrended = signal.detrend(z_positions)
                
                # Window the data to reduce spectral leakage
                window = signal.windows.hann(len(x_detrended))
                x_windowed = x_detrended * window
                y_windowed = y_detrended * window
                z_windowed = z_detrended * window
                
                # Perform FFT on each axis
                # Calculate frequencies for the transform
                n = len(x_windowed)
                freqs = np.fft.rfftfreq(n, d=1/current_fps)
                
                # Calculate FFT magnitudes
                x_fft = np.abs(np.fft.rfft(x_windowed))
                y_fft = np.abs(np.fft.rfft(y_windowed))
                z_fft = np.abs(np.fft.rfft(z_windowed))
                
                # Combine power across all axes (weighted)
                fft_power = (x_fft * 0.4) + (y_fft * 0.4) + (z_fft * 0.2)
                
                # Calculate total power in the PD tremor frequency range (4-7 Hz)
                # This range is characteristic of PD tremor
                pd_freq_mask = (freqs >= 4) & (freqs <= 7)
                if np.any(pd_freq_mask):
                    # Total power in PD frequency range
                    pd_power = np.sum(fft_power[pd_freq_mask])
                    
                    # Total power in all movement frequencies (1-15 Hz)
                    # Exclude very low frequencies (general movement) and noise
                    all_freq_mask = (freqs >= 1) & (freqs <= 15)
                    all_power = np.sum(fft_power[all_freq_mask]) if np.any(all_freq_mask) else 1.0
                    
                    # Calculate ratio of PD-range power to total power
                    # Higher values indicate more energy in PD frequency range
                    pd_power_ratio = pd_power / all_power if all_power > 0 else 0
                    
                    # Also check amplitude of oscillations (PD tremors are small but consistent)
                    avg_amplitude = (np.std(x_detrended) + np.std(y_detrended) + np.std(z_detrended)) / 3
                    
                    # Small but persistent oscillations are more indicative of PD
                    # Too large oscillations are likely voluntary movements
                    amplitude_factor = np.exp(-(avg_amplitude - 0.001)**2 / (2 * 0.0005**2))
                    
                    # Combined tremor score for this landmark
                    tremor_score = pd_power_ratio * amplitude_factor
                    
                    tremor_scores.append(tremor_score)
                    pd_frequency_power.append(pd_power)
                
                # Also detect rapid direction changes (another PD characteristic)
                # Calculate first differences to get velocities
                x_vel = np.diff(x_detrended)
                y_vel = np.diff(y_detrended)
                
                # Count times velocity changes sign (direction reversals)
                x_sign_changes = np.sum(np.abs(np.diff(np.signbit(x_vel))))
                y_sign_changes = np.sum(np.abs(np.diff(np.signbit(y_vel))))
                
                # Normalize by number of frames
                direction_change_rate = (x_sign_changes + y_sign_changes) / (2 * (len(x_vel) - 1))
                
                # Add to tremor scores if high direction change rate detected
                # But only if changes are small (large changes are voluntary)
                if direction_change_rate > 0.3:
                    small_changes = avg_amplitude < 0.01
                    if small_changes:
                        tremor_scores.append(direction_change_rate * 0.5)
        
        # Calculate overall tremor/jitter score
        if tremor_scores:
            # Use 80th percentile for robustness while being sensitive
            # to PD tremors that might only appear in certain facial regions
            overall_tremor = np.percentile(tremor_scores, 80)
            
            # Scale appropriately with sigmoid function for better discrimination
            # between normal variation and PD tremor
            sensitivity = 100.0  # Higher values make the sigmoid steeper
            threshold = 0.1    # Value that maps to 0.5 output
            
            scaled_tremor = 1.0 / (1.0 + np.exp(-sensitivity * (overall_tremor - threshold)))
            
            # Return the scaled tremor score
            return scaled_tremor
        else:
            return 0.0

# Step 5: Enhanced PD Probability Calculator from Emotions
def enhanced_pd_from_emotions(emotion_probs):
    """Calculate PD probability from emotion distribution with improved weighting"""
    # Based on clinical research showing:
    # - Reduced expressivity (especially happiness)
    # - Increased neutral expression ("mask face")
    # - Reduced emotional reactivity overall
    
    # Updated weights based on clinical findings
    weights = np.array([
        0.1,    # Angry - not strongly associated with PD
        0.1,    # Disgust - not strongly associated with PD
        0.1,    # Fear - not strongly associated with PD
        -0.6,   # Happy - REDUCED in PD (negative weight)
        0.1,    # Sad - not strongly associated with PD
        -0.3,   # Surprise - reduced in PD
        0.7     # Neutral - INCREASED in PD ("mask face")
    ])
    
    # Calculate weighted emotion score
    emotion_score = np.sum(weights * emotion_probs)
    
    # Calculate emotion variability (reduced in PD)
    # Exclude neutral from variability calculation
    non_neutral_variability = np.std(emotion_probs[:-1])
    
    # PD typically shows reduced emotional variability and elevated neutral emotion
    combined_score = (emotion_score * 0.7) + ((1.0 - non_neutral_variability) * 0.3)
    
    # Normalize to 0-1 range with refined threshold adjustment
    pd_prob = (combined_score + 0.6) / 1.2
    pd_prob = max(0.0, min(1.0, pd_prob))
    
    return pd_prob

# Step 6: Complete Enhanced PD Detection System
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
        
        # Initialize enhanced components
        self.feature_extractor = FacialFeatureExtractor(history_size=60)
        self.jitter_calculator = EnhancedJitterCalculator(buffer_size=90)
        
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
        
        # Store recent detection results for stability
        self.recent_results = []
        self.max_results_history = 10
    
    def process_frame(self, frame):
        """Process a single frame for enhanced PD detection"""
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
            'pd_likelihood': 'Unknown',
            'confidence': 0.0  # New confidence metric
        }
        
        # Check if face was detected
        if not results.multi_face_landmarks:
            return result
        
        # Face was detected
        result['face_detected'] = True
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Convert landmarks to numpy array
        landmark_points = np.array([(lm.x * w, lm.y * h, lm.z) for lm in landmarks])
        
        # Get current timestamp for accurate timing
        current_time = time.time()
        
        # Add landmarks to jitter calculator
        self.jitter_calculator.add_frame(landmark_points, current_time)
        
        # Calculate jitter (micro-tremor measure)
        jitter = self.jitter_calculator.calculate_positional_jitter()
        result['jitter'] = jitter
        
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
                emotion_pd_prob = enhanced_pd_from_emotions(emotion_probs)
        
        # Calculate overall PD probability with improved weighting
        # Features from the research paper with refined weights
        feature_pd_prob = features['mask_face_score']
        expression_range = features.get('expression_range', 0.0)
        
        # Weight distribution - adjusted based on clinical importance:
        # - Facial features (mask face, asymmetry)
        # - Emotion pattern (if available)
        # - Micro-tremors (enhanced jitter)
        # - Expression range over time
        if self.emotion_model:
            pd_probability = (
                feature_pd_prob * 0.35 + 
                emotion_pd_prob * 0.25 + 
                jitter * 0.3 +
                (1.0 - expression_range) * 0.1  # Low expression range indicates PD
            )
        else:
            pd_probability = (
                feature_pd_prob * 0.5 + 
                jitter * 0.4 +
                (1.0 - expression_range) * 0.1
            )
        
        # Calculate confidence based on data quality and consistency
        # Higher confidence when we have:
        # - More frames of data (better jitter analysis)
        # - Consistent detection results 
        # - Good facial feature extraction
        data_amount_factor = min(1.0, len(list(self.jitter_calculator.landmark_buffers.values())[0]) / 60.0) if self.jitter_calculator.landmark_buffers else 0.0
        feature_quality_factor = 1.0 if all(v > 0 for k, v in features.items() if k != 'mask_face_score') else 0.5
        
        result['confidence'] = (data_amount_factor * 0.7 + feature_quality_factor * 0.3)
        
        # Apply calibration to reduce false positives
        # More conservative threshold for higher specificity
        pd_probability = max(0.0, min(1.0, (pd_probability - 0.25) * 1.25))
        
        # Add to recent results for stability
        self.recent_results.append(pd_probability)
        if len(self.recent_results) > self.max_results_history:
            self.recent_results.pop(0)
        
        # Use stabilized probability (reduces flickering)
        stable_probability = np.median(self.recent_results) if self.recent_results else pd_probability
        
        # Determine PD likelihood category with more nuanced thresholds
        if stable_probability > 0.7:
            pd_likelihood = "High"
        elif stable_probability > 0.45:  # Lowered medium threshold
            pd_likelihood = "Medium"
        elif stable_probability > 0.25:  # Added "Low" category for borderline cases
            pd_likelihood = "Low"
        else:
            pd_likelihood = "Very Low"
        
        # Store final results
        result['pd_probability'] = stable_probability
        result['pd_likelihood'] = pd_likelihood
        
        return result

# Step 7: Run Live PD Detection
def run_pd_detection(model_path=None):
    """Run real-time PD detection using webcam"""
    # Create PD detection system
    pd_system = PDDetectionSystem(model_path)
    
    # Open webcam
    cap = cv2.VideoCapture(0)  # Try camera 0 first
    if not cap.isOpened():
        print("Could not open camera 0, trying camera 1...")
        cap = cv2.VideoCapture(1)  # Try camera 1 as fallback
        if not cap.isOpened():
            print("Could not open any webcam")
            return
    
    print("Parkinson's Disease Detection started. Press 'q' to quit.")
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    
    # For logging results over time (optional)
    results_history = []
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Process frame
        result = pd_system.process_frame(frame)
        
        # Optional: Store results for longitudinal analysis
        if result['face_detected']:
            timestamp = time.time()
            results_history.append({
                'timestamp': timestamp,
                'pd_probability': result['pd_probability'],
                'features': result['features'],
                'jitter': result.get('jitter', 0.0)
            })
            
            # Keep only last 1000 results
            if len(results_history) > 1000:
                results_history.pop(0)
        
        # Display results on frame
        if result['face_detected']:
            # Display PD probability
            if result['pd_likelihood'] == "High":
                color = (0, 0, 255)  # Red
            elif result['pd_likelihood'] == "Medium":
                color = (0, 165, 255)  # Orange
            elif result['pd_likelihood'] == "Low":
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green for Very Low
            
            # Display basic info
            cv2.putText(frame, f"PD Probability: {result['pd_probability']:.2f} ({result['pd_likelihood']})", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display confidence
            confidence_text = f"Confidence: {result['confidence']:.2f}"
            cv2.putText(frame, confidence_text, (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display emotion if available
            if result['emotion']:
                cv2.putText(frame, f"Emotion: {result['emotion']}", 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display key facial features
            y_pos = 120
            features = result['features']
            if features:
                key_features = {
                    'Smile Symmetry': features['smile_symmetry'],
                    'Facial Mobility': features['facial_mobility'],
                    'Mask Face Score': features['mask_face_score'],
                    'Tremor (Micro-Jitter)': result.get('jitter', 0.0)
                }
                
                for name, value in key_features.items():
                    # Normalize value for visualization
                    normalized = min(1.0, value)
                    feature_text = f"{name}: {value:.2f}"
                    cv2.putText(frame, feature_text, (10, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Draw bar representation
                    bar_color = (0, 255, 0)  # Green for low values
                    if name in ['Smile Symmetry', 'Mask Face Score', 'Tremor (Micro-Jitter)'] and value > 0.3:
                        # Higher values for these features indicate PD
                        bar_color = (0, 165, 255) if value > 0.3 else (0, 255, 255)
                        if value > 0.5:
                            bar_color = (0, 0, 255)  # Red for high values
                    elif name == 'Facial Mobility' and value < 0.3:
                        # Lower values for mobility indicate PD
                        bar_color = (0, 165, 255) if value < 0.3 else (0, 255, 255)
                        if value < 0.2:
                            bar_color = (0, 0, 255)  # Red for low values
                    
                    bar_length = int(150 * normalized)
                    cv2.rectangle(frame, (200, y_pos-15), (200+bar_length, y_pos-5), bar_color, -1)
                    
                    y_pos += 25
            
            # Display PD indicators from research paper
            indicators = []
            if features and features['smile_symmetry'] > 0.3:
                indicators.append("Asymmetrical smile")
            if features and features['mask_face_score'] > 0.5:
                indicators.append("Reduced expressivity (mask face)")
            if result.get('jitter', 0.0) > 0.3:
                indicators.append("Facial micro-tremors detected")
            if features and features['facial_mobility'] < 0.2:
                indicators.append("Reduced facial mobility")
            if features.get('expression_range', 0) < 0.2:
                indicators.append("Limited expression range")
            
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
    
    # Optional: Save results history for later analysis
    # np.save('pd_detection_results.npy', np.array(results_history))

# Add function to analyze stored detection results over time (optional)
def analyze_detection_history(results_file):
    """Analyze stored detection results for longitudinal analysis"""
    try:
        results = np.load(results_file, allow_pickle=True)
        
        # Extract timestamps and probabilities
        timestamps = [r['timestamp'] for r in results]
        probabilities = [r['pd_probability'] for r in results]
        
        # Convert to relative time in seconds from start
        start_time = timestamps[0]
        rel_times = [(t - start_time) for t in timestamps]
        
        # Calculate statistics
        avg_prob = np.mean(probabilities)
        max_prob = np.max(probabilities)
        min_prob = np.min(probabilities)
        std_dev = np.std(probabilities)
        
        print(f"Analysis of {len(results)} detection points:")
        print(f"Average PD probability: {avg_prob:.4f}")
        print(f"Maximum PD probability: {max_prob:.4f}")
        print(f"Minimum PD probability: {min_prob:.4f}")
        print(f"Standard deviation: {std_dev:.4f}")
        
        # You could also plot the results using matplotlib
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(12, 6))
        # plt.plot(rel_times, probabilities)
        # plt.xlabel('Time (seconds)')
        # plt.ylabel('PD Probability')
        # plt.title('Parkinson\'s Disease Detection Over Time')
        # plt.grid(True)
        # plt.savefig('pd_detection_trend.png')
        # plt.show()
        
        return True
        
    except Exception as e:
        print(f"Error analyzing results: {e}")
        return False

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
    
    # Uncomment to analyze saved results (if available)
    # analyze_detection_history('pd_detection_results.npy')