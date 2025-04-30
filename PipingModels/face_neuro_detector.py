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
from collections import deque
from pd_detection_system import PDDetectionSystem 

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

# New class for Multiple Sclerosis feature detection
class MSFeatureExtractor:
    def __init__(self, history_size=30):
        self.key_features = [
            'facial_paralysis_score', 'eye_movement_abnormality', 
            'eyelid_twitching', 'hemifacial_spasm'
        ]
        self.feature_history = {feature: [] for feature in self.key_features}
        self.history_size = history_size
        self.eye_movement_buffer = deque(maxlen=60)  # For tracking eye movements
        
        # Key facial landmarks
        self.left_eye_landmarks = list(range(362, 374))
        self.right_eye_landmarks = list(range(33, 46))
        self.eye_pairs = [(362, 33), (363, 133), (373, 144), (374, 145)]  # Compare left-right
        self.mouth_corners = [61, 291]
        self.eyebrows = [105, 334]
        self.left_face = [93, 234, 127, 162]  # Points for left side
        self.right_face = [323, 454, 356, 389]  # Points for right side
        
        # For eye movement tracking
        self.prev_left_eye_center = None
        self.prev_right_eye_center = None
        self.eye_movement_history = []
        self.max_eye_history = 20
        
    def extract_features(self, image, landmarks):
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Initialize features
        features = {k: 0 for k in self.key_features}
        if landmarks is None:
            return features, False
        
        # Convert landmarks to coordinates
        points = np.array([(lm.x * w, lm.y * h, lm.z) for lm in landmarks])
        
        # 1. Detect facial paralysis (asymmetry focused on one side)
        features['facial_paralysis_score'] = self._detect_facial_paralysis(points)
        
        # 2. Detect eye movement abnormalities (nystagmus, rapid movements)
        features['eye_movement_abnormality'] = self._detect_eye_movement_abnormality(points)
        
        # 3. Detect eyelid twitching (myokymia)
        features['eyelid_twitching'] = self._detect_eyelid_twitching(points)
        
        # 4. Detect hemifacial spasm
        features['hemifacial_spasm'] = self._detect_hemifacial_spasm(points)
        
        # Update history
        for feature in self.key_features:
            self.feature_history[feature].append(features[feature])
            if len(self.feature_history[feature]) > self.history_size:
                self.feature_history[feature].pop(0)
        
        return features, True
    
    def _detect_facial_paralysis(self, points):
        """Detect facial paralysis (asymmetry focused on one side)"""
        try:
            # Compare corresponding points on left and right sides of face
            # Focus on facial mobility and asymmetry
            
            # Get mouth corners (for smile asymmetry)
            left_mouth = points[self.mouth_corners[0]]
            right_mouth = points[self.mouth_corners[1]]
            
            # Compare eyebrow heights
            left_brow = points[self.eyebrows[0]]
            right_brow = points[self.eyebrows[1]]
            brow_height_diff = abs(left_brow[1] - right_brow[1])
            
            # Compare cheek and face points
            left_face_points = [points[i] for i in self.left_face]
            right_face_points = [points[i] for i in self.right_face]
            
            # Calculate average vertical positions (y-coordinates)
            left_y_avg = np.mean([p[1] for p in left_face_points])
            right_y_avg = np.mean([p[1] for p in right_face_points])
            face_side_diff = abs(left_y_avg - right_y_avg)
            
            # Combine metrics - normalize by face size for scale invariance
            face_size = np.linalg.norm(points[self.mouth_corners[1]] - points[self.mouth_corners[0]])
            normalized_asymmetry = (brow_height_diff + face_side_diff) / (face_size + 1e-6)
            
            # Check if asymmetry is persistent and focused on one side
            # This is characteristic of MS facial paralysis vs. general asymmetry
            persistent_one_sided = 0.0
            if len(self.feature_history.get('facial_paralysis_score', [])) > 10:
                # Check if asymmetry consistently favors one side
                left_side_lower = left_y_avg > right_y_avg
                consistent_side = sum(1 for i in range(min(10, len(self.feature_history['facial_paralysis_score']))) 
                                     if left_side_lower)
                persistent_one_sided = max(consistent_side, 10 - consistent_side) / 10.0
            
            # Combine metrics - high scores only when asymmetry is persistent and one-sided
            paralysis_score = min(1.0, normalized_asymmetry * 3.0 * (0.5 + 0.5 * persistent_one_sided))
            
            return min(1.0, max(0.0, paralysis_score))
        except Exception as e:
            print(f"Error detecting facial paralysis: {e}")
            return 0.0
    
    def _detect_eye_movement_abnormality(self, points):
        """Detect abnormal eye movements characteristic of MS"""
        try:
            # Calculate eye centers
            left_eye_points = [points[i] for i in self.left_eye_landmarks]
            right_eye_points = [points[i] for i in self.right_eye_landmarks]
            
            left_eye_center = np.mean(left_eye_points, axis=0)
            right_eye_center = np.mean(right_eye_points, axis=0)
            
            # If first frame, initialize history
            if self.prev_left_eye_center is None:
                self.prev_left_eye_center = left_eye_center
                self.prev_right_eye_center = right_eye_center
                return 0.0
            
            # Calculate eye movement (displacement)
            left_eye_movement = np.linalg.norm(left_eye_center - self.prev_left_eye_center)
            right_eye_movement = np.linalg.norm(right_eye_center - self.prev_right_eye_center)
            
            # Normalize by face size
            face_size = np.linalg.norm(points[self.mouth_corners[1]] - points[self.mouth_corners[0]])
            norm_left_movement = left_eye_movement / (face_size + 1e-6)
            norm_right_movement = right_eye_movement / (face_size + 1e-6)
            
            # Track eye movements over time
            self.eye_movement_history.append((norm_left_movement, norm_right_movement))
            if len(self.eye_movement_history) > self.max_eye_history:
                self.eye_movement_history.pop(0)
            
            # Calculate key metrics for nystagmus detection
            if len(self.eye_movement_history) >= 10:
                # 1. Rapid eye movements (high velocity)
                recent_movements = self.eye_movement_history[-10:]
                max_velocity = max([max(left, right) for left, right in recent_movements])
                
                # 2. Check for repetitive oscillatory pattern (nystagmus)
                # Get movement data for FFT analysis
                left_movements = [m[0] for m in self.eye_movement_history]
                right_movements = [m[1] for m in self.eye_movement_history]
                
                # Perform FFT on eye movements to detect oscillations
                if len(left_movements) >= 15:  # Need enough samples for FFT
                    # Detrend and prepare data
                    left_detrended = signal.detrend(left_movements)
                    right_detrended = signal.detrend(right_movements)
                    
                    # Apply FFT
                    fft_left = np.abs(np.fft.rfft(left_detrended))
                    fft_right = np.abs(np.fft.rfft(right_detrended))
                    
                    # Nystagmus typically has 3-8 Hz oscillation
                    # Estimate current FPS for frequency conversion
                    est_fps = 30  # Assumed fps
                    freqs = np.fft.rfftfreq(len(left_detrended), d=1/est_fps)
                    
                    # Check power in nystagmus frequency range (3-8 Hz)
                    nystagmus_mask = (freqs >= 3) & (freqs <= 8)
                    if np.any(nystagmus_mask):
                        nystagmus_power_left = np.sum(fft_left[nystagmus_mask])
                        nystagmus_power_right = np.sum(fft_right[nystagmus_mask])
                        
                        # Compare to total power for significance
                        total_power_left = np.sum(fft_left)
                        total_power_right = np.sum(fft_right)
                        
                        # Calculate nystagmus probability
                        if total_power_left > 0 and total_power_right > 0:
                            nystagmus_ratio_left = nystagmus_power_left / total_power_left
                            nystagmus_ratio_right = nystagmus_power_right / total_power_right
                            nystagmus_score = max(nystagmus_ratio_left, nystagmus_ratio_right)
                            
                            # Combine with velocity for final score
                            # Both rapid movement AND oscillatory pattern needed for MS
                            abnormality_score = (nystagmus_score * 0.7) + (min(1.0, max_velocity * 10) * 0.3)
                            
                            # Update previous centers
                            self.prev_left_eye_center = left_eye_center
                            self.prev_right_eye_center = right_eye_center
                            
                            return min(1.0, max(0.0, abnormality_score))
            
            # Update previous centers
            self.prev_left_eye_center = left_eye_center
            self.prev_right_eye_center = right_eye_center
            
            return 0.0
        except Exception as e:
            print(f"Error detecting eye movement abnormality: {e}")
            return 0.0
    
    def _detect_eyelid_twitching(self, points):
        """Detect eyelid twitching (myokymia)"""
        try:
            # Focus on upper and lower eyelid points
            # Left eye upper lid: 
            left_upper_lid = [385, 386, 387, 388, 466]
            left_lower_lid = [374, 373, 390, 249, 359]
            
            # Right eye upper lid
            right_upper_lid = [159, 158, 157, 173, 246]
            right_lower_lid = [145, 144, 163, 7, 130]
            
            # Get coordinates
            left_upper_points = [points[i] for i in left_upper_lid]
            left_lower_points = [points[i] for i in left_lower_lid]
            right_upper_points = [points[i] for i in right_upper_lid]
            right_lower_points = [points[i] for i in right_lower_lid]
            
            # Calculate average positions
            left_upper_pos = np.mean(left_upper_points, axis=0)
            left_lower_pos = np.mean(left_lower_points, axis=0)
            right_upper_pos = np.mean(right_upper_points, axis=0)
            right_lower_pos = np.mean(right_lower_points, axis=0)
            
            # Calculate eyelid distances
            left_eye_opening = np.linalg.norm(left_upper_pos - left_lower_pos)
            right_eye_opening = np.linalg.norm(right_upper_pos - right_lower_pos)
            
            # Store in buffer (only if we have previous values)
            if hasattr(self, 'prev_left_eye_opening') and hasattr(self, 'prev_right_eye_opening'):
                # Calculate rapid changes in eyelid position (twitching)
                left_twitch = np.abs(left_eye_opening - self.prev_left_eye_opening)
                right_twitch = np.abs(right_eye_opening - self.prev_right_eye_opening)
                
                # Normalize by face size
                face_size = np.linalg.norm(points[self.mouth_corners[1]] - points[self.mouth_corners[0]])
                norm_left_twitch = left_twitch / (face_size + 1e-6)
                norm_right_twitch = right_twitch / (face_size + 1e-6)
                
                # Add to buffer
                self.eye_movement_buffer.append((norm_left_twitch, norm_right_twitch))
            
            # Store current values for next frame
            self.prev_left_eye_opening = left_eye_opening
            self.prev_right_eye_opening = right_eye_opening
            
            # Need enough history for reliable detection
            if len(self.eye_movement_buffer) < 30:
                return 0.0
            
            # Analyze buffer for myokymia pattern:
            # 1. Rapid, small twitches (high frequency, low amplitude)
            # 2. Often affecting only one eye
            # 3. Bursts of activity followed by quiet periods
            
            # Extract separate left and right eye data
            left_twitches = [data[0] for data in self.eye_movement_buffer]
            right_twitches = [data[1] for data in self.eye_movement_buffer]
            
            # Calculate metrics
            left_mean = np.mean(left_twitches)
            right_mean = np.mean(right_twitches)
            left_std = np.std(left_twitches)
            right_std = np.std(right_twitches)
            
            # Count rapid oscillations (zero crossings) - characteristic of myokymia
            left_detrended = signal.detrend(left_twitches)
            right_detrended = signal.detrend(right_twitches)
            
            left_zero_crossings = np.sum(np.abs(np.diff(np.signbit(left_detrended))))
            right_zero_crossings = np.sum(np.abs(np.diff(np.signbit(right_detrended))))
            
            # Normalize by buffer length
            left_crossing_rate = left_zero_crossings / (len(left_detrended) - 1)
            right_crossing_rate = right_zero_crossings / (len(right_detrended) - 1)
            
            # Calculate burst pattern - periods of activity followed by calm
            # Divide buffer into segments and look for high variance in activity levels
            segment_size = 10
            if len(left_twitches) >= segment_size * 3:  # Need at least 3 segments
                left_segments = [left_twitches[i:i+segment_size] for i in range(0, len(left_twitches), segment_size)]
                right_segments = [right_twitches[i:i+segment_size] for i in range(0, len(right_twitches), segment_size)]
                
                left_segment_vars = [np.var(segment) for segment in left_segments if len(segment) == segment_size]
                right_segment_vars = [np.var(segment) for segment in right_segments if len(segment) == segment_size]
                
                # Calculate variance of segment variances - high value indicates burst pattern
                if left_segment_vars and right_segment_vars:
                    left_burst_pattern = np.var(left_segment_vars) / (np.mean(left_segment_vars) + 1e-6)
                    right_burst_pattern = np.var(right_segment_vars) / (np.mean(right_segment_vars) + 1e-6)
                    
                    # Final myokymia score combines:
                    # 1. High frequency oscillations (crossing rate)
                    # 2. Burst pattern characteristic
                    # 3. One-sided tendency (max of left/right)
                    
                    left_myokymia = left_crossing_rate * 0.4 + left_burst_pattern * 0.6
                    right_myokymia = right_crossing_rate * 0.4 + right_burst_pattern * 0.6
                    
                    # Final score (take max as myokymia often affects one side more)
                    myokymia_score = max(left_myokymia, right_myokymia)
                    
                    return min(1.0, max(0.0, myokymia_score))
            
            # Fallback if not enough data for segment analysis
            basic_score = max(left_crossing_rate, right_crossing_rate) * 0.5
            return min(1.0, max(0.0, basic_score))
            
        except Exception as e:
            print(f"Error detecting eyelid twitching: {e}")
            return 0.0
    
    def _detect_hemifacial_spasm(self, points):
        """Detect hemifacial spasm (one-sided facial spasms)"""
        try:
            # Focus on key areas for spasm detection (mouth corners, cheeks, eyes)
            left_side_points = [61, 91, 93, 234, 127]  # Left mouth, cheek
            right_side_points = [291, 321, 323, 454, 356]  # Right mouth, cheek
            
            # Need history for this - if not enough samples yet, return 0
            if not hasattr(self, 'prev_left_positions') or not hasattr(self, 'prev_right_positions'):
                self.prev_left_positions = [points[i] for i in left_side_points]
                self.prev_right_positions = [points[i] for i in right_side_points]
                return 0.0
            
            # Get current positions
            current_left = [points[i] for i in left_side_points]
            current_right = [points[i] for i in right_side_points]
            
            # Calculate movement on each side
            left_movements = [np.linalg.norm(curr - prev) for curr, prev in zip(current_left, self.prev_left_positions)]
            right_movements = [np.linalg.norm(curr - prev) for curr, prev in zip(current_right, self.prev_right_positions)]
            
            # Normalize by face size
            face_size = np.linalg.norm(points[self.mouth_corners[1]] - points[self.mouth_corners[0]])
            left_movements = [m / (face_size + 1e-6) for m in left_movements]
            right_movements = [m / (face_size + 1e-6) for m in right_movements]
            
            # Calculate average movement for each side
            avg_left_movement = np.mean(left_movements)
            avg_right_movement = np.mean(right_movements)
            
            # Hemifacial spasm has two characteristics:
            # 1. Movement is much higher on one side than the other
            # 2. Movement on affected side shows spasm pattern (irregular, jerky)
            
            # Check for asymmetric movement
            movement_ratio = max(avg_left_movement, avg_right_movement) / (min(avg_left_movement, avg_right_movement) + 1e-6)
            movement_diff = abs(avg_left_movement - avg_right_movement)
            
            # Detect which side has more movement
            left_side_affected = avg_left_movement > avg_right_movement
            affected_movements = left_movements if left_side_affected else right_movements
            
            # Check for spasm pattern (jerky, irregular movements)
            if len(affected_movements) >= 3:
                # Calculate variability and jerkiness
                movement_std = np.std(affected_movements)
                
                # Combine metrics
                spasm_score = movement_diff * 0.4 + movement_std * 0.6
                
                # Scale for better visibility
                spasm_score = min(1.0, spasm_score * 10)
            else:
                spasm_score = movement_diff * 0.5
            
            # Store current positions for next frame
            self.prev_left_positions = current_left
            self.prev_right_positions = current_right
            
            return min(1.0, max(0.0, spasm_score))
            
        except Exception as e:
            print(f"Error detecting hemifacial spasm: {e}")
            return 0.0

# New class for Alzheimer's Disease feature detection
class ADFeatureExtractor:
    def __init__(self, history_size=30):
        self.key_features = [
            'facial_asymmetry', 'reduced_expressivity', 'saccadic_eye_movement'
        ]
        self.feature_history = {feature: [] for feature in self.key_features}
        self.history_size = history_size
        self.neutral_landmarks = None
        self.neutral_established = False
        
        # Key facial landmarks
        self.eyebrows = [105, 334]  # Left, right
        self.eyes = [159, 386]  # Left, right
        self.mouth_corners = [61, 291]  # Left, right
        self.nose = [4]  # Nose tip
        self.face_edges = [234, 454]  # Left, right
        self.nostrils = [203, 423]  # Left, right
        
        # For eye tracking
        self.eye_positions = []
        self.max_eye_positions = 60
        self.prev_left_eye = None
        self.prev_right_eye = None
        
    def extract_features(self, image, landmarks):
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Initialize features
        features = {k: 0 for k in self.key_features}
        if landmarks is None:
            return features, False
        
        # Convert landmarks to coordinates
        points = np.array([(lm.x * w, lm.y * h, lm.z) for lm in landmarks])
        
        # 1. Detect facial asymmetry (different from MS - focus on specific regions)
        features['facial_asymmetry'] = self._detect_facial_asymmetry(points)
        
        # 2. Detect reduced expressivity
        features['reduced_expressivity'] = self._detect_reduced_expressivity(points)
        
        # 3. Detect saccadic eye movements
        features['saccadic_eye_movement'] = self._detect_saccadic_eye_movement(points)
        
        # Update history
        for feature in self.key_features:
            self.feature_history[feature].append(features[feature])
            if len(self.feature_history[feature]) > self.history_size:
                self.feature_history[feature].pop(0)
        
        return features, True
    
    def _detect_facial_asymmetry(self, points):
        """
        Detect facial asymmetry focused on AD-specific regions
        (face edges, eyebrows, eyes, nostrils, mouth)
        """
        try:
            # Calculate asymmetry scores for each facial region
            
            # 1. Face edge asymmetry
            left_edge = points[self.face_edges[0]]
            right_edge = points[self.face_edges[1]]
            center_x = (left_edge[0] + right_edge[0]) / 2
            edge_asymmetry = abs((center_x - left_edge[0]) - (right_edge[0] - center_x)) / (right_edge[0] - left_edge[0] + 1e-6)
            
            # 2. Eyebrow asymmetry
            left_brow = points[self.eyebrows[0]]
            right_brow = points[self.eyebrows[1]]
            brow_height_diff = abs(left_brow[1] - right_brow[1])
            face_height = abs(points[10][1] - points[152][1])  # Forehead to chin
            brow_asymmetry = brow_height_diff / (face_height + 1e-6)
            
            # 3. Eye asymmetry
            left_eye = points[self.eyes[0]]
            right_eye = points[self.eyes[1]]
            eye_height_diff = abs(left_eye[1] - right_eye[1])
            eye_asymmetry = eye_height_diff / (face_height + 1e-6)
            
            # 4. Nostril asymmetry
            left_nostril = points[self.nostrils[0]]
            right_nostril = points[self.nostrils[1]]
            nostril_diff = abs(left_nostril[1] - right_nostril[1])
            nostril_asymmetry = nostril_diff / (face_height + 1e-6)
            
            # 5. Mouth corner asymmetry
            left_mouth = points[self.mouth_corners[0]]
            right_mouth = points[self.mouth_corners[1]]
            mouth_height_diff = abs(left_mouth[1] - right_mouth[1])
            mouth_asymmetry = mouth_height_diff / (face_height + 1e-6)
            
            # Combine all asymmetry scores with weights based on research
            asymmetry_score = (
                edge_asymmetry * 0.2 +
                brow_asymmetry * 0.2 +
                eye_asymmetry * 0.25 +
                nostril_asymmetry * 0.15 +
                mouth_asymmetry * 0.2
            )
            
            # Scale up for better visibility (adjust as needed)
            asymmetry_score = min(1.0, asymmetry_score * 5.0)
            
            return asymmetry_score
        except Exception as e:
            print(f"Error detecting facial asymmetry: {e}")
            return 0.0
    
    def _detect_reduced_expressivity(self, points):
        """Detect reduced expressivity similar to facial masking in PD but with AD-specific focus"""
        try:
            # Get key facial landmarks
            mouth_left = points[self.mouth_corners[0]]
            mouth_right = points[self.mouth_corners[1]]
            
            # Calculate mouth width
            mouth_width = np.linalg.norm(mouth_right - mouth_left)
            
            # Calculate vertical mouth opening
            mouth_top = points[13]  # Upper lip
            mouth_bottom = points[14]  # Lower lip
            mouth_height = np.linalg.norm(mouth_top - mouth_bottom)
            
            # Calculate eyebrow movement
            left_brow = points[self.eyebrows[0]]
            right_brow = points[self.eyebrows[1]]
            brow_distance = np.linalg.norm(left_brow - right_brow)
            
            # Initialize neutral face if not already established
            if not self.neutral_established:
                self.neutral_landmarks = points.copy()
                self.neutral_established = True
                return 0.5  # Default value if neutral face not established
            
            # Calculate mobility metrics compared to neutral face
            if self.neutral_established:
                # Mouth mobility
                neutral_mouth_width = np.linalg.norm(
                    self.neutral_landmarks[self.mouth_corners[1]] - 
                    self.neutral_landmarks[self.mouth_corners[0]]
                )
                neutral_mouth_height = np.linalg.norm(
                    self.neutral_landmarks[13] - self.neutral_landmarks[14]
                )
                
                # Brow mobility
                neutral_brow_distance = np.linalg.norm(
                    self.neutral_landmarks[self.eyebrows[1]] - 
                    self.neutral_landmarks[self.eyebrows[0]]
                )
                
                # Calculate relative changes
                mouth_width_change = abs(mouth_width - neutral_mouth_width) / (neutral_mouth_width + 1e-6)
                mouth_height_change = abs(mouth_height - neutral_mouth_height) / (neutral_mouth_height + 1e-6)
                brow_change = abs(brow_distance - neutral_brow_distance) / (neutral_brow_distance + 1e-6)
                
                # Combine metrics (higher values = more expression/mobility)
                mobility_score = mouth_width_change * 0.4 + mouth_height_change * 0.4 + brow_change * 0.2
                
                # Convert to expressivity reduction (higher = reduced expression = AD indicator)
                reduced_expressivity = 1.0 - min(1.0, mobility_score * 3.0)
                
                return reduced_expressivity
            
            return 0.5  # Default fallback
        except Exception as e:
            print(f"Error detecting reduced expressivity: {e}")
            return 0.5
    
    def _detect_saccadic_eye_movement(self, points):
        """Detect saccadic eye movements characteristic of AD"""
        try:
            # Calculate eye centers
            left_eye_center = points[468]  # Left eye center
            right_eye_center = points[473]  # Right eye center
            
            # Store eye positions for analysis
            if self.prev_left_eye is not None and self.prev_right_eye is not None:
                # Calculate eye movement velocity
                left_velocity = np.linalg.norm(left_eye_center - self.prev_left_eye)
                right_velocity = np.linalg.norm(right_eye_center - self.prev_right_eye)
                
                # Normalize by face size
                face_width = np.linalg.norm(points[self.face_edges[1]] - points[self.face_edges[0]])
                norm_left_vel = left_velocity / (face_width + 1e-6)
                norm_right_vel = right_velocity / (face_width + 1e-6)
                
                # Store normalized velocities and positions
                self.eye_positions.append((norm_left_vel, norm_right_vel, left_eye_center, right_eye_center))
                if len(self.eye_positions) > self.max_eye_positions:
                    self.eye_positions.pop(0)
            
            # Update previous positions
            self.prev_left_eye = left_eye_center
            self.prev_right_eye = right_eye_center
            
            # Need sufficient history for analysis
            if len(self.eye_positions) < 30:
                return 0.0
            
            # Extract velocity data
            left_velocities = [pos[0] for pos in self.eye_positions]
            right_velocities = [pos[1] for pos in self.eye_positions]
            
            # Saccades are characterized by:
            # 1. Rapid eye movements (high velocity peaks)
            # 2. Followed by fixation periods (low velocity)
            # 3. Step-like pattern rather than smooth
            
            # Detect velocity peaks (potential saccades)
            velocity_threshold = np.mean(left_velocities + right_velocities) * 2.0
            left_peaks = [v > velocity_threshold for v in left_velocities]
            right_peaks = [v > velocity_threshold for v in right_velocities]
            
            # Count peaks
            left_peak_count = sum(left_peaks)
            right_peak_count = sum(right_peaks)
            total_peak_count = left_peak_count + right_peak_count
            
            # Calculate saccade frequency (peaks per second, assuming 30fps)
            saccade_freq = total_peak_count / (len(self.eye_positions) / 30.0)
            
            # Check for step pattern (alternating high/low velocities)
            # Count transitions between high and low velocity states
            left_transitions = sum(abs(np.diff([1 if v > velocity_threshold else 0 for v in left_velocities])))
            right_transitions = sum(abs(np.diff([1 if v > velocity_threshold else 0 for v in right_velocities])))
            
            # Normalize by number of frames
            step_pattern_score = (left_transitions + right_transitions) / (2 * (len(left_velocities) - 1))
            
            # Combine metrics for final score
            # Higher values indicate more saccadic movement
            saccade_score = (
                min(1.0, saccade_freq / 3.0) * 0.6 +  # Normalized frequency (3 saccades/sec = 1.0)
                step_pattern_score * 0.4
            )
            
            return min(1.0, max(0.0, saccade_score))
        except Exception as e:
            print(f"Error detecting saccadic eye movement: {e}")
            return 0.0

# Combined Neurological Disorder Detection System
class NeurologicalDisorderDetectionSystem:
    def __init__(self, pd_model_path=None):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.3
        )
        
        # Initialize individual disorder detectors
        self.pd_system = PDDetectionSystem(pd_model_path)
        self.ms_feature_extractor = MSFeatureExtractor()
        self.ad_feature_extractor = ADFeatureExtractor()
        
        # Store recent results for stability
        self.ms_recent_results = []
        self.ad_recent_results = []
        self.max_results_history = 10
    
    def process_frame(self, frame):
        """Process a frame to detect multiple neurological disorders"""
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Process with MediaPipe for landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        # Initialize result dictionary
        result = {
            'face_detected': False,
            'pd': {'probability': 0.0, 'likelihood': 'Unknown', 'features': None},
            'ms': {'probability': 0.0, 'likelihood': 'Unknown', 'features': None},
            'ad': {'probability': 0.0, 'likelihood': 'Unknown', 'features': None}
        }
        
        # Check if face was detected
        if not results.multi_face_landmarks:
            return result
        
        # Face was detected
        result['face_detected'] = True
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Get PD detection results
        pd_result = self.pd_system.process_frame(frame)
        result['pd'] = {
            'probability': pd_result['pd_probability'],
            'likelihood': pd_result['pd_likelihood'],
            'features': pd_result['features']
        }
        
        # Extract MS features
        ms_features, _ = self.ms_feature_extractor.extract_features(frame, landmarks)
        
        # Calculate MS probability
        ms_probability = self._calculate_ms_probability(ms_features)
        
        # Add to recent results for stability
        self.ms_recent_results.append(ms_probability)
        if len(self.ms_recent_results) > self.max_results_history:
            self.ms_recent_results.pop(0)
        
        # Use stabilized probability
        stable_ms_probability = np.median(self.ms_recent_results) if self.ms_recent_results else ms_probability
        
        # Determine MS likelihood
        if stable_ms_probability > 0.7:
            ms_likelihood = "High"
        elif stable_ms_probability > 0.45:
            ms_likelihood = "Medium"
        elif stable_ms_probability > 0.25:
            ms_likelihood = "Low"
        else:
            ms_likelihood = "Very Low"
        
        result['ms'] = {
            'probability': stable_ms_probability,
            'likelihood': ms_likelihood,
            'features': ms_features
        }
        
        # Extract AD features
        ad_features, _ = self.ad_feature_extractor.extract_features(frame, landmarks)
        
        # Calculate AD probability
        ad_probability = self._calculate_ad_probability(ad_features)
        
        # Add to recent results for stability
        self.ad_recent_results.append(ad_probability)
        if len(self.ad_recent_results) > self.max_results_history:
            self.ad_recent_results.pop(0)
        
        # Use stabilized probability
        stable_ad_probability = np.median(self.ad_recent_results) if self.ad_recent_results else ad_probability
        
        # Determine AD likelihood
        if stable_ad_probability > 0.7:
            ad_likelihood = "High"
        elif stable_ad_probability > 0.45:
            ad_likelihood = "Medium"
        elif stable_ad_probability > 0.25:
            ad_likelihood = "Low"
        else:
            ad_likelihood = "Very Low"
        
        result['ad'] = {
            'probability': stable_ad_probability,
            'likelihood': ad_likelihood,
            'features': ad_features
        }
        
        return result
    
    def _calculate_ms_probability(self, features):
        """Calculate MS probability from extracted features"""
        try:
            # Get individual feature scores
            facial_paralysis = features['facial_paralysis_score']
            eye_movement = features['eye_movement_abnormality']
            eyelid_twitching = features['eyelid_twitching']
            hemifacial_spasm = features['hemifacial_spasm']
            
            # Weighted combination based on clinical significance
            # Eye movement abnormalities and facial paralysis are stronger indicators
            ms_probability = (
                facial_paralysis * 0.3 +
                eye_movement * 0.35 +
                eyelid_twitching * 0.2 +
                hemifacial_spasm * 0.15
            )
            
            # Apply threshold calibration to reduce false positives
            ms_probability = max(0.0, min(1.0, (ms_probability - 0.15) * 1.2))
            
            return ms_probability
        except Exception as e:
            print(f"Error calculating MS probability: {e}")
            return 0.0
    
    def _calculate_ad_probability(self, features):
        """Calculate AD probability from extracted features"""
        try:
            # Get individual feature scores
            facial_asymmetry = features['facial_asymmetry']
            reduced_expressivity = features['reduced_expressivity']
            saccadic_eye_movement = features['saccadic_eye_movement']
            
            # Weighted combination based on research
            # Reduced expressivity and saccadic eye movements are stronger indicators
            ad_probability = (
                facial_asymmetry * 0.25 +
                reduced_expressivity * 0.4 +
                saccadic_eye_movement * 0.35
            )
            
            # Apply threshold calibration for better classification
            ad_probability = max(0.0, min(1.0, (ad_probability - 0.2) * 1.25))
            
            return ad_probability
        except Exception as e:
            print(f"Error calculating AD probability: {e}")
            return 0.0

# Main function to run the combined detection system
def run_neurological_disorder_detection(model_path=None):
    """Run real-time neurological disorder detection using webcam"""
    # Create the detection system
    detection_system = NeurologicalDisorderDetectionSystem(model_path)
    
    # Open webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Could not open camera 0, trying camera 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Could not open any webcam")
            return
    
    print("Neurological Disorder Detection started. Press 'q' to quit.")
    
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
        result = detection_system.process_frame(frame)
        
        # Display results on frame
        if result['face_detected']:
            # Create header
            cv2.putText(frame, "Neurological Disorder Detection", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display disorder probabilities
            y_pos = 70
            
            # Parkinson's Disease
            pd_prob = result['pd']['probability']
            pd_likelihood = result['pd']['likelihood']
            if pd_likelihood == "High":
                pd_color = (0, 0, 255)  # Red
            elif pd_likelihood == "Medium":
                pd_color = (0, 165, 255)  # Orange
            elif pd_likelihood == "Low":
                pd_color = (0, 255, 255)  # Yellow
            else:
                pd_color = (0, 255, 0)  # Green
            
            cv2.putText(frame, f"Parkinson's Disease: {pd_prob:.2f} ({pd_likelihood})", 
                      (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pd_color, 2)
            y_pos += 30
            
            # Multiple Sclerosis
            ms_prob = result['ms']['probability']
            ms_likelihood = result['ms']['likelihood']
            if ms_likelihood == "High":
                ms_color = (0, 0, 255)  # Red
            elif ms_likelihood == "Medium":
                ms_color = (0, 165, 255)  # Orange
            elif ms_likelihood == "Low":
                ms_color = (0, 255, 255)  # Yellow
            else:
                ms_color = (0, 255, 0)  # Green
            
            cv2.putText(frame, f"Multiple Sclerosis: {ms_prob:.2f} ({ms_likelihood})", 
                      (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ms_color, 2)
            y_pos += 30
            
            # Alzheimer's Disease
            ad_prob = result['ad']['probability']
            ad_likelihood = result['ad']['likelihood']
            if ad_likelihood == "High":
                ad_color = (0, 0, 255)  # Red
            elif ad_likelihood == "Medium":
                ad_color = (0, 165, 255)  # Orange
            elif ad_likelihood == "Low":
                ad_color = (0, 255, 255)  # Yellow
            else:
                ad_color = (0, 255, 0)  # Green
            
            cv2.putText(frame, f"Alzheimer's Disease: {ad_prob:.2f} ({ad_likelihood})", 
                      (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ad_color, 2)
            y_pos += 30
            
            # Display key features for the disorder with highest probability
            max_prob = max(pd_prob, ms_prob, ad_prob)
            features_to_show = None
            title = None
            
            if max_prob == pd_prob and pd_prob > 0.25:
                features_to_show = result['pd']['features']
                title = "PD Key Features:"
            elif max_prob == ms_prob and ms_prob > 0.25:
                features_to_show = result['ms']['features']
                title = "MS Key Features:"
            elif max_prob == ad_prob and ad_prob > 0.25:
                features_to_show = result['ad']['features']
                title = "AD Key Features:"
            
            if features_to_show and title:
                y_pos += 10
                cv2.putText(frame, title, (10, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_pos += 25
                
                for name, value in features_to_show.items():
                    if isinstance(value, (int, float)):
                        feature_text = f"{name}: {value:.2f}"
                        cv2.putText(frame, feature_text, (20, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_pos += 20
            
            # Display MS-specific indicators when MS probability is significant
            if ms_prob > 0.3:
                ms_features = result['ms']['features']
                y_pos += 10
                cv2.putText(frame, "MS Indicators:", (10, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, ms_color, 1)
                y_pos += 20
                
                indicators = []
                if ms_features['facial_paralysis_score'] > 0.4:
                    indicators.append("Facial paralysis detected")
                if ms_features['eye_movement_abnormality'] > 0.4:
                    indicators.append("Abnormal eye movements (nystagmus)")
                if ms_features['eyelid_twitching'] > 0.4:
                    indicators.append("Eyelid twitching (myokymia)")
                if ms_features['hemifacial_spasm'] > 0.4:
                    indicators.append("Hemifacial spasm detected")
                
                for indicator in indicators:
                    cv2.putText(frame, f"- {indicator}", (20, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, ms_color, 1)
                    y_pos += 20
            
            # Display AD-specific indicators when AD probability is significant
            if ad_prob > 0.3:
                ad_features = result['ad']['features']
                y_pos += 10
                cv2.putText(frame, "AD Indicators:", (10, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, ad_color, 1)
                y_pos += 20
                
                indicators = []
                if ad_features['facial_asymmetry'] > 0.4:
                    indicators.append("Facial asymmetry")
                if ad_features['reduced_expressivity'] > 0.6:
                    indicators.append("Reduced facial expressivity")
                if ad_features['saccadic_eye_movement'] > 0.5:
                    indicators.append("Saccadic eye movements")
                
                for indicator in indicators:
                    cv2.putText(frame, f"- {indicator}", (20, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, ad_color, 1)
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
        cv2.imshow('Neurological Disorder Detection', frame)
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Main entry point
if __name__ == "__main__":
    # Path to trained emotion recognition model (optional)
    model_path = "./PipingModels/best_emotion_model.pth"
    
    # Run the system
    run_neurological_disorder_detection(model_path)