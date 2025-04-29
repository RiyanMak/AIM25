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
import threading
from collections import deque
import argparse
import matplotlib.pyplot as plt
import csv

# ----------------------------------------------------------------------
# PART 1: MODEL DEFINITIONS AND FEATURE EXTRACTORS
# ----------------------------------------------------------------------

# CNN Model for Emotion Recognition
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

# Enhanced model for RAF-DB dataset integration
class RAFDBEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(RAFDBEmotionCNN, self).__init__()
        # Using deeper architecture for RAF-DB which has more complex expressions
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # RGB input (3 channels)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
        # Calculate final size based on input 100x100 RGB images
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize MediaPipe for Facial Landmark Detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.3,
    refine_landmarks=True)  # Enable refined landmarks for better eye tracking

# Enhanced Facial Feature Extractor
class FacialFeatureExtractor:
    def __init__(self, history_size=60):
        # Base features for all disorders
        self.key_features = [
            # Parkinson's features
            'mask_face_score', 'smile_amplitude', 'smile_symmetry', 'facial_mobility',
            # Alzheimer's features
            'attention_score', 'emotion_response_delay', 'expression_spatial_organization',
            'expression_intensity',  # Added for increased expressivity detection
            # MS features
            'facial_symmetry', 'blink_regularity', 'eye_movement_abnormalities',
            'facial_weakness', 'pupil_misalignment', 'unilateral_twitching'  # Added MS-specific features
        ]
        self.feature_history = {feature: [] for feature in self.key_features}
        self.history_size = history_size
        self.neutral_landmarks = None
        self.neutral_established = False
        self.previous_landmarks = None
        self.last_frame_time = None
        
        # Define landmark indices for specific facial regions
        self.mouth_corners = [61, 291]  # Left, Right
        self.mouth_vertical = [13, 14]  # Top, Bottom
        self.eyebrows = [105, 334]      # Left, Right
        self.eyes = [159, 386]          # Left, Right
        self.cheeks = [117, 346]        # Left, Right
        self.forehead = [10]
        
        # Eyes for attention tracking
        self.left_eye = [33, 133, 160, 159, 158, 153, 145, 144]
        self.right_eye = [362, 263, 386, 385, 384, 381, 374, 373]
        self.iris_left = [468, 469, 470, 471, 472]
        self.iris_right = [473, 474, 475, 476, 477]
        
        # Pupil centers for misalignment detection
        self.left_pupil = [473]  # Center of left iris
        self.right_pupil = [468]  # Center of right iris
        
        # Lips for speech analysis
        self.upper_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        self.lower_lip = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        
        # Left side and right side points for unilateral analysis
        self.left_side = [61, 97, 117, 234, 34, 139, 135, 169, 170, 140]  # Left side facial points
        self.right_side = [291, 326, 346, 454, 264, 362, 367, 397, 398, 369]  # Right side facial points
        
        # Timing for attention and reaction analysis
        self.blink_times = []
        self.emotion_timestamps = []
        self.fixation_points = []
        self.twitching_events = {'left': [], 'right': []}  # Track twitching events per side

    def reset(self):
        """Reset the feature extractor for a new session"""
        self.feature_history = {feature: [] for feature in self.key_features}
        self.neutral_landmarks = None
        self.neutral_established = False
        self.previous_landmarks = None
        self.blink_times = []
        self.emotion_timestamps = []
        self.fixation_points = []
        self.twitching_events = {'left': [], 'right': []}

    def extract_features(self, image, current_time=None):
        """Extract comprehensive facial features for neurological disorder detection"""
        if current_time is None:
            current_time = time.time()
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb_image.shape[:2]
        results = face_mesh.process(rgb_image)
        
        # Initialize features with default values
        features = {f: 0.0 for f in self.key_features}
        features.update({
            'left_eye_openness': 0.0,
            'right_eye_openness': 0.0,
            'left_mouth_corner_pos': None,
            'right_mouth_corner_pos': None,
            'reaction_time_ms': 0.0,
            'pupil_misalignment': 0.0,
            'unilateral_twitching': 0.0,
            'expression_intensity': 0.0
        })

        # Calculate time delta
        if self.last_frame_time is not None:
            time_delta = current_time - self.last_frame_time
        else:
            time_delta = 0.033  # Assume ~30 FPS if first frame
            
        self.last_frame_time = current_time

        if not results.multi_face_landmarks:
            return features, False

        landmarks = results.multi_face_landmarks[0].landmark
        
        try:
            # Convert landmarks to NumPy array
            points = np.array([(lm.x * w, lm.y * h, lm.z) for lm in landmarks])

            # Establish neutral face if not already done
            if not self.neutral_established:
                self.neutral_landmarks = points.copy()
                self.neutral_established = True
                self.previous_landmarks = points.copy()
                return features, True
                
            # ------ PD-Specific Feature Extraction ------
            
            # Calculate mouth geometry
            mouth_left = points[self.mouth_corners[0]]
            mouth_right = points[self.mouth_corners[1]]
            mouth_top = points[self.mouth_vertical[0]]
            mouth_bottom = points[self.mouth_vertical[1]]
            
            # Store for UI display
            features['left_mouth_corner_pos'] = mouth_left[:2]
            features['right_mouth_corner_pos'] = mouth_right[:2]

            # Calculate mouth measurements
            mouth_width = np.linalg.norm(mouth_right[:2] - mouth_left[:2])
            mouth_height = np.linalg.norm(mouth_top[:2] - mouth_bottom[:2])
            
            # Normalize mouth width by face width for smile amplitude
            face_width = np.linalg.norm(points[self.cheeks[0]][:2] - points[self.cheeks[1]][:2])
            if face_width > 0:
                smile_measure = mouth_width / face_width
                features['smile_amplitude'] = min(1.0, smile_measure * 1.5)
            
            # Calculate smile symmetry (asymmetry)
            # Enhanced calculation matching the second code file
            nose_tip = points[1]
            
            # Get more points around the mouth for better analysis
            left_smile_region = [
                self.mouth_corners[0],     # Main corner
                78,                        # Above left corner
                95,                        # Below left corner
                191                        # Additional point near left corner
            ]
            
            right_smile_region = [
                self.mouth_corners[1],     # Main corner
                308,                       # Above right corner
                325,                       # Below right corner
                410                        # Additional point near right corner
            ]
            
            # Calculate average position of smile corner regions
            left_smile_points = np.array([points[idx] for idx in left_smile_region])
            right_smile_points = np.array([points[idx] for idx in right_smile_region])
            
            left_smile_center = np.mean(left_smile_points, axis=0)
            right_smile_center = np.mean(right_smile_points, axis=0)
            
            # Get mouth center position
            mouth_center = (mouth_top + mouth_bottom) / 2
            
            # Calculate vertical distances of smile corners from mouth center
            left_corner_height = mouth_center[1] - left_smile_center[1]
            right_corner_height = mouth_center[1] - right_smile_center[1]
            
            # Calculate asymmetry as the difference in heights
            height_difference = abs(left_corner_height - right_corner_height)
            normalized_asymmetry = height_difference / (mouth_width + 1e-6)
            
            # Calculate lip corner angles relative to mouth center
            left_angle = np.arctan2(left_smile_center[1] - mouth_center[1], 
                                   left_smile_center[0] - mouth_center[0])
            right_angle = np.arctan2(right_smile_center[1] - mouth_center[1], 
                                    right_smile_center[0] - mouth_center[0])
            
            # Angle difference (in radians)
            angle_asymmetry = abs(abs(left_angle) - abs(right_angle))
            
            # Combine height and angle measurements for final asymmetry score
            # Scale up to make small asymmetries more noticeable
            features['smile_symmetry'] = min(1.0, (normalized_asymmetry * 5.0 + angle_asymmetry * 2.0) / 2.0)

            # Calculate facial mobility by comparing to neutral face
            if self.neutral_established:
                # Define key movement points
                movement_points = [
                    self.mouth_corners,
                    self.mouth_vertical,
                    self.eyebrows,
                    self.eyes
                ]
                
                mobility_scores = []
                
                for point_group in movement_points:
                    for idx in point_group:
                        # Calculate displacement from neutral position
                        neutral_pos = self.neutral_landmarks[idx][:2]
                        current_pos = points[idx][:2]
                        displacement = np.linalg.norm(current_pos - neutral_pos)
                        
                        # Normalize by face size
                        normalized_displacement = displacement / face_width
                        mobility_scores.append(normalized_displacement)
                
                # Overall facial mobility score (higher = more mobile)
                if mobility_scores:
                    features['facial_mobility'] = min(1.0, np.mean(mobility_scores) * 20)
                    
                    # Calculate mask face score (inverse of mobility and expression)
                    features['mask_face_score'] = max(0.0, min(1.0, 
                        0.3 * (1.0 - features['facial_mobility']) + 
                        0.3 * (1.0 - min(1.0, features['smile_amplitude'] * 2)) + 
                        0.4 * features['smile_symmetry']
                    ))
                    
                    # Calculate expression intensity (for Alzheimer's increased expressivity)
                    # Higher values indicate increased (potentially exaggerated) expressions
                    if features['facial_mobility'] > 0.5:  # If mobility is above average
                        features['expression_intensity'] = min(1.0, features['facial_mobility'] * 1.5) 
                    else:
                        features['expression_intensity'] = 0.0
            
            # ------ Alzheimer's-Specific Feature Extraction ------
            
            # Eye openness calculation (for attention and blink analysis)
            def calculate_eye_openness(eye_indices):
                if len(eye_indices) < 2:
                    return 0.0
                    
                # Split into upper and lower eye points
                mid_point = len(eye_indices) // 2
                upper_indices = eye_indices[:mid_point]
                lower_indices = eye_indices[mid_point:]
                
                # Calculate average points
                upper_point = np.mean([points[p][:2] for p in upper_indices], axis=0)
                lower_point = np.mean([points[p][:2] for p in lower_indices], axis=0)
                
                # Calculate eye height
                eye_height = np.linalg.norm(upper_point - lower_point)
                
                # Normalize by face size
                if face_width > 0:
                    normalized_openness = eye_height / face_width
                    return normalized_openness * 10  # Scale appropriately
                
                return 0.0
            
            # Calculate eye openness
            left_eye_openness = calculate_eye_openness(self.left_eye)
            right_eye_openness = calculate_eye_openness(self.right_eye)
            features['left_eye_openness'] = left_eye_openness
            features['right_eye_openness'] = right_eye_openness
            
            # Detect blinks for blink regularity analysis
            blink_threshold = 0.2
            current_blink_state = ((left_eye_openness + right_eye_openness) / 2) < blink_threshold
            
            if self.previous_landmarks is not None:
                # Calculate previous eye openness
                prev_left_openness = calculate_eye_openness(self.left_eye)
                prev_right_openness = calculate_eye_openness(self.right_eye)
                prev_blink_state = ((prev_left_openness + prev_right_openness) / 2) < blink_threshold
                
                # Detect blink onset (transition from open to closed)
                if not prev_blink_state and current_blink_state:
                    self.blink_times.append(current_time)
                    
                    # Keep only recent blinks for analysis
                    if len(self.blink_times) > 10:
                        self.blink_times.pop(0)
            
            # Calculate blink regularity (irregular blinks can indicate MS)
            if len(self.blink_times) >= 3:
                blink_intervals = np.diff(self.blink_times)
                if len(blink_intervals) > 0:
                    mean_interval = np.mean(blink_intervals)
                    if mean_interval > 0:
                        blink_interval_variation = np.std(blink_intervals) / mean_interval
                        features['blink_regularity'] = min(1.0, blink_interval_variation * 2)
            else:
                features['blink_regularity'] = 0.5  # Neutral value if not enough data
            
            # Calculate attention score based on eye fixation
            if self.previous_landmarks is not None:
                # Calculate center points of both irises
                left_iris_center = np.mean([points[p][:2] for p in self.iris_left], axis=0)
                right_iris_center = np.mean([points[p][:2] for p in self.iris_right], axis=0)
                
                # Add current fixation point for attention analysis
                fixation_point = (left_iris_center + right_iris_center) / 2
                self.fixation_points.append((fixation_point, current_time))
                
                # Keep only recent fixations
                while len(self.fixation_points) > 30:
                    self.fixation_points.pop(0)
                
                # Calculate attention score based on fixation stability
                if len(self.fixation_points) > 5:
                    # Get recent fixation points
                    recent_points = np.array([p[0] for p in self.fixation_points[-10:]])
                    
                    # Calculate spatial variance of fixation (higher = more wandering = less attention)
                    fixation_variance = np.mean(np.var(recent_points, axis=0))
                    
                    # Normalize by face width
                    normalized_variance = min(1.0, fixation_variance / (face_width * 0.01))
                    
                    # Calculate attention score (lower variance = better attention)
                    features['attention_score'] = max(0.0, 1.0 - normalized_variance)
                else:
                    features['attention_score'] = 0.5  # Neutral if not enough fixation data
                    
                # Detect pupil misalignment (double vision indicator for MS)
                # Improved calculation with more strict thresholds
                if self.left_pupil and self.right_pupil:
                    # Get pupil centers 
                    left_pupil_center = points[self.left_pupil[0]][:2]
                    right_pupil_center = points[self.right_pupil[0]][:2]
                    
                    # Calculate relative positions to eye centers
                    left_eye_center = np.mean([points[p][:2] for p in self.left_eye], axis=0)
                    right_eye_center = np.mean([points[p][:2] for p in self.right_eye], axis=0)
                    
                    # Calculate normalized vectors from eye center to pupil
                    # Ensure we avoid division by zero
                    left_eye_size = np.linalg.norm(left_eye_center)
                    right_eye_size = np.linalg.norm(right_eye_center)
                    
                    if left_eye_size > 0 and right_eye_size > 0:
                        left_gaze_vector = (left_pupil_center - left_eye_center) / left_eye_size
                        right_gaze_vector = (right_pupil_center - right_eye_center) / right_eye_size
                        
                        # Calculate gaze alignment (higher value = more misalignment)
                        gaze_dot_product = np.dot(left_gaze_vector, right_gaze_vector)
                        
                        # Ensure value is in valid range for arccos
                        gaze_dot_product = max(-1.0, min(1.0, gaze_dot_product))
                        
                        gaze_angle = np.arccos(gaze_dot_product) * 180 / np.pi
                        
                        # Normalize to 0-1 range with higher threshold to reduce false positives
                        # Only consider significant misalignment (>10 degrees)
                        if gaze_angle > 10:
                            features['pupil_misalignment'] = min(1.0, (gaze_angle - 10) / 20.0)
                        else:
                            features['pupil_misalignment'] = 0.0
                    else:
                        features['pupil_misalignment'] = 0.0
            
            # Measure expression spatial organization (important for Alzheimer's detection)
            if self.neutral_established and self.previous_landmarks is not None:
                # Calculate displacement vectors from neutral for key expression regions
                regions = [
                    self.mouth_corners,  # Mouth corners
                    self.eyebrows,       # Eyebrows
                    self.cheeks          # Cheeks
                ]
                
                displacement_vectors = []
                for region in regions:
                    for idx in region:
                        neutral_pos = self.neutral_landmarks[idx][:2]
                        current_pos = points[idx][:2]
                        displacement = current_pos - neutral_pos
                        displacement_vectors.append(displacement)
                
                # Calculate spatial coherence through correlation of movements
                if len(displacement_vectors) >= 2:
                    coherence_scores = []
                    for i in range(len(displacement_vectors)):
                        for j in range(i+1, len(displacement_vectors)):
                            v1 = displacement_vectors[i]
                            v2 = displacement_vectors[j]
                            
                            # Calculate vector similarity (dot product normalized)
                            dot_product = np.dot(v1, v2)
                            magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
                            
                            if magnitude_product > 1e-6:
                                similarity = abs(dot_product / magnitude_product)
                                coherence_scores.append(similarity)
                    
                    # Average spatial coherence (higher = more organized expressions)
                    if coherence_scores:
                        avg_coherence = np.mean(coherence_scores)
                        features['expression_spatial_organization'] = avg_coherence
                    else:
                        features['expression_spatial_organization'] = 0.5
                else:
                    features['expression_spatial_organization'] = 0.5
            
            # Track emotion response timing (for Alzheimer's detection)
            if self.previous_landmarks is not None:
                # Define key expression points for change detection
                key_expression_points = []
                for idx in (self.mouth_corners + self.eyebrows + [self.forehead[0]]):
                    key_expression_points.append(idx)
                
                # Continue only if we have valid expression points
                if key_expression_points:
                    expression_change = 0
                    for idx in key_expression_points:
                        prev_pos = self.previous_landmarks[idx][:2]
                        curr_pos = points[idx][:2]
                        expression_change += np.linalg.norm(curr_pos - prev_pos)
                    
                    # Normalize by face width and number of points
                    if key_expression_points:
                        expression_change_normalized = expression_change / (face_width * len(key_expression_points))
                        
                        # If significant change detected, consider it a new emotion onset
                        emotion_change_threshold = 0.005
                        if expression_change_normalized > emotion_change_threshold:
                            self.emotion_timestamps.append(current_time)
                            
                            # Keep only recent timestamps
                            while len(self.emotion_timestamps) > 10:
                                self.emotion_timestamps.pop(0)
                            
                            # Calculate response delay if we have multiple timestamps
                            if len(self.emotion_timestamps) >= 2:
                                # Calculate average response time between emotional changes
                                response_times = np.diff(self.emotion_timestamps)
                                avg_response_time = np.mean(response_times)
                                features['reaction_time_ms'] = avg_response_time * 1000  # Convert to ms
                                
                                # Normalize to 0-1 scale (higher value = longer delay = more concerning)
                                features['emotion_response_delay'] = min(1.0, 
                                    max(0.0, (avg_response_time * 1000 - 400) / 1000))
                            else:
                                features['emotion_response_delay'] = 0.0  # Default if not enough data
            
            # ------ MS-Specific Feature Extraction ------
            
            # Calculate overall facial symmetry for MS detection
            paired_landmarks = [
                (61, 291),   # Mouth corners
                (101, 331),  # Cheeks
                (33, 263),   # Eyes outer corners
                (133, 362),  # Eyes inner corners
                (70, 300),   # Upper lip
                (105, 334),  # Eyebrows
            ]
            
            asymmetry_scores = []
            for left_idx, right_idx in paired_landmarks:
                left_point = points[left_idx][:2]
                right_point = points[right_idx][:2]
                
                # Get distances from midline (nose bridge)
                midline_idx = 168  # Nose bridge point
                midline = points[midline_idx][:2]
                left_dist = np.linalg.norm(left_point - midline)
                right_dist = np.linalg.norm(right_point - midline)
                
                # Calculate asymmetry ratio
                pair_asymmetry = abs(left_dist - right_dist) / (max(left_dist, right_dist) + 1e-6)
                asymmetry_scores.append(pair_asymmetry)
            
            # Overall facial symmetry (higher value = more asymmetry)
            # Increased threshold to reduce false positives
            if asymmetry_scores:
                raw_asymmetry = np.mean(asymmetry_scores) * 5
                # Only consider significant asymmetry
                if raw_asymmetry > 0.3:
                    features['facial_symmetry'] = min(1.0, raw_asymmetry)
                else:
                    features['facial_symmetry'] = 0.0
            
            # Iris movement analysis for eye movement abnormalities (MS)
            if self.previous_landmarks is not None:
                # Calculate current iris positions
                current_left_iris = np.mean([points[p][:2] for p in self.iris_left], axis=0)
                current_right_iris = np.mean([points[p][:2] for p in self.iris_right], axis=0)
                
                # Calculate previous iris positions
                prev_left_iris = np.mean([self.previous_landmarks[p][:2] for p in self.iris_left], axis=0)
                prev_right_iris = np.mean([self.previous_landmarks[p][:2] for p in self.iris_right], axis=0)
                
                # Calculate velocities
                if time_delta > 0:
                    # Velocity of iris movement
                    left_velocity = np.linalg.norm(current_left_iris - prev_left_iris) / time_delta
                    right_velocity = np.linalg.norm(current_right_iris - prev_right_iris) / time_delta
                    
                    # Normalized by face width
                    left_velocity_norm = left_velocity / face_width
                    right_velocity_norm = right_velocity / face_width
                    
                    # Detect nystagmus-like rapid movements (common in MS)
                    # Increased threshold to reduce false positives
                    nystagmus_threshold = 0.03  # Higher threshold than before
                    left_rapid = left_velocity_norm > nystagmus_threshold
                    right_rapid = right_velocity_norm > nystagmus_threshold
                    
                    # Count consistent rapid movements over multiple frames
                    rapid_movement_counter = 0
                    if left_rapid or right_rapid:
                        rapid_movement_counter += 1
                    
                    # Detect internuclear ophthalmoplegia (different velocities in each eye - MS indicator)
                    max_vel = max(left_velocity_norm, right_velocity_norm)
                    if max_vel > 0:
                        ino_indicator = abs(left_velocity_norm - right_velocity_norm) / max_vel
                    else:
                        ino_indicator = 0.0
                    
                    # Combine indicators for eye movement abnormalities
                    # Higher threshold to reduce false positives
                    if rapid_movement_counter > 0 or ino_indicator > 0.5:
                        features['eye_movement_abnormalities'] = min(1.0, 
                            (float(left_rapid) * 0.3) + 
                            (float(right_rapid) * 0.3) + 
                            (ino_indicator * 0.4)
                        )
                    else:
                        features['eye_movement_abnormalities'] = 0.0
            
            # Detect unilateral facial twitching (MS symptom)
            # Improved to reduce false positives
            if self.previous_landmarks is not None:
                # Calculate movement for left and right sides separately
                left_movement = []
                right_movement = []
                
                # Calculate movement for each side
                for idx in self.left_side:
                    prev_pos = self.previous_landmarks[idx][:2]
                    curr_pos = points[idx][:2]
                    left_movement.append(np.linalg.norm(curr_pos - prev_pos))
                
                for idx in self.right_side:
                    prev_pos = self.previous_landmarks[idx][:2]
                    curr_pos = points[idx][:2]
                    right_movement.append(np.linalg.norm(curr_pos - prev_pos))
                
                # Calculate statistics
                avg_left_movement = np.mean(left_movement) if left_movement else 0
                avg_right_movement = np.mean(right_movement) if right_movement else 0
                
                # Normalize by face width
                avg_left_movement_norm = avg_left_movement / face_width
                avg_right_movement_norm = avg_right_movement / face_width
                
                # Calculate movement asymmetry ratio
                # Increased threshold for detecting twitches
                twitch_threshold = 0.015  # Higher threshold to reduce false positives
                
                # Record twitching events with stronger side imbalance requirement
                if avg_left_movement_norm > twitch_threshold and avg_left_movement_norm > 3 * avg_right_movement_norm:
                    self.twitching_events['left'].append(current_time)
                elif avg_right_movement_norm > twitch_threshold and avg_right_movement_norm > 3 * avg_left_movement_norm:
                    self.twitching_events['right'].append(current_time)
                
                # Keep only recent events
                recent_cutoff = current_time - 3.0  # Last 3 seconds
                self.twitching_events['left'] = [t for t in self.twitching_events['left'] if t > recent_cutoff]
                self.twitching_events['right'] = [t for t in self.twitching_events['right'] if t > recent_cutoff]
                
                # Calculate unilateral twitching score
                left_twitch_count = len(self.twitching_events['left'])
                right_twitch_count = len(self.twitching_events['right'])
                
                # Higher score if twitches are concentrated on one side
                # Only consider significant twitching (multiple events)
                max_twitches = max(left_twitch_count, right_twitch_count)
                min_twitches = min(left_twitch_count, right_twitch_count)
                
                # Require at least 2 twitches to reduce false positives
                if max_twitches >= 2:
                    twitch_ratio = 1.0 - (min_twitches / max_twitches) if max_twitches > 0 else 0
                    features['unilateral_twitching'] = min(1.0, (max_twitches / 4.0) * twitch_ratio)
                else:
                    features['unilateral_twitching'] = 0.0
            
            # Calculate facial weakness indicator (important for MS)
            if self.previous_landmarks is not None:
                # Define left and right side landmarks
                left_landmarks = [61, 97, 117, 234, 34, 139]  # Left side
                right_landmarks = [291, 326, 346, 454, 264, 362]  # Right side
                
                # Calculate movement magnitude on both sides of face
                left_movement = 0
                right_movement = 0
                
                # Calculate movement magnitude for each side
                for idx in left_landmarks:
                    prev_pos = self.previous_landmarks[idx][:2]
                    curr_pos = points[idx][:2]
                    left_movement += np.linalg.norm(curr_pos - prev_pos)
                
                for idx in right_landmarks:
                    prev_pos = self.previous_landmarks[idx][:2]
                    curr_pos = points[idx][:2]
                    right_movement += np.linalg.norm(curr_pos - prev_pos)
                
                # Normalize by number of landmarks
                left_movement /= len(left_landmarks)
                right_movement /= len(right_landmarks)
                
                # Scale by face width
                left_movement_normalized = left_movement / face_width
                right_movement_normalized = right_movement / face_width
                
                # Global weakness - combines both sides
                global_movement = (left_movement_normalized + right_movement_normalized) / 2
                global_weakness = max(0.0, 1.0 - (global_movement * 50))  # Scale appropriately
                
                # Asymmetrical weakness - difference between sides
                side_difference = abs(left_movement_normalized - right_movement_normalized)
                
                # Higher threshold for asymmetry to reduce false positives
                if side_difference > 0.01:  # Only consider significant side differences
                    asymmetrical_weakness = min(1.0, side_difference * 50)  # Scale appropriately
                else:
                    asymmetrical_weakness = 0.0
                
                # Combined weakness score (MS score)
                # Only consider significant weakness
                if global_weakness > 0.3 or asymmetrical_weakness > 0.3:
                    features['facial_weakness'] = min(1.0,
                        (global_weakness * 0.4) +
                        (asymmetrical_weakness * 0.6)  # Weight asymmetry more as it's more specific to MS
                    )
                else:
                    features['facial_weakness'] = 0.0
            
            # Update feature history
            for feature in self.key_features:
                # Ensure the feature value is a simple scalar
                feature_value = features.get(feature, 0.0)
                
                # Add to history
                self.feature_history[feature].append(feature_value)
                
                # Trim history if needed
                if len(self.feature_history[feature]) > self.history_size:
                    self.feature_history[feature].pop(0)
            
            # Store current landmarks for next frame comparison
            self.previous_landmarks = points.copy()
            
            return features, True
            
        except Exception as e:
            print(f"Error extracting facial features: {e}")
            return features, False

# Enhanced Jitter Calculator for Micro-Tremor Detection
class EnhancedJitterCalculator:
    def __init__(self, buffer_size=90):
        self.buffer_size = buffer_size
        self.landmark_buffers = {}
        self.face_size_history = []
        self.max_face_size_history = 10
        self.fps = 30  # Estimate, will be refined during operation
        self.last_frame_time = None
        self.frame_times = []
        
        # Specific frequencies to analyze for different disorders
        self.pd_freq_range = (4, 7)      # PD tremor: 4-7 Hz
        self.essential_tremor_range = (8, 12)  # Essential tremor: 8-12 Hz
        self.ms_tremor_range = (2.5, 6.5)  # MS tremor: Can overlap with PD but often wider range
    
    def reset(self):
        """Reset the jitter calculator for a new session or person"""
        self.landmark_buffers = {}
        self.face_size_history = []
        self.frame_times = []
        self.last_frame_time = None
        
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
                avg_frame_time = np.mean(self.frame_times)
                if avg_frame_time > 0:
                    self.fps = 1.0 / avg_frame_time
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
            
            # Trim buffer if needed
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
    
    def calculate_tremor_features(self):
        """Enhanced tremor analysis for different neurological disorders"""
        # Results for different tremor types
        tremor_results = {
            'pd_tremor': 0.0,
            'ms_tremor': 0.0,
            'eye_tremor': 0.0,
            'facial_tremor': 0.0,  # Added specifically for MS facial tremors
            'tremor_frequency': 0.0,
            'tremor_amplitude': 0.0,
            'tremor_consistency': 0.0
        }
        
        # Skip if we don't have enough data yet
        if not self.landmark_buffers or len(list(self.landmark_buffers.values())[0]) < 60:
            return tremor_results
        
        # Key points most likely to show tremors for different disorders
        key_regions = {
            'parkinsons': [
                61, 291,        # Mouth corners (left, right)
                9, 8,           # Forehead
                17, 0, 13, 14,  # Lips region
                4, 5            # Chin (added from secondary code)
            ],
            'ms': [
                61, 291,        # Mouth corners (left, right)
                101, 331,       # Cheeks
                37, 267,        # Eye corners
                8, 9,           # Forehead
            ],
            'eye_movement': [
                468, 473,       # Iris centers 
                33, 133, 159, 145,  # Left eye perimeter
                362, 263, 386, 374  # Right eye perimeter
            ],
            'facial_tremor': [  # Added for MS facial tremors detection
                61, 291,        # Mouth corners
                101, 331,       # Cheeks
                37, 267,        # Eye corners
                8, 9,           # Forehead
                17, 0, 13, 14,  # Lips region
            ]
        }
        
        # Current estimated FPS (needed for frequency analysis)
        current_fps = max(15.0, self.fps)  # Ensure reasonable minimum
        
        # ------ PD-specific tremors analysis (adapted from second file) ------
        pd_tremor_scores = []
        pd_frequency_powers = []
        
        # Filter for valid landmark indices
        valid_pd_regions = [idx for idx in key_regions['parkinsons'] 
                           if idx in self.landmark_buffers and len(self.landmark_buffers[idx]) >= 60]
        
        for idx in valid_pd_regions:
            # Extract points, normalized by face size
            buffer_data = self.landmark_buffers[idx]
            points = np.array([data[0] for data in buffer_data])
            face_sizes = np.array([data[1] for data in buffer_data])
            
            # Normalize by face size to account for distance/movement
            normalized_points = np.zeros_like(points)
            for i in range(len(points)):
                if face_sizes[i] > 0:
                    normalized_points[i] = points[i] / face_sizes[i]
                else:
                    normalized_points[i] = points[i]
            
            # Analyze each dimension separately
            x_positions = normalized_points[:, 0]
            y_positions = normalized_points[:, 1]
            z_positions = normalized_points[:, 2] if normalized_points.shape[1] > 2 else np.zeros(len(normalized_points))
            
            # Detrend data to remove general movement
            x_detrended = signal.detrend(x_positions)
            y_detrended = signal.detrend(y_positions)
            z_detrended = signal.detrend(z_positions)
            
            # Window the data to reduce spectral leakage
            window = signal.windows.hann(len(x_detrended))
            x_windowed = x_detrended * window
            y_windowed = y_detrended * window
            z_windowed = z_detrended * window
            
            # Calculate FFT
            n = len(x_windowed)
            freqs = np.fft.rfftfreq(n, d=1/current_fps)
            
            x_fft = np.abs(np.fft.rfft(x_windowed))
            y_fft = np.abs(np.fft.rfft(y_windowed))
            z_fft = np.abs(np.fft.rfft(z_windowed))
            
            # Combine power across dimensions
            fft_power = (x_fft * 0.4) + (y_fft * 0.4) + (z_fft * 0.2)
            
            # PD tremor frequencies (4-7 Hz)
            pd_freq_mask = (freqs >= self.pd_freq_range[0]) & (freqs <= self.pd_freq_range[1])
            if np.any(pd_freq_mask):
                pd_power = np.sum(fft_power[pd_freq_mask])
                
                # For comparison, get power in wider range (1-15 Hz)
                all_freq_mask = (freqs >= 1) & (freqs <= 15)
                all_power = np.sum(fft_power[all_freq_mask]) if np.any(all_freq_mask) else 1.0
                
                # Calculate ratio
                pd_power_ratio = pd_power / all_power if all_power > 0 else 0
                
                # Amplitude factor - PD tremors are small but consistent
                avg_amplitude = (np.std(x_detrended) + np.std(y_detrended) + np.std(z_detrended)) / 3
                amplitude_factor = np.exp(-(avg_amplitude - 0.001)**2 / (2 * 0.0005**2))
                
                # Combined tremor score
                tremor_score = pd_power_ratio * amplitude_factor
                pd_tremor_scores.append(tremor_score)
                pd_frequency_powers.append(pd_power)
                
                # Store frequency with maximum power in PD range
                if pd_power > 0 and np.any(pd_freq_mask):
                    max_freq_idx = np.argmax(fft_power[pd_freq_mask])
                    max_freq = freqs[pd_freq_mask][max_freq_idx]
                    tremor_results['tremor_frequency'] = max_freq
                    
            # Also detect rapid direction changes (another PD characteristic) from second file
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
                    pd_tremor_scores.append(direction_change_rate * 0.5)
        
        # Calculate overall PD tremor score
        if pd_tremor_scores:
            # Use 80th percentile for robustness
            pd_tremor = np.percentile(pd_tremor_scores, 80)
            
            # Scale with sigmoid function
            sensitivity = 100.0
            threshold = 0.1
            pd_tremor_scaled = 1.0 / (1.0 + np.exp(-sensitivity * (pd_tremor - threshold)))
            tremor_results['pd_tremor'] = min(1.0, max(0.0, pd_tremor_scaled))
            
            # Store amplitude for UI display
            if pd_frequency_powers:
                tremor_results['tremor_amplitude'] = np.mean(pd_frequency_powers) * 1000
        
        # ------ MS-specific tremors analysis ------
        ms_tremor_scores = []
        
        # Filter for valid landmark indices
        valid_ms_regions = [idx for idx in key_regions['ms'] 
                           if idx in self.landmark_buffers and len(self.landmark_buffers[idx]) >= 60]
        
        for idx in valid_ms_regions:
            # Extract points and normalize by face size
            buffer_data = self.landmark_buffers[idx]
            points = np.array([data[0] for data in buffer_data])
            face_sizes = np.array([data[1] for data in buffer_data])
            
            # Normalize by face size
            normalized_points = np.zeros_like(points)
            for i in range(len(points)):
                if face_sizes[i] > 0:
                    normalized_points[i] = points[i] / face_sizes[i]
                else:
                    normalized_points[i] = points[i]
            
            # Process and analyze for MS specific tremor patterns
            # MS tremors often have wider frequency range and burst pattern
            x_positions = normalized_points[:, 0]
            y_positions = normalized_points[:, 1]
            
            x_detrended = signal.detrend(x_positions)
            y_detrended = signal.detrend(y_positions)
            
            # Window the data
            window = signal.windows.hann(len(x_detrended))
            x_windowed = x_detrended * window
            y_windowed = y_detrended * window
            
            # Calculate FFT
            n = len(x_windowed)
            freqs = np.fft.rfftfreq(n, d=1/current_fps)
            
            x_fft = np.abs(np.fft.rfft(x_windowed))
            y_fft = np.abs(np.fft.rfft(y_windowed))
            
            # Combine power across dimensions
            fft_power = (x_fft * 0.5) + (y_fft * 0.5)
            
            # MS tremor frequencies
            ms_freq_mask = (freqs >= self.ms_tremor_range[0]) & (freqs <= self.ms_tremor_range[1])
            if np.any(ms_freq_mask):
                ms_power = np.sum(fft_power[ms_freq_mask])
                
                # Compare to all movement power
                all_freq_mask = (freqs >= 1) & (freqs <= 15)
                all_power = np.sum(fft_power[all_freq_mask]) if np.any(all_freq_mask) else 1.0
                
                # Calculate ratio
                ms_power_ratio = ms_power / all_power if all_power > 0 else 0
                
                # Add to scores
                ms_tremor_scores.append(ms_power_ratio)
        
        # Calculate overall MS tremor score with increased threshold
        if ms_tremor_scores:
            ms_tremor = np.percentile(ms_tremor_scores, 80)
            
            # Scale with higher threshold to reduce false positives
            sensitivity = 80.0
            threshold = 0.15  # Increased from 0.12
            ms_tremor_scaled = 1.0 / (1.0 + np.exp(-sensitivity * (ms_tremor - threshold)))
            tremor_results['ms_tremor'] = min(1.0, max(0.0, ms_tremor_scaled))
        
        # ------ Eye movement analysis for MS ------
        eye_tremor_scores = []
        
        # Filter for valid eye landmark indices
        valid_eye_regions = [idx for idx in key_regions['eye_movement'] 
                           if idx in self.landmark_buffers and len(self.landmark_buffers[idx]) >= 60]
        
        for idx in valid_eye_regions:
            # Focus on horizontal eye movements (nystagmus is often horizontal)
            buffer_data = self.landmark_buffers[idx]
            points = np.array([data[0] for data in buffer_data])
            face_sizes = np.array([data[1] for data in buffer_data])
            
            # Normalize and analyze for nystagmus-like patterns
            if len(points) > 0 and points.shape[1] >= 2:
                x_positions = points[:, 0] / face_sizes if np.all(face_sizes > 0) else points[:, 0]
                
                # Detrend data
                x_detrended = signal.detrend(x_positions)
                
                # Calculate velocity
                x_velocity = np.diff(x_detrended)
                
                # Find fast movements (potential nystagmus fast phases)
                velocity_std = np.std(x_velocity)
                velocity_threshold = velocity_std * 2.5  # Increased threshold
                
                if velocity_threshold > 0:
                    fast_movements = np.abs(x_velocity) > velocity_threshold
                    
                    # Count fast movements
                    fast_movement_count = np.sum(fast_movements)
                    
                    # Normalize by data length
                    if len(x_velocity) > 0:
                        normalized_fast_count = fast_movement_count / len(x_velocity)
                        
                        # Check for directional bias
                        left_fast = np.sum(x_velocity < -velocity_threshold)
                        right_fast = np.sum(x_velocity > velocity_threshold)
                        
                        # Strong directional bias is characteristic of nystagmus
                        max_fast = max(left_fast, right_fast)
                        if max_fast > 0:
                            balance_ratio = min(left_fast, right_fast) / max_fast
                            directional_bias = 1.0 - balance_ratio
                        else:
                            directional_bias = 0.0
                        
                        # Combined eye movement abnormality score
                        # Require more significant values
                        if normalized_fast_count > 0.1 and directional_bias > 0.3:  # Higher thresholds
                            eye_score = (normalized_fast_count * 0.4 + directional_bias * 0.6)
                            eye_tremor_scores.append(eye_score)
        
        # Calculate overall eye tremor/nystagmus score
        if eye_tremor_scores:
            eye_tremor = np.percentile(eye_tremor_scores, 80)
            
            # Scale with higher threshold
            sensitivity = 70.0
            threshold = 0.2  # Increased from 0.15
            eye_tremor_scaled = 1.0 / (1.0 + np.exp(-sensitivity * (eye_tremor - threshold)))
            tremor_results['eye_tremor'] = min(1.0, max(0.0, eye_tremor_scaled))
            
        # Analysis for MS facial tremors - increased thresholds to reduce false positives
        facial_tremor_scores = []
        valid_facial_tremor_regions = [idx for idx in key_regions['facial_tremor'] 
                                      if idx in self.landmark_buffers and len(self.landmark_buffers[idx]) >= 60]
        
        for idx in valid_facial_tremor_regions:
            buffer_data = self.landmark_buffers[idx]
            points = np.array([data[0] for data in buffer_data])
            face_sizes = np.array([data[1] for data in buffer_data])
            
            # Normalize by face size
            normalized_points = np.zeros_like(points)
            for i in range(len(points)):
                if face_sizes[i] > 0:
                    normalized_points[i] = points[i] / face_sizes[i]
                else:
                    normalized_points[i] = points[i]
            
            # Analyze in all dimensions as facial tremors can be multidirectional
            x_positions = normalized_points[:, 0]
            y_positions = normalized_points[:, 1]
            
            # Detrend to remove intentional movements
            x_detrended = signal.detrend(x_positions)
            y_detrended = signal.detrend(y_positions)
            
            # Window for better FFT
            window = signal.windows.hann(len(x_detrended))
            x_windowed = x_detrended * window
            y_windowed = y_detrended * window
            
            # Calculate FFT
            n = len(x_windowed)
            freqs = np.fft.rfftfreq(n, d=1/current_fps)
            
            x_fft = np.abs(np.fft.rfft(x_windowed))
            y_fft = np.abs(np.fft.rfft(y_windowed))
            
            # Combine power
            fft_power = (x_fft * 0.5) + (y_fft * 0.5)
            
            # MS facial tremors are often in a similar range as MS tremors
            ms_face_freq_mask = (freqs >= 2.5) & (freqs <= 7.0)
            
            if np.any(ms_face_freq_mask):
                facial_tremor_power = np.sum(fft_power[ms_face_freq_mask])
                
                # Compare to all movement
                all_freq_mask = (freqs >= 1) & (freqs <= 15)
                all_power = np.sum(fft_power[all_freq_mask]) if np.any(all_freq_mask) else 1.0
                
                if all_power > 0:
                    facial_tremor_ratio = facial_tremor_power / all_power
                    
                    # Add burst detection which is characteristic of MS facial tremors
                    # Use the standard deviation of the power as a measure of burstiness
                    if np.mean(fft_power[ms_face_freq_mask]) > 0:
                        burst_factor = np.std(fft_power[ms_face_freq_mask]) / np.mean(fft_power[ms_face_freq_mask])
                    else:
                        burst_factor = 0
                    
                    # Higher threshold for significant burst pattern
                    if burst_factor > 1.5:  # Only consider significant burst patterns
                        # Combined score that considers both power and burst pattern
                        facial_tremor_score = facial_tremor_ratio * (1.0 + burst_factor)
                        facial_tremor_scores.append(facial_tremor_score)
        
        # Calculate facial tremor score for MS with higher threshold
        if facial_tremor_scores:
            facial_tremor = np.percentile(facial_tremor_scores, 80)
            
            # Scale with sigmoid and higher threshold
            sensitivity = 60.0
            threshold = 0.15  # Increased from 0.1
            facial_tremor_scaled = 1.0 / (1.0 + np.exp(-sensitivity * (facial_tremor - threshold)))
            tremor_results['facial_tremor'] = min(1.0, max(0.0, facial_tremor_scaled))
        
        # Calculate tremor consistency across frames
        if pd_tremor_scores and len(pd_tremor_scores) > 5:
            pd_tremor_mean = np.mean(pd_tremor_scores)
            if pd_tremor_mean > 0:
                tremor_consistency = 1.0 - (np.std(pd_tremor_scores) / pd_tremor_mean)
                tremor_results['tremor_consistency'] = max(0.0, min(1.0, tremor_consistency))
        
        return tremor_results

# ----------------------------------------------------------------------
# PART 2: DIRECT PROBABILITY CALCULATORS
# ----------------------------------------------------------------------

def enhanced_pd_from_emotions(emotion_probs):
    """Calculate PD probability from emotion distribution with improved weighting from second file"""
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

def calculate_direct_pd_probability(features, emotion_probs=None, tremor_results=None):
    """Calculate PD probability using Parkinson's detection approach from the second file"""
    if emotion_probs is not None:
        # Use enhanced emotional pattern analysis
        emotion_pd_prob = enhanced_pd_from_emotions(emotion_probs)
    else:
        emotion_pd_prob = 0.0
        
    # Get tremor score from results
    tremor_score = tremor_results.get('pd_tremor', 0.0) if tremor_results else 0.0
    
    # Calculate overall PD probability with improved weighting
    # Features from the research paper with refined weights
    feature_pd_prob = features.get('mask_face_score', 0.0)
    expression_range = features.get('expression_range', 0.0)
    smile_symmetry = features.get('smile_symmetry', 0.0)
    facial_mobility = features.get('facial_mobility', 0.0)
    
    # Calculate total PD probability with key weights
    if emotion_probs is not None:
        pd_probability = (
            feature_pd_prob * 0.3 + 
            emotion_pd_prob * 0.25 + 
            tremor_score * 0.3 +
            smile_symmetry * 0.1 + 
            max(0.0, 0.3 - facial_mobility) * 0.05  # Low mobility contributes to PD probability
        )
    else:
        pd_probability = (
            feature_pd_prob * 0.45 + 
            tremor_score * 0.4 +
            smile_symmetry * 0.1 + 
            max(0.0, 0.3 - facial_mobility) * 0.05
        )
    
    # Track active indicators
    active_indicators = {}
    
    if feature_pd_prob > 0.3:
        active_indicators['mask_face_score'] = feature_pd_prob
    if smile_symmetry > 0.3:
        active_indicators['smile_symmetry'] = smile_symmetry
    if facial_mobility < 0.3:
        active_indicators['facial_mobility'] = 0.3 - facial_mobility  # Invert for scoring
    if tremor_score > 0.3:
        active_indicators['tremor'] = tremor_score
    if emotion_pd_prob > 0.3:
        active_indicators['emotion_pattern'] = emotion_pd_prob
        
    # Apply calibration to reduce false positives
    # More conservative threshold for higher specificity
    pd_prob = max(0.0, min(1.0, pd_probability))
    
    return pd_prob, active_indicators

def calculate_direct_alzheimers_probability(features, emotion_probs=None):
    """Calculate Alzheimer's probability using fixed thresholds instead of baselines"""
    # Define thresholds for AD indicators - UPDATED for mask face OR increased expressivity
    thresholds = {
        'attention_score': 0.4,                  # Low attention score indicates AD (inverted)
        'emotion_response_delay': 0.5,           # High response delay indicates AD
        'expression_spatial_organization': 0.4,   # Low organization indicates AD (inverted)
        'mask_face_score': 0.6,                  # High mask face score can indicate AD
        'expression_intensity': 0.7              # High expression intensity can indicate AD
    }
    
    weights = {
        'attention_score': 0.25,
        'emotion_response_delay': 0.2,
        'expression_spatial_organization': 0.15,
        'mask_face_or_exaggerated': 0.25,        # Combined weight for either symptom
        'slow_reaction': 0.15
    }
    
    # Start with zero probability
    raw_indicator_scores = {}
    
    # Check standard features against fixed thresholds
    for feature_name, threshold in thresholds.items():
        if feature_name in features and feature_name not in ['mask_face_score', 'expression_intensity']:
            value = features[feature_name]
            if not isinstance(value, (int, float)) or value is None:
                raw_indicator_scores[feature_name] = 0.0
                continue
            
            if feature_name in ['attention_score', 'expression_spatial_organization']:
                # For these features, lower is worse (invert the value and threshold)
                if value < threshold:
                    normalized_score = (threshold - value) / threshold
                    raw_indicator_scores[feature_name] = normalized_score * weights.get(feature_name, 0.1)
                else:
                    raw_indicator_scores[feature_name] = 0.0
            else:
                # For other features, higher is worse
                if value > threshold:
                    normalized_score = (value - threshold) / (1.0 - threshold)
                    raw_indicator_scores[feature_name] = normalized_score * weights.get(feature_name, 0.1)
                else:
                    raw_indicator_scores[feature_name] = 0.0
    
    # Special handling for mask face OR increased expressivity
    # For AD, either extreme can be a symptom - look for either one
    mask_face_score = features.get('mask_face_score', 0)
    expression_intensity = features.get('expression_intensity', 0)
    
    mask_face_threshold = thresholds['mask_face_score']
    intensity_threshold = thresholds['expression_intensity']
    
    # Check if either extreme is present
    if mask_face_score > mask_face_threshold:
        # High mask face score (reduced expressivity)
        normalized_score = (mask_face_score - mask_face_threshold) / (1.0 - mask_face_threshold)
        raw_indicator_scores['mask_face_or_exaggerated'] = normalized_score * weights['mask_face_or_exaggerated']
    elif expression_intensity > intensity_threshold:
        # High expression intensity (increased expressivity)
        normalized_score = (expression_intensity - intensity_threshold) / (1.0 - intensity_threshold)
        raw_indicator_scores['mask_face_or_exaggerated'] = normalized_score * weights['mask_face_or_exaggerated']
    else:
        raw_indicator_scores['mask_face_or_exaggerated'] = 0.0
    
    # Add reaction time indicator if available
    if 'reaction_time_ms' in features and features['reaction_time_ms'] > 0:
        reaction_time = features['reaction_time_ms']
        
        # Check if reaction time is abnormally slow (> 800ms is concerning)
        if reaction_time > 800:
            # Scale based on severity (800-1500ms range)
            normalized_delay = min(1.0, (reaction_time - 800) / 700)
            raw_indicator_scores['slow_reaction'] = normalized_delay * weights['slow_reaction']
        else:
            raw_indicator_scores['slow_reaction'] = 0.0
    
    # Calculate total AD probability and active indicators
    total_probability = sum(raw_indicator_scores.values())
    ad_prob = max(0.0, min(1.0, total_probability))
    active_indicators = {name: score for name, score in raw_indicator_scores.items() if score > 0.05}
    
    return ad_prob, active_indicators

def calculate_direct_ms_probability(features, tremor_results=None):
    """Calculate MS probability with higher thresholds to reduce false positives"""
    # Define thresholds for MS indicators - UPDATED with higher thresholds
    thresholds = {
        'facial_symmetry': 0.6,            # Increased from 0.5
        'facial_weakness': 0.6,            # Increased from 0.5
        'blink_regularity': 0.7,           # Increased from 0.6
        'eye_movement_abnormalities': 0.6,  # Increased from 0.5
        'pupil_misalignment': 0.5,         # Increased from 0.4
        'unilateral_twitching': 0.5        # Increased from 0.4
    }
    
    # Adjusted weights to prioritize the specific MS symptoms
    weights = {
        'facial_symmetry': 0.05,          # Reduced weight
        'facial_weakness': 0.05,          # Reduced weight
        'blink_regularity': 0.05,         # Reduced weight
        'eye_movement_abnormalities': 0.1,
        'ms_tremor': 0.05,               # Reduced weight
        'eye_tremor': 0.1,
        'facial_tremor': 0.3,             # Increased weight for facial tremors in MS
        'pupil_misalignment': 0.2,        # Higher weight for pupil misalignment (double vision)
        'unilateral_twitching': 0.2       # Higher weight for one-sided twitching
    }
    
    # Start with zero probability
    raw_indicator_scores = {}
    
    # Check features against higher thresholds
    for feature_name, threshold in thresholds.items():
        if feature_name in features:
            value = features[feature_name]
            if not isinstance(value, (int, float)) or value is None:
                raw_indicator_scores[feature_name] = 0.0
                continue
            
            # For all MS features, higher is worse - but use higher thresholds
            if value > threshold:
                normalized_score = (value - threshold) / (1.0 - threshold)
                raw_indicator_scores[feature_name] = normalized_score * weights[feature_name]
            else:
                raw_indicator_scores[feature_name] = 0.0
    
    # Add tremor indicators if available
    if tremor_results:
        ms_tremor = tremor_results.get('ms_tremor', 0.0)
        eye_tremor = tremor_results.get('eye_tremor', 0.0)
        facial_tremor = tremor_results.get('facial_tremor', 0.0)
        
        # Only count if tremors are significant - higher thresholds
        if ms_tremor > 0.4:  # Increased from 0.3
            raw_indicator_scores['ms_tremor'] = ms_tremor * weights['ms_tremor']
        else:
            raw_indicator_scores['ms_tremor'] = 0.0
            
        if eye_tremor > 0.4:  # Increased from 0.3
            raw_indicator_scores['eye_tremor'] = eye_tremor * weights['eye_tremor']
        else:
            raw_indicator_scores['eye_tremor'] = 0.0
            
        # Add facial tremor score with higher weight for MS detection
        if facial_tremor > 0.35:  # Increased from 0.25
            raw_indicator_scores['facial_tremor'] = facial_tremor * weights['facial_tremor']
        else:
            raw_indicator_scores['facial_tremor'] = 0.0
    
    # Calculate total MS probability and active indicators
    total_probability = sum(raw_indicator_scores.values())
    
    # MS requires multiple symptoms to be present for accurate diagnosis
    # Increase requirement to reduce false positives
    significant_indicators = sum(1 for score in raw_indicator_scores.values() if score > 0.05)
    if significant_indicators < 3:  # Increased from 2
        # Scale down probability more severely if fewer than 3 indicators
        total_probability *= significant_indicators / 3
    
    ms_prob = max(0.0, min(1.0, total_probability))
    active_indicators = {name: score for name, score in raw_indicator_scores.items() if score > 0.05}
    
    return ms_prob, active_indicators

# ----------------------------------------------------------------------
# PART 3: MAIN DETECTION SYSTEM
# ----------------------------------------------------------------------

class NeurologicalDisorderDetectionSystem:
    def __init__(self, emotion_model_path=None, rafdb_model_path=None):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else 
                                  "cpu")
        print(f"Using device: {self.device}")
        
        # Load emotion recognition model
        self.emotion_model = None
        self.rafdb_model = None
        
        # Try loading the standard emotion model
        if emotion_model_path and os.path.exists(emotion_model_path):
            try:
                self.emotion_model = FacialExpressionCNN(num_classes=7).to(self.device)
                self.emotion_model.load_state_dict(torch.load(emotion_model_path, map_location=self.device))
                self.emotion_model.eval()
                print(f"Emotion model loaded from: {emotion_model_path}")
            except Exception as e:
                print(f"Error loading emotion model: {e}")
                print("Running without standard emotion recognition")
                
        # Try loading the RAF-DB model (has priority if both available)
        if rafdb_model_path and os.path.exists(rafdb_model_path):
            try:
                self.rafdb_model = RAFDBEmotionCNN(num_classes=7).to(self.device)
                self.rafdb_model.load_state_dict(torch.load(rafdb_model_path, map_location=self.device))
                self.rafdb_model.eval()
                print(f"RAF-DB model loaded from: {rafdb_model_path}")
            except Exception as e:
                print(f"Error loading RAF-DB model: {e}")
                print("Running without RAF-DB emotion recognition")
                
        if not self.emotion_model and not self.rafdb_model:
            print("No emotion models provided or files not found.")
            print("Running with facial features only.")
        
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
        
        # Initialize enhanced components
        self.feature_extractor = FacialFeatureExtractor(history_size=60)
        self.jitter_calculator = EnhancedJitterCalculator(buffer_size=90)
        
        # Skip baseline learning by setting directly to active mode
        self.is_in_learning_phase = False
        
        # Transforms for different models
        self.standard_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ])
        
        self.rafdb_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ])
        
        # Store recent detection results for stability
        self.recent_results = {
            'pd': [],
            'alzheimers': [],
            'ms': []
        }
        self.max_results_history = 10
        
        # Track active symptoms
        self.active_indicators = {
            'pd': {},
            'alzheimers': {},
            'ms': {}
        }
        
        # Session history for longitudinal analysis
        self.session_history = []
        self.max_session_history = 1000
        
        print("Disorder detection system initialized in direct analysis mode.")

    def process_frame(self, frame):
        """Process a single frame for neurological disorder detection"""
        try:
            # Get current time
            current_time = time.time()
            
            # Initialize result dictionary
            result = {
                'face_detected': False,
                'emotion': None,
                'emotion_probs': None,
                'features': None,
                'tremor_results': None,
                'pd_probability': 0.0,
                'alzheimers_probability': 0.0,
                'ms_probability': 0.0,
                'pd_symptoms': [],
                'alzheimers_symptoms': [],
                'ms_symptoms': [],
                'most_likely_disorder': "No Disorder Detected",
                'most_likely_probability': 0.0,
                'confidence': 0.0,
                'key_symptoms': []
            }
            
            # Get frame dimensions
            if frame is None:
                return result
                
            h, w = frame.shape[:2]
            
            # Process with MediaPipe for landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            # Check if face was detected
            if not results.multi_face_landmarks:
                return result
            
            # Face was detected
            result['face_detected'] = True
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Convert landmarks to NumPy array
            try:
                landmark_points = np.array([(lm.x * w, lm.y * h, lm.z) for lm in landmarks])
            except Exception as e:
                print(f"Error converting landmarks: {e}")
                return result
            
            # Add landmarks to jitter calculator
            self.jitter_calculator.add_frame(landmark_points, current_time)
            
            # Extract facial features
            features, feature_success = self.feature_extractor.extract_features(frame, current_time)
            result['features'] = features
            
            # Calculate tremor results
            tremor_results = self.jitter_calculator.calculate_tremor_features()
            result['tremor_results'] = tremor_results
            
            # Predict emotion if models are available
            emotion_probs = None
            if self.rafdb_model:  # Prioritize RAF-DB model if available
                # Prepare image for emotion model
                try:
                    emotion_input = self.rafdb_transform(rgb_frame).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        emotion_outputs = self.rafdb_model(emotion_input)
                        emotion_probs = torch.softmax(emotion_outputs, dim=1)[0].cpu().numpy()
                        _, emotion_class = torch.max(emotion_outputs, 1)
                        
                        # Get emotion name
                        emotion = self.emotions[emotion_class.item()]
                        
                        # Store emotion results
                        result['emotion'] = emotion
                        result['emotion_probs'] = emotion_probs
                        result['emotion_model_used'] = 'RAF-DB'
                except Exception as e:
                    print(f"Error in RAF-DB emotion prediction: {e}")
                    emotion_probs = None
            
            # Fall back to standard model if RAF-DB failed or isn't available
            if emotion_probs is None and self.emotion_model:
                try:
                    emotion_input = self.standard_transform(rgb_frame).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        emotion_outputs = self.emotion_model(emotion_input)
                        emotion_probs = torch.softmax(emotion_outputs, dim=1)[0].cpu().numpy()
                        _, emotion_class = torch.max(emotion_outputs, 1)
                        
                        # Get emotion name
                        emotion = self.emotions[emotion_class.item()]
                        
                        # Store emotion results
                        result['emotion'] = emotion
                        result['emotion_probs'] = emotion_probs
                        result['emotion_model_used'] = 'Standard'
                except Exception as e:
                    print(f"Error in standard emotion prediction: {e}")
            
            # Calculate disorder probabilities with direct thresholds
            pd_prob, pd_indicators = calculate_direct_pd_probability(
                features, emotion_probs, tremor_results)
            
            alzheimers_prob, alzheimers_indicators = calculate_direct_alzheimers_probability(
                features, emotion_probs)
            
            ms_prob, ms_indicators = calculate_direct_ms_probability(
                features, tremor_results)
            
            # Store active indicators
            self.active_indicators['pd'] = pd_indicators
            self.active_indicators['alzheimers'] = alzheimers_indicators
            self.active_indicators['ms'] = ms_indicators
            
            # Add to recent results for stability
            self.recent_results['pd'].append(pd_prob)
            self.recent_results['alzheimers'].append(alzheimers_prob)
            self.recent_results['ms'].append(ms_prob)
            
            for key in self.recent_results:
                if len(self.recent_results[key]) > self.max_results_history:
                    self.recent_results[key].pop(0)
            
            # Use stabilized probabilities
            stable_pd_prob = np.median(self.recent_results['pd']) if self.recent_results['pd'] else pd_prob
            stable_ad_prob = np.median(self.recent_results['alzheimers']) if self.recent_results['alzheimers'] else alzheimers_prob
            stable_ms_prob = np.median(self.recent_results['ms']) if self.recent_results['ms'] else ms_prob
            
            # Calculate which disorder is most likely
            disorder_probs = {
                'Parkinson\'s Disease': stable_pd_prob,
                'Alzheimer\'s Disease': stable_ad_prob,
                'Multiple Sclerosis': stable_ms_prob
            }
            
            # Find most likely disorder
            most_likely_disorder = max(disorder_probs, key=disorder_probs.get)
            most_likely_probability = disorder_probs[most_likely_disorder]
            
            # Only report a likely disorder if probability exceeds threshold
            # Higher threshold for MS to reduce false positives
            ms_threshold = 0.5  # Increased from 0.45
            other_threshold = 0.45
            
            if most_likely_disorder == "Multiple Sclerosis" and most_likely_probability < ms_threshold:
                most_likely_disorder = "No Disorder Detected"
                most_likely_probability = 1.0 - max(stable_pd_prob, stable_ad_prob, stable_ms_prob)
            elif most_likely_disorder != "Multiple Sclerosis" and most_likely_probability < other_threshold:
                most_likely_disorder = "No Disorder Detected"
                most_likely_probability = 1.0 - max(stable_pd_prob, stable_ad_prob, stable_ms_prob)
            
            # Translate indicators to symptom descriptions
            # PD symptoms
            pd_symptoms = []
            for indicator, value in pd_indicators.items():
                if indicator == 'mask_face_score' and value > 0.05:
                    pd_symptoms.append("Mask face (reduced expressivity)")
                elif indicator == 'smile_symmetry' and value > 0.05:
                    pd_symptoms.append("Facial asymmetry")
                elif indicator == 'facial_mobility' and value > 0.05:
                    pd_symptoms.append("Reduced facial mobility")
                elif indicator == 'tremor' and value > 0.05:
                    pd_symptoms.append(f"Facial tremor at {tremor_results.get('tremor_frequency', 0):.1f} Hz")
                elif indicator == 'emotion_pattern' and value > 0.05:
                    pd_symptoms.append("Predominant neutral expression")
            
            # Alzheimer's symptoms
            alzheimers_symptoms = []
            for indicator, value in alzheimers_indicators.items():
                if indicator == 'attention_score' and value > 0.05:
                    alzheimers_symptoms.append("Reduced attention/focus")
                elif indicator == 'emotion_response_delay' and value > 0.05:
                    alzheimers_symptoms.append("Delayed emotional responses")
                elif indicator == 'expression_spatial_organization' and value > 0.05:
                    alzheimers_symptoms.append("Disorganized facial expressions")
                elif indicator == 'slow_reaction' and value > 0.05:
                    alzheimers_symptoms.append(f"Slow reaction time ({features.get('reaction_time_ms', 0):.0f} ms)")
                elif indicator == 'mask_face_or_exaggerated' and value > 0.05:
                    # Determine which extreme is present
                    if features.get('mask_face_score', 0) > features.get('expression_intensity', 0):
                        alzheimers_symptoms.append("Reduced facial expressivity (mask face)")
                    else:
                        alzheimers_symptoms.append("Exaggerated facial expressions")
            
            # MS symptoms - UPDATED for specific MS symptoms
            ms_symptoms = []
            for indicator, value in ms_indicators.items():
                if indicator == 'facial_symmetry' and value > 0.05:
                    ms_symptoms.append("Pronounced facial asymmetry")
                elif indicator == 'facial_weakness' and value > 0.05:
                    ms_symptoms.append("Facial weakness/paresis")
                elif indicator == 'blink_regularity' and value > 0.05:
                    ms_symptoms.append("Irregular blink pattern")
                elif indicator == 'eye_movement_abnormalities' and value > 0.05:
                    ms_symptoms.append("Eye movement abnormalities")
                elif indicator == 'eye_tremor' and value > 0.05:
                    ms_symptoms.append("Nystagmus (eye tremor)")
                elif indicator == 'ms_tremor' and value > 0.05:
                    ms_symptoms.append("General tremor patterns")
                elif indicator == 'facial_tremor' and value > 0.05:
                    ms_symptoms.append("Facial tremors")
                elif indicator == 'pupil_misalignment' and value > 0.05:
                    ms_symptoms.append("Pupil misalignment (double vision)")
                elif indicator == 'unilateral_twitching' and value > 0.05:
                    ms_symptoms.append("One-sided facial twitching")
            
            # Store all symptoms in result
            result['pd_symptoms'] = pd_symptoms
            result['alzheimers_symptoms'] = alzheimers_symptoms
            result['ms_symptoms'] = ms_symptoms
            
            # Determine key symptoms based on most likely disorder
            if most_likely_disorder == "Parkinson's Disease":
                result['key_symptoms'] = pd_symptoms
            elif most_likely_disorder == "Alzheimer's Disease":
                result['key_symptoms'] = alzheimers_symptoms
            elif most_likely_disorder == "Multiple Sclerosis":
                result['key_symptoms'] = ms_symptoms
            else:
                result['key_symptoms'] = []
            
            # Store final results
            result['pd_probability'] = stable_pd_prob
            result['alzheimers_probability'] = stable_ad_prob
            result['ms_probability'] = stable_ms_prob
            result['most_likely_disorder'] = most_likely_disorder
            result['most_likely_probability'] = most_likely_probability
            
            # Calculate confidence based on data quality
            data_amount_factor = min(1.0, len(list(self.jitter_calculator.landmark_buffers.values())[0]) / 60.0) if self.jitter_calculator.landmark_buffers else 0.0
            feature_quality_factor = 1.0 if feature_success else 0.5
            symptom_count_factor = min(1.0, len(result['key_symptoms']) / 3.0)
            
            result['confidence'] = (
                data_amount_factor * 0.4 + 
                feature_quality_factor * 0.3 +
                symptom_count_factor * 0.3
            )
            
            # Add to session history for longitudinal analysis
            self.session_history.append({
                'timestamp': current_time,
                'pd_probability': stable_pd_prob,
                'alzheimers_probability': stable_ad_prob,
                'ms_probability': stable_ms_prob,
                'key_symptoms': result['key_symptoms'],
                'most_likely_disorder': most_likely_disorder
            })
            
            if len(self.session_history) > self.max_session_history:
                self.session_history.pop(0)
                
            return result
        except Exception as e:
            print(f"Error in process_frame: {e}")
            # Return default result on error
            return {
                'face_detected': False,
                'emotion': None,
                'emotion_probs': None,
                'features': None,
                'tremor_results': None,
                'pd_probability': 0.0,
                'alzheimers_probability': 0.0,
                'ms_probability': 0.0,
                'pd_symptoms': [],
                'alzheimers_symptoms': [],
                'ms_symptoms': [],
                'most_likely_disorder': "No Disorder Detected",
                'most_likely_probability': 0.0,
                'confidence': 0.0,
                'key_symptoms': []
            }

    def analyze_session_history(self):
        """Analyze session history for longitudinal patterns in disorder detection"""
        if not self.session_history:
            return {
                'pd_avg': 0.0,
                'alzheimers_avg': 0.0,
                'ms_avg': 0.0,
                'symptom_frequency': {},
                'dominant_disorder': 'None',
                'detection_confidence': 0.0,
                'trend': 'stable'
            }
        
        # Extract probabilities and timestamps
        timestamps = [entry['timestamp'] for entry in self.session_history]
        pd_probs = [entry['pd_probability'] for entry in self.session_history]
        ad_probs = [entry['alzheimers_probability'] for entry in self.session_history]
        ms_probs = [entry['ms_probability'] for entry in self.session_history]
        
        # Calculate averages
        pd_avg = np.mean(pd_probs)
        ad_avg = np.mean(ad_probs)
        ms_avg = np.mean(ms_probs)
        
        # Find most frequent disorder
        disorders = [entry['most_likely_disorder'] for entry in self.session_history]
        if disorders:
            disorder_counts = {}
            for disorder in disorders:
                if disorder not in disorder_counts:
                    disorder_counts[disorder] = 0
                disorder_counts[disorder] += 1
            
            dominant_disorder = max(disorder_counts, key=disorder_counts.get)
            detection_confidence = disorder_counts[dominant_disorder] / len(disorders)
        else:
            dominant_disorder = 'None'
            detection_confidence = 0.0
        
        # Collect symptom frequency
        symptom_frequency = {}
        for entry in self.session_history:
            for symptom in entry['key_symptoms']:
                if symptom not in symptom_frequency:
                    symptom_frequency[symptom] = 0
                symptom_frequency[symptom] += 1
        
        # Sort symptoms by frequency
        symptom_frequency = dict(sorted(
            symptom_frequency.items(), 
            key=lambda item: item[1], 
            reverse=True
        ))
        
        # Analyze trend (if we have enough data)
        trend = 'stable'
        if len(self.session_history) >= 10:
            # Get the disorder that matches the dominant_disorder
            if dominant_disorder == "Parkinson's Disease":
                recent_probs = pd_probs[-10:]
            elif dominant_disorder == "Alzheimer's Disease":
                recent_probs = ad_probs[-10:]
            elif dominant_disorder == "Multiple Sclerosis":
                recent_probs = ms_probs[-10:]
            else:
                recent_probs = [0.0] * 10
            
            # Simple linear trend analysis
            if len(recent_probs) >= 2:
                first_half_avg = np.mean(recent_probs[:len(recent_probs)//2])
                second_half_avg = np.mean(recent_probs[len(recent_probs)//2:])
                
                diff = second_half_avg - first_half_avg
                if diff > 0.05:
                    trend = 'worsening'
                elif diff < -0.05:
                    trend = 'improving'
                else:
                    trend = 'stable'
        
        return {
            'pd_avg': pd_avg,
            'alzheimers_avg': ad_avg,
            'ms_avg': ms_avg,
            'symptom_frequency': symptom_frequency,
            'dominant_disorder': dominant_disorder,
            'detection_confidence': detection_confidence,
            'trend': trend
        }