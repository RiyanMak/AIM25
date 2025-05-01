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


# Initialize MediaPipe for Facial Landmark Detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.3)  # Lower threshold for better detection

# Initialize MediaPipe face mesh with refined model
mp_face_mesh_refined = mp.solutions.face_mesh
face_mesh_refined = mp_face_mesh_refined.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    refine_landmarks=True)  # Enable refined landmarks for better eye tracking


# Head Pose Estimation Helper Class
class HeadPoseEstimator:
    def __init__(self):
        # 3D model points (standard face model)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        # Indices of the corresponding landmarks in MediaPipe Face Mesh
        self.face_landmarks_idx = [4, 152, 33, 263, 61, 291]

        # Camera matrix (will be initialized based on image size)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # Rotation and translation vectors
        self.rotation_vector = None
        self.translation_vector = None

    def estimate_pose(self, image, landmarks):
        """Estimate head pose from facial landmarks"""
        h, w = image.shape[:2]

        # Initialize camera matrix if not already done
        if self.camera_matrix is None:
            focal_length = w
            center = (w / 2, h / 2)
            self.camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )

        # Get 2D landmarks for pose estimation
        image_points = np.array([
            (landmarks[self.face_landmarks_idx[0]].x * w, landmarks[self.face_landmarks_idx[0]].y * h),  # Nose tip
            (landmarks[self.face_landmarks_idx[1]].x * w, landmarks[self.face_landmarks_idx[1]].y * h),  # Chin
            (landmarks[self.face_landmarks_idx[2]].x * w, landmarks[self.face_landmarks_idx[2]].y * h),
            # Left eye left corner
            (landmarks[self.face_landmarks_idx[3]].x * w, landmarks[self.face_landmarks_idx[3]].y * h),
            # Right eye right corner
            (landmarks[self.face_landmarks_idx[4]].x * w, landmarks[self.face_landmarks_idx[4]].y * h),
            # Left mouth corner
            (landmarks[self.face_landmarks_idx[5]].x * w, landmarks[self.face_landmarks_idx[5]].y * h)
            # Right mouth corner
        ], dtype="double")

        # Solve for pose
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeffs)

        if success:
            self.rotation_vector = rotation_vector
            self.translation_vector = translation_vector

            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

            # Get Euler angles (in degrees)
            euler_angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
            return euler_angles

        return np.zeros(3)  # Default to no rotation if pose estimation fails

    def _rotation_matrix_to_euler_angles(self, R):
        """Convert rotation matrix to Euler angles (pitch, yaw, roll) in degrees"""
        # Calculate pitch
        pitch = -np.arcsin(R[2, 0]) * 180 / np.pi

        # Calculate yaw
        yaw = np.arctan2(R[2, 1], R[2, 2]) * 180 / np.pi

        # Calculate roll
        roll = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi

        return np.array([pitch, yaw, roll])

    def get_normalized_landmarks(self, landmarks, head_pose, image_shape):
        """Normalize landmarks accounting for head pose"""
        # Convert landmarks to numpy array
        h, w = image_shape
        points = np.array([(lm.x * w, lm.y * h, lm.z) for lm in landmarks])

        # Get rotation angles
        pitch, yaw, roll = head_pose

        # Create rotation matrices
        pitch_rad = pitch * np.pi / 180
        yaw_rad = yaw * np.pi / 180
        roll_rad = roll * np.pi / 180

        # Normalize for head rotation
        # This is a simplified approach - in production, would use proper 3D transformation
        normalized_points = points.copy()

        # Correct for yaw (left-right head rotation)
        if abs(yaw) > 5:  # Only correct if rotation is significant
            # Find center of face
            nose_tip = points[4]  # Nose tip

            # Adjust horizontal positions based on yaw angle
            for i in range(len(points)):
                # Scale factor based on how far the point is from center
                depth_factor = 1.0 + (points[i, 2] - nose_tip[2]) * 0.01
                # Adjust x-coordinate
                correction = np.tan(yaw_rad * 0.5) * depth_factor * w * 0.01
                normalized_points[i, 0] += correction * np.sign(yaw)

        # Correct for roll (tilting head)
        if abs(roll) > 5:  # Only correct if tilt is significant
            # Find center of face
            center = np.mean(points[[4, 152]], axis=0)  # Between nose tip and chin

            # Create rotation matrix for roll
            c, s = np.cos(-roll_rad), np.sin(-roll_rad)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

            # Rotate points around center
            for i in range(len(points)):
                # Vector from center to point
                vec = points[i] - center
                # Apply rotation
                rotated = np.dot(R, vec)
                # Update normalized point
                normalized_points[i] = center + rotated

        return normalized_points


# Class for Multiple Sclerosis feature detection (improved)
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

        # For eye movement tracking (enhanced)
        self.prev_left_eye_center = None
        self.prev_right_eye_center = None
        self.eye_movement_history = []
        self.max_eye_history = 30  # Increased for better detection

        # For nystagmus detection
        self.eye_pos_history = []
        self.max_eye_pos_history = 60  # Store more positions for better FFT

        # Thresholds and calibration
        self.min_nystagmus_freq = 2  # Hz (typical nystagmus range is 2-10 Hz)
        self.max_nystagmus_freq = 10  # Hz

        # For better head pose compensation
        self.head_pose_estimator = HeadPoseEstimator()
        self.last_head_pose = np.zeros(3)

    def extract_features(self, image, landmarks, refined_landmarks=None):
        """Extract facial features relevant to MS detection, with head pose normalization"""
        # Get image dimensions
        h, w = image.shape[:2]

        # Initialize features
        features = {k: 0 for k in self.key_features}
        if landmarks is None:
            return features, False

        # Estimate head pose
        head_pose = self.head_pose_estimator.estimate_pose(image, landmarks)
        self.last_head_pose = head_pose

        # Get normalized landmarks accounting for head pose
        points = self.head_pose_estimator.get_normalized_landmarks(landmarks, head_pose, (h, w))

        # 1. Detect facial paralysis (asymmetry focused on one side)
        features['facial_paralysis_score'] = self._detect_facial_paralysis(points, head_pose)

        # 2. Detect eye movement abnormalities (nystagmus, rapid movements)
        # Use refined landmarks for better eye detection if available
        if refined_landmarks is not None:
            refined_points = np.array([(lm.x * w, lm.y * h, lm.z) for lm in refined_landmarks])
            features['eye_movement_abnormality'] = self._detect_eye_movement_abnormality(refined_points, head_pose)
        else:
            features['eye_movement_abnormality'] = self._detect_eye_movement_abnormality(points, head_pose)

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

    def _detect_facial_paralysis(self, points, head_pose, frame=None):
        """Final version: fine-grained pixel-based scoring for facial paralysis detection"""
        try:
            # Extract Y-coordinates of key landmarks
            left_mouth_y = points[self.mouth_corners[0]][1]
            right_mouth_y = points[self.mouth_corners[1]][1]
            left_brow_y = points[self.eyebrows[0]][1]
            right_brow_y = points[self.eyebrows[1]][1]

            # Absolute differences (in pixels)
            mouth_diff_px = abs(left_mouth_y - right_mouth_y)
            brow_diff_px = abs(left_brow_y - right_brow_y)
            max_diff_px = max(mouth_diff_px, brow_diff_px)

            # Custom ramp: 3 → 0.0, 10 → ~0.7, 20 → ~0.95 (cap)
            if max_diff_px <= 3:
                score = 0.0
            elif max_diff_px >= 20:
                score = 0.95
            else:
                # Smooth exponential scaling from 3 to 20
                score = 0.95 * ((max_diff_px - 3) / 17) ** 1.5

            # Debug output
            print("[Facial Paralysis DEBUG]")
            print(f"  Mouth diff (px): {mouth_diff_px:.2f}")
            print(f"  Brow  diff (px): {brow_diff_px:.2f}")
            print(f"  Max    diff (px): {max_diff_px:.2f}")
            print(f"  Final Score     : {score:.3f}")
            print("-----------------------------")

            # Optional overlay
            if frame is not None:
                for idx in [self.mouth_corners[0], self.mouth_corners[1],
                            self.eyebrows[0], self.eyebrows[1]]:
                    x, y = int(points[idx][0]), int(points[idx][1])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            return score

        except Exception as e:
            print(f"[ERROR] Facial paralysis detection failed: {e}")
            return 0.0

    def _detect_eye_movement_abnormality(self, points, head_pose):
        """
        Detect abnormal eye movements characteristic of MS,
        with improved nystagmus detection
        """
        try:
            # Calculate eye centers
            left_eye_points = [points[i] for i in self.left_eye_landmarks if i < len(points)]
            right_eye_points = [points[i] for i in self.right_eye_landmarks if i < len(points)]

            if not left_eye_points or not right_eye_points:
                return 0.0

            left_eye_center = np.mean(left_eye_points, axis=0)
            right_eye_center = np.mean(right_eye_points, axis=0)

            # Store eye positions for nystagmus analysis
            self.eye_pos_history.append((left_eye_center, right_eye_center))
            if len(self.eye_pos_history) > self.max_eye_pos_history:
                self.eye_pos_history.pop(0)

            # If first frame, initialize history
            if self.prev_left_eye_center is None:
                self.prev_left_eye_center = left_eye_center
                self.prev_right_eye_center = right_eye_center
                return 0.0

            # Calculate eye movement (displacement)
            left_eye_movement = np.linalg.norm(left_eye_center - self.prev_left_eye_center)
            right_eye_movement = np.linalg.norm(right_eye_center - self.prev_right_eye_center)

            # Extract head movement contribution to eye movement
            pitch, yaw, roll = head_pose
            head_movement = np.sqrt(np.sum(np.diff(np.array([self.last_head_pose, head_pose]), axis=0)[0] ** 2))

            # Normalize movement by face size
            face_size = np.linalg.norm(points[self.mouth_corners[1]] - points[self.mouth_corners[0]])
            norm_left_movement = left_eye_movement / (face_size + 1e-6)
            norm_right_movement = right_eye_movement / (face_size + 1e-6)

            # Compensate for head movement
            head_contribution = min(1.0, head_movement * 0.05)
            norm_left_movement = max(0, norm_left_movement - head_contribution)
            norm_right_movement = max(0, norm_right_movement - head_contribution)

            # Track eye movements over time
            self.eye_movement_history.append((norm_left_movement, norm_right_movement))
            if len(self.eye_movement_history) > self.max_eye_history:
                self.eye_movement_history.pop(0)

            # We need enough history for reliable nystagmus detection
            if len(self.eye_pos_history) < 30:
                # Update previous centers
                self.prev_left_eye_center = left_eye_center
                self.prev_right_eye_center = right_eye_center
                return 0.0

            # Extract horizontal and vertical eye position sequences
            left_x = [pos[0][0] for pos in self.eye_pos_history]
            left_y = [pos[0][1] for pos in self.eye_pos_history]
            right_x = [pos[1][0] for pos in self.eye_pos_history]
            right_y = [pos[1][1] for pos in self.eye_pos_history]

            # Detrend and normalize positions
            left_x = signal.detrend(left_x)
            left_y = signal.detrend(left_y)
            right_x = signal.detrend(right_x)
            right_y = signal.detrend(right_y)

            # Apply window function to reduce spectral leakage
            window = signal.windows.hamming(len(left_x))
            left_x_windowed = left_x * window
            left_y_windowed = left_y * window
            right_x_windowed = right_x * window
            right_y_windowed = right_y * window

            # Compute FFT
            sample_rate = 30  # Assumed fps

            # Compute power spectrum
            def compute_power_spectrum(signal_data):
                fft_result = np.abs(np.fft.rfft(signal_data)) ** 2
                freqs = np.fft.rfftfreq(len(signal_data), d=1 / sample_rate)
                return freqs, fft_result

            freqs_lx, psd_lx = compute_power_spectrum(left_x_windowed)
            freqs_ly, psd_ly = compute_power_spectrum(left_y_windowed)
            freqs_rx, psd_rx = compute_power_spectrum(right_x_windowed)
            freqs_ry, psd_ry = compute_power_spectrum(right_y_windowed)

            # Check power in nystagmus frequency range (2-10 Hz)
            nystagmus_mask = (freqs_lx >= self.min_nystagmus_freq) & (freqs_lx <= self.max_nystagmus_freq)

            if np.any(nystagmus_mask):
                # Calculate power in nystagmus range
                nystagmus_power_lx = np.sum(psd_lx[nystagmus_mask])
                nystagmus_power_ly = np.sum(psd_ly[nystagmus_mask])
                nystagmus_power_rx = np.sum(psd_rx[nystagmus_mask])
                nystagmus_power_ry = np.sum(psd_ry[nystagmus_mask])

                # Total power across all frequencies
                total_power_lx = np.sum(psd_lx)
                total_power_ly = np.sum(psd_ly)
                total_power_rx = np.sum(psd_rx)
                total_power_ry = np.sum(psd_ry)

                # Calculate nystagmus probability
                if total_power_lx > 0 and total_power_rx > 0:
                    # Ratio of nystagmus-range power to total power
                    nystagmus_ratio_lx = nystagmus_power_lx / total_power_lx
                    nystagmus_ratio_ly = nystagmus_power_ly / total_power_ly
                    nystagmus_ratio_rx = nystagmus_power_rx / total_power_rx
                    nystagmus_ratio_ry = nystagmus_power_ry / total_power_ry

                    # Find the maximum ratio (strongest nystagmus signal)
                    max_nystagmus_ratio = max(nystagmus_ratio_lx, nystagmus_ratio_ly,
                                              nystagmus_ratio_rx, nystagmus_ratio_ry)

                    # Calculate peak frequency
                    def find_peak_freq(freqs, psd):
                        mask = (freqs >= self.min_nystagmus_freq) & (freqs <= self.max_nystagmus_freq)
                        if np.any(mask) and np.sum(psd[mask]) > 0:
                            peak_idx = np.argmax(psd[mask]) + np.where(mask)[0][0]
                            return freqs[peak_idx]
                        return 0

                    peak_freqs = [
                        find_peak_freq(freqs_lx, psd_lx),
                        find_peak_freq(freqs_ly, psd_ly),
                        find_peak_freq(freqs_rx, psd_rx),
                        find_peak_freq(freqs_ry, psd_ry)
                    ]

                    # Check if there's a consistent peak frequency (typical of nystagmus)
                    peak_freqs = [f for f in peak_freqs if f > 0]
                    freq_consistency = 0.0
                    if peak_freqs:
                        # Calculate how clustered the peak frequencies are
                        mean_peak = np.mean(peak_freqs)
                        freq_std = np.std(peak_freqs)
                        freq_consistency = 1.0 - min(1.0, freq_std / 2.0)

                    # Find peak amplitude in nystagmus range
                    def find_peak_amp(psd, mask):
                        return np.max(psd[mask]) if np.any(mask) and np.sum(psd[mask]) > 0 else 0

                    peak_amps = [
                        find_peak_amp(psd_lx, nystagmus_mask),
                        find_peak_amp(psd_ly, nystagmus_mask),
                        find_peak_amp(psd_rx, nystagmus_mask),
                        find_peak_amp(psd_ry, nystagmus_mask)
                    ]

                    max_peak_amp = max(peak_amps)

                    # Calculate overall intensity of eye movements
                    recent_movements = self.eye_movement_history[-10:]
                    mean_movement = np.mean([max(left, right) for left, right in recent_movements])

                    # Combine metrics for final score
                    # Weight the spectral analysis higher for nystagmus detection
                    nystagmus_score = (
                            max_nystagmus_ratio * 0.4 +  # Strength of signal in nystagmus range
                            freq_consistency * 0.3 +  # Consistency of peak frequencies
                            min(1.0, max_peak_amp * 20) * 0.2 +  # Peak amplitude
                            min(1.0, mean_movement * 10) * 0.1  # Overall movement intensity
                    )

                    # Stronger emphasis on the spectral signature
                    abnormality_score = nystagmus_score

                    # Apply sensitivity boost to make detection more responsive
                    abnormality_score = min(1.0, abnormality_score * 1.5)

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

            # Filter valid indices
            left_upper_lid = [i for i in left_upper_lid if i < len(points)]
            left_lower_lid = [i for i in left_lower_lid if i < len(points)]
            right_upper_lid = [i for i in right_upper_lid if i < len(points)]
            right_lower_lid = [i for i in right_lower_lid if i < len(points)]

            if not left_upper_lid or not left_lower_lid or not right_upper_lid or not right_lower_lid:
                return 0.0

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
            segment_size = 10
            if len(left_twitches) >= segment_size * 3:  # Need at least 3 segments
                left_segments = [left_twitches[i:i + segment_size] for i in range(0, len(left_twitches), segment_size)]
                right_segments = [right_twitches[i:i + segment_size] for i in
                                  range(0, len(right_twitches), segment_size)]

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

                    # Boost detection sensitivity
                    left_myokymia *= 1.5
                    right_myokymia *= 1.5

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

            # Filter valid indices
            left_side_points = [i for i in left_side_points if i < len(points)]
            right_side_points = [i for i in right_side_points if i < len(points)]

            if not left_side_points or not right_side_points:
                return 0.0

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
            right_movements = [np.linalg.norm(curr - prev) for curr, prev in
                               zip(current_right, self.prev_right_positions)]

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
            movement_ratio = max(avg_left_movement, avg_right_movement) / (
                        min(avg_left_movement, avg_right_movement) + 1e-6)
            movement_diff = abs(avg_left_movement - avg_right_movement)

            # Detect which side has more movement
            left_side_affected = avg_left_movement > avg_right_movement
            affected_movements = left_movements if left_side_affected else right_movements

            # Check for spasm pattern (jerky, irregular movements)
            movement_std = np.std(affected_movements)

            # Combine metrics
            spasm_score = movement_diff * 0.4 + movement_std * 0.6

            # Scale for better visibility and increase sensitivity
            spasm_score = min(1.0, spasm_score * 12)

            # Store current positions for next frame
            self.prev_left_positions = current_left
            self.prev_right_positions = current_right

            return min(1.0, max(0.0, spasm_score))

        except Exception as e:
            print(f"Error detecting hemifacial spasm: {e}")
            return 0.0


# Class for Alzheimer's Disease feature detection (improved)
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

        # For head pose compensation
        self.head_pose_estimator = HeadPoseEstimator()

        # For tracking saccades
        self.saccade_history = []
        self.max_saccade_history = 60

        # Base expression detection
        self.expression_baseline = None
        self.expression_variance = []
        self.max_expression_history = 30

    def extract_features(self, image, landmarks):
        """Extract facial features relevant to AD detection, with improved sensitivity"""
        # Get image dimensions
        h, w = image.shape[:2]

        # Initialize features
        features = {k: 0 for k in self.key_features}
        if landmarks is None:
            return features, False

        # Estimate head pose
        head_pose = self.head_pose_estimator.estimate_pose(image, landmarks)

        # Convert landmarks to coordinates
        points = np.array([(lm.x * w, lm.y * h, lm.z) for lm in landmarks])

        # Get normalized landmarks accounting for head pose
        normalized_points = self.head_pose_estimator.get_normalized_landmarks(landmarks, head_pose, (h, w))

        # 1. Detect facial asymmetry (different from MS - focus on specific regions)
        features['facial_asymmetry'] = self._detect_facial_asymmetry(normalized_points, head_pose)

        # 2. Detect reduced expressivity - use normalized points for better accuracy
        features['reduced_expressivity'] = self._detect_reduced_expressivity(normalized_points, points)

        # 3. Detect saccadic eye movements - enhanced detection
        features['saccadic_eye_movement'] = self._detect_saccadic_eye_movement(normalized_points, head_pose)

        # Update history
        for feature in self.key_features:
            self.feature_history[feature].append(features[feature])
            if len(self.feature_history[feature]) > self.history_size:
                self.feature_history[feature].pop(0)

        return features, True

    def _detect_facial_asymmetry(self, points, head_pose):
        """
        Detect facial asymmetry focused on AD-specific regions
        with compensation for head pose
        """
        try:
            # Extract head pose angles
            pitch, yaw, roll = head_pose

            # Adjust sensitivity based on head pose
            # to reduce false positives during head rotation
            pose_factor = 1.0
            if abs(yaw) > 15:  # Significant left-right head rotation
                pose_factor = max(0.3, 1.0 - (abs(yaw) - 15) / 45)
            if abs(roll) > 15:  # Significant head tilt
                pose_factor = min(pose_factor, max(0.3, 1.0 - (abs(roll) - 15) / 45))

            # Calculate asymmetry scores for each facial region

            # 1. Face edge asymmetry
            left_edge_idx = self.face_edges[0] if self.face_edges[0] < len(points) else 234
            right_edge_idx = self.face_edges[1] if self.face_edges[1] < len(points) else 454

            left_edge = points[left_edge_idx]
            right_edge = points[right_edge_idx]
            center_x = (left_edge[0] + right_edge[0]) / 2
            edge_asymmetry = abs((center_x - left_edge[0]) - (right_edge[0] - center_x)) / (
                        right_edge[0] - left_edge[0] + 1e-6)

            # 2. Eyebrow asymmetry
            left_brow_idx = self.eyebrows[0] if self.eyebrows[0] < len(points) else 105
            right_brow_idx = self.eyebrows[1] if self.eyebrows[1] < len(points) else 334

            left_brow = points[left_brow_idx]
            right_brow = points[right_brow_idx]
            brow_height_diff = abs(left_brow[1] - right_brow[1])

            # Measure face height more reliably
            forehead_idx = 10 if 10 < len(points) else 0
            chin_idx = 152 if 152 < len(points) else len(points) - 1
            face_height = abs(points[forehead_idx][1] - points[chin_idx][1])

            brow_asymmetry = brow_height_diff / (face_height + 1e-6)

            # 3. Eye asymmetry
            left_eye_idx = self.eyes[0] if self.eyes[0] < len(points) else 159
            right_eye_idx = self.eyes[1] if self.eyes[1] < len(points) else 386

            left_eye = points[left_eye_idx]
            right_eye = points[right_eye_idx]
            eye_height_diff = abs(left_eye[1] - right_eye[1])
            eye_asymmetry = eye_height_diff / (face_height + 1e-6)

            # 4. Nostril asymmetry
            left_nostril_idx = self.nostrils[0] if self.nostrils[0] < len(points) else 203
            right_nostril_idx = self.nostrils[1] if self.nostrils[1] < len(points) else 423

            left_nostril = points[left_nostril_idx]
            right_nostril = points[right_nostril_idx]
            nostril_diff = abs(left_nostril[1] - right_nostril[1])
            nostril_asymmetry = nostril_diff / (face_height + 1e-6)

            # 5. Mouth corner asymmetry
            left_mouth_idx = self.mouth_corners[0] if self.mouth_corners[0] < len(points) else 61
            right_mouth_idx = self.mouth_corners[1] if self.mouth_corners[1] < len(points) else 291

            left_mouth = points[left_mouth_idx]
            right_mouth = points[right_mouth_idx]
            mouth_height_diff = abs(left_mouth[1] - right_mouth[1])
            mouth_asymmetry = mouth_height_diff / (face_height + 1e-6)

            # Combined asymmetry score with weights based on research
            # Calculate individual components with increased sensitivity
            edge_component = min(1.0, edge_asymmetry * 10)
            brow_component = min(1.0, brow_asymmetry * 12)
            eye_component = min(1.0, eye_asymmetry * 15)
            nostril_component = min(1.0, nostril_asymmetry * 12)
            mouth_component = min(1.0, mouth_asymmetry * 10)

            # Combine with research-based weights
            asymmetry_score = (
                    edge_component * 0.15 +
                    brow_component * 0.25 +
                    eye_component * 0.3 +
                    nostril_component * 0.1 +
                    mouth_component * 0.2
            )

            # Apply head pose factor to reduce false positives during head rotation
            asymmetry_score *= pose_factor

            # Apply enhanced sensitivity scaling
            asymmetry_score = min(1.0, asymmetry_score * 1.5)

            # Ensure asymmetry score is at least 0.2 if any component is significant
            max_component = max(edge_component, brow_component, eye_component,
                                nostril_component, mouth_component)
            if max_component > 0.5 and asymmetry_score < 0.2:
                asymmetry_score = max(asymmetry_score, 0.2 + (max_component - 0.5) * 0.2)

            return asymmetry_score
        except Exception as e:
            print(f"Error detecting facial asymmetry: {e}")
            return 0.0

    def _detect_reduced_expressivity(self, normalized_points, original_points):
        """
        Detect reduced expressivity similar to facial masking in PD
        but with AD-specific focus and improved neutral face detection
        """
        try:
            # Get key facial landmarks
            mouth_left_idx = self.mouth_corners[0] if self.mouth_corners[0] < len(normalized_points) else 61
            mouth_right_idx = self.mouth_corners[1] if self.mouth_corners[1] < len(normalized_points) else 291

            mouth_left = normalized_points[mouth_left_idx]
            mouth_right = normalized_points[mouth_right_idx]

            # Calculate mouth width
            mouth_width = np.linalg.norm(mouth_right - mouth_left)

            # Calculate vertical mouth opening
            upper_lip_idx = 13 if 13 < len(normalized_points) else 0
            lower_lip_idx = 14 if 14 < len(normalized_points) else 17

            mouth_top = normalized_points[upper_lip_idx]
            mouth_bottom = normalized_points[lower_lip_idx]
            mouth_height = np.linalg.norm(mouth_top - mouth_bottom)

            # Calculate eyebrow movement
            left_brow_idx = self.eyebrows[0] if self.eyebrows[0] < len(normalized_points) else 105
            right_brow_idx = self.eyebrows[1] if self.eyebrows[1] < len(normalized_points) else 334

            left_brow = normalized_points[left_brow_idx]
            right_brow = normalized_points[right_brow_idx]
            brow_distance = np.linalg.norm(left_brow - right_brow)

            # Calculate additional expressive features
            # Eye openness
            left_eye_top_idx = 159 if 159 < len(normalized_points) else 386
            left_eye_bottom_idx = 145 if 145 < len(normalized_points) else 374
            right_eye_top_idx = 386 if 386 < len(normalized_points) else 159
            right_eye_bottom_idx = 374 if 374 < len(normalized_points) else 145

            left_eye_top = normalized_points[left_eye_top_idx]
            left_eye_bottom = normalized_points[left_eye_bottom_idx]
            right_eye_top = normalized_points[right_eye_top_idx]
            right_eye_bottom = normalized_points[right_eye_bottom_idx]

            left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
            right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
            eye_openness = (left_eye_height + right_eye_height) / 2

            # Create feature vector for current expression
            expression_features = np.array([
                mouth_width, mouth_height, brow_distance, eye_openness
            ])

            # Initialize neutral expression if not established
            if not self.neutral_established:
                self.neutral_landmarks = original_points.copy()
                self.expression_baseline = expression_features
                self.neutral_established = True
                return 0.5  # Default value if neutral face not established

            # Track expression variance over time
            if self.expression_baseline is not None:
                # Calculate relative change from baseline
                relative_change = np.abs(expression_features - self.expression_baseline) / (
                            self.expression_baseline + 1e-6)

                # Add to variance history
                self.expression_variance.append(np.mean(relative_change))
                if len(self.expression_variance) > self.max_expression_history:
                    self.expression_variance.pop(0)

                # Calculate expressivity based on:
                # 1. Current deviation from neutral
                current_expressivity = np.mean(relative_change)

                # 2. Variance of expression over time (low variance = reduced expressivity)
                if len(self.expression_variance) > 10:
                    expression_variance_score = np.std(self.expression_variance)

                    # Combine metrics (higher scores = more expressivity)
                    expressivity_score = current_expressivity * 0.4 + expression_variance_score * 0.6

                    # Convert to reduced expressivity (higher = reduced expression = AD indicator)
                    # Apply enhanced sensitivity
                    reduced_expressivity = 1.0 - min(1.0, expressivity_score * 5.0)

                    # Adaptive baseline update (slow drift toward current expression)
                    # This helps adapt to the person's natural expression
                    adaptation_rate = 0.05
                    self.expression_baseline = (
                                                           1 - adaptation_rate) * self.expression_baseline + adaptation_rate * expression_features

                    return reduced_expressivity
                else:
                    # Gradual adaptation to baseline
                    self.expression_baseline = 0.9 * self.expression_baseline + 0.1 * expression_features
                    return 0.5  # Default until we have enough history

            return 0.5  # Default fallback
        except Exception as e:
            print(f"Error detecting reduced expressivity: {e}")
            return 0.5

    def _detect_saccadic_eye_movement(self, points, head_pose):
        """
        Detect saccadic eye movements characteristic of AD
        with improved sensitivity and head pose compensation
        """
        try:
            # Calculate eye centers using more reliable indices
            left_eye_center_idx = 468 if 468 < len(points) else 159
            right_eye_center_idx = 473 if 473 < len(points) else 386

            left_eye_center = points[left_eye_center_idx]
            right_eye_center = points[right_eye_center_idx]

            # Extract head pose angles for compensation
            pitch, yaw, roll = head_pose

            # Store eye positions for analysis
            if self.prev_left_eye is not None and self.prev_right_eye is not None:
                # Calculate raw eye movement velocity
                left_velocity = np.linalg.norm(left_eye_center - self.prev_left_eye)
                right_velocity = np.linalg.norm(right_eye_center - self.prev_right_eye)

                # Normalize by face size
                face_width = np.linalg.norm(points[self.face_edges[1]] - points[self.face_edges[0]])
                norm_left_vel = left_velocity / (face_width + 1e-6)
                norm_right_vel = right_velocity / (face_width + 1e-6)

                # Compensate for head movement
                # Large values of yaw/roll/pitch indicate head movement that could be
                # misinterpreted as eye movement
                head_movement_factor = np.sqrt(np.sum(np.array([pitch, yaw, roll]) ** 2)) / 50.0
                compensation = min(0.8, head_movement_factor)

                # Apply compensation (reduce velocity if head is moving)
                norm_left_vel = max(0.0, norm_left_vel - compensation * norm_left_vel)
                norm_right_vel = max(0.0, norm_right_vel - compensation * norm_right_vel)

                # Store normalized velocities and positions
                self.eye_positions.append((norm_left_vel, norm_right_vel, left_eye_center, right_eye_center))
                if len(self.eye_positions) > self.max_eye_positions:
                    self.eye_positions.pop(0)

                # Store for saccade detection
                self.saccade_history.append((norm_left_vel, norm_right_vel))
                if len(self.saccade_history) > self.max_saccade_history:
                    self.saccade_history.pop(0)

            # Update previous positions
            self.prev_left_eye = left_eye_center
            self.prev_right_eye = right_eye_center

            # Need sufficient history for analysis
            if len(self.saccade_history) < 30:
                return 0.0

            # Extract velocity data
            left_velocities = [pos[0] for pos in self.saccade_history]
            right_velocities = [pos[1] for pos in self.saccade_history]

            # For AD, saccadic eye movements are characterized by:
            # 1. Longer latency to initiate saccades (delay)
            # 2. More step-like movements rather than smooth pursuits
            # 3. Increased directional errors

            # Calculate metrics specific to AD saccades

            # 1. Detect velocity peaks (potential saccades)
            # AD typically has fewer high-velocity saccades
            mean_velocity = np.mean(left_velocities + right_velocities)
            velocity_threshold = mean_velocity * 2.0

            left_peaks = [v > velocity_threshold for v in left_velocities]
            right_peaks = [v > velocity_threshold for v in right_velocities]

            # Count peaks (fewer peaks may indicate AD)
            left_peak_count = sum(left_peaks)
            right_peak_count = sum(right_peaks)
            total_peak_count = left_peak_count + right_peak_count

            # 2. Calculate saccade initiation (latency)
            # Look for episodes of low movement followed by sudden movement
            latency_episodes = 0
            for i in range(5, len(left_velocities)):
                # Check for periods of low movement followed by sudden movement
                prev_window = left_velocities[i - 5:i]
                if max(prev_window) < velocity_threshold * 0.5 and left_velocities[i] > velocity_threshold:
                    latency_episodes += 1

                prev_window = right_velocities[i - 5:i]
                if max(prev_window) < velocity_threshold * 0.5 and right_velocities[i] > velocity_threshold:
                    latency_episodes += 1

            # Normalize latency episodes
            norm_latency = min(1.0, latency_episodes / 10.0)

            # 3. Calculate "step-like" nature of movements (characteristic of AD)
            # AD tends to have more abrupt movements rather than smooth pursuits
            left_diff = np.abs(np.diff(left_velocities))
            right_diff = np.abs(np.diff(right_velocities))

            # Calculate "jerkiness" - higher is more characteristic of AD
            left_jerkiness = np.mean(left_diff) / (mean_velocity + 1e-6)
            right_jerkiness = np.mean(right_diff) / (mean_velocity + 1e-6)

            avg_jerkiness = (left_jerkiness + right_jerkiness) / 2.0

            # 4. Calculate directional changes (characteristic of AD)
            # AD tends to have more directional errors in saccades
            direction_changes = 0

            # If we have positional data
            if len(self.eye_positions) > 10:
                # Extract trajectories
                left_x = [pos[2][0] for pos in self.eye_positions[-10:]]
                left_y = [pos[2][1] for pos in self.eye_positions[-10:]]
                right_x = [pos[3][0] for pos in self.eye_positions[-10:]]
                right_y = [pos[3][1] for pos in self.eye_positions[-10:]]

                # Calculate direction changes
                for i in range(2, len(left_x)):
                    # Left eye
                    prev_dir_x = left_x[i - 1] - left_x[i - 2]
                    curr_dir_x = left_x[i] - left_x[i - 1]
                    prev_dir_y = left_y[i - 1] - left_y[i - 2]
                    curr_dir_y = left_y[i] - left_y[i - 1]

                    # Check if direction changed significantly
                    if prev_dir_x * curr_dir_x < 0 or prev_dir_y * curr_dir_y < 0:
                        direction_changes += 1

                    # Right eye
                    prev_dir_x = right_x[i - 1] - right_x[i - 2]
                    curr_dir_x = right_x[i] - right_x[i - 1]
                    prev_dir_y = right_y[i - 1] - right_y[i - 2]
                    curr_dir_y = right_y[i] - right_y[i - 1]

                    if prev_dir_x * curr_dir_x < 0 or prev_dir_y * curr_dir_y < 0:
                        direction_changes += 1

            # Normalize direction changes
            norm_direction_changes = min(1.0, direction_changes / 15.0)

            # Combine metrics for AD saccade score
            # Weight factors based on clinical significance for AD
            saccade_score = (
                    norm_latency * 0.35 +  # Delayed saccade initiation
                    avg_jerkiness * 0.3 +  # Step-like movements
                    norm_direction_changes * 0.35  # Directional errors
            )

            # Apply sensitivity enhancement
            saccade_score = min(1.0, saccade_score * 1.7)

            return saccade_score

        except Exception as e:
            print(f"Error detecting saccadic eye movement: {e}")
            return 0.0


# Combined Neurological Disorder Detection System (improved)
class NeurologicalDisorderDetectionSystem:
    def __init__(self, pd_model_path=None):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.3
        )

        # Initialize MediaPipe with refined landmarks for better eye tracking
        self.mp_face_mesh_refined = mp.solutions.face_mesh
        self.face_mesh_refined = self.mp_face_mesh_refined.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.4,
            refine_landmarks=True  # Enable refined landmarks for better eye tracking
        )

        # Initialize individual disorder detectors
        self.pd_system = PDDetectionSystem(pd_model_path)
        self.ms_feature_extractor = MSFeatureExtractor()
        self.ad_feature_extractor = ADFeatureExtractor()

        # Store recent results for stability
        self.ms_recent_results = []
        self.ad_recent_results = []
        self.max_results_history = 15  # Increased for better stability

        # Store head pose history for compensation
        self.head_pose_history = []
        self.max_head_history = 10

        # Head pose estimator
        self.head_pose_estimator = HeadPoseEstimator()

    def process_frame(self, frame):
        """Process a frame to detect multiple neurological disorders"""
        # Get frame dimensions
        h, w = frame.shape[:2]

        # Process with MediaPipe for landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        # Process with refined MediaPipe model for better eye tracking
        refined_results = self.face_mesh_refined.process(rgb_frame)

        # Initialize result dictionary
        result = {
            'face_detected': False,
            'pd': {'probability': 0.0, 'likelihood': 'Unknown', 'features': None},
            'ms': {'probability': 0.0, 'likelihood': 'Unknown', 'features': None},
            'ad': {'probability': 0.0, 'likelihood': 'Unknown', 'features': None},
            'head_pose': None
        }

        # Check if face was detected
        if not results.multi_face_landmarks:
            return result

        # Face was detected
        result['face_detected'] = True
        landmarks = results.multi_face_landmarks[0].landmark

        # Get refined landmarks if available
        refined_landmarks = None
        if refined_results.multi_face_landmarks:
            refined_landmarks = refined_results.multi_face_landmarks[0].landmark

        # Estimate head pose
        head_pose = self.head_pose_estimator.estimate_pose(frame, landmarks)
        result['head_pose'] = head_pose

        # Store head pose history
        self.head_pose_history.append(head_pose)
        if len(self.head_pose_history) > self.max_head_history:
            self.head_pose_history.pop(0)

        # Calculate average head pose for stability
        avg_head_pose = np.mean(self.head_pose_history, axis=0) if self.head_pose_history else head_pose

        # Get PD detection results
        pd_result = self.pd_system.process_frame(frame)
        result['pd'] = {
            'probability': pd_result['pd_probability'],
            'likelihood': pd_result['pd_likelihood'],
            'features': pd_result['features']
        }

        # Extract MS features (with head pose compensation)
        ms_features, _ = self.ms_feature_extractor.extract_features(frame, landmarks, refined_landmarks)

        # Calculate MS probability with adjustments for head pose
        ms_probability = self._calculate_ms_probability(ms_features, head_pose)

        # Add to recent results for stability
        self.ms_recent_results.append(ms_probability)
        if len(self.ms_recent_results) > self.max_results_history:
            self.ms_recent_results.pop(0)

        # Use stabilized probability (using median to filter outliers)
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

        # Extract AD features (with head pose compensation)
        ad_features, _ = self.ad_feature_extractor.extract_features(frame, landmarks)

        # Calculate AD probability with adjustments for head pose
        ad_probability = self._calculate_ad_probability(ad_features, head_pose)

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

    def _calculate_ms_probability(self, features, head_pose):
        """
        Calculate MS probability from extracted features
        with adjustments for head pose
        """
        try:
            # Extract head pose angles
            pitch, yaw, roll = head_pose

            # Calculate head movement factor (large movements should reduce detection confidence)
            head_movement_factor = 1.0
            if abs(yaw) > 15 or abs(roll) > 15 or abs(pitch) > 15:
                # Reduce confidence during significant head movement
                angles_sum = abs(yaw) + abs(roll) + abs(pitch)
                head_movement_factor = max(0.6, 1.0 - (angles_sum - 15) / 100)

            # Get individual feature scores
            facial_paralysis = features['facial_paralysis_score']
            eye_movement = features['eye_movement_abnormality']
            eyelid_twitching = features['eyelid_twitching']
            hemifacial_spasm = features['hemifacial_spasm']

            # Weighted combination based on clinical significance
            # Eye movement abnormalities and facial paralysis are stronger indicators
            ms_probability = (
                    facial_paralysis * 0.25 +
                    eye_movement * 0.4 +  # Increased weight for jerky eye movements
                    eyelid_twitching * 0.25 +
                    hemifacial_spasm * 0.1
            )

            # Apply head movement factor
            ms_probability *= head_movement_factor

            # Apply improved threshold calibration with higher sensitivity
            ms_probability = max(0.0, min(1.0, (ms_probability - 0.10) * 1.3))

            return ms_probability
        except Exception as e:
            print(f"Error calculating MS probability: {e}")
            return 0.0

    def _calculate_ad_probability(self, features, head_pose):
        """
        Calculate AD probability from extracted features
        with adjustments for head pose
        """
        try:
            # Extract head pose angles
            pitch, yaw, roll = head_pose

            # Calculate head movement factor
            head_movement_factor = 1.0
            if abs(yaw) > 15 or abs(roll) > 15 or abs(pitch) > 15:
                # Reduce confidence during significant head movement
                angles_sum = abs(yaw) + abs(roll) + abs(pitch)
                head_movement_factor = max(0.6, 1.0 - (angles_sum - 15) / 100)

            # Get individual feature scores
            facial_asymmetry = features['facial_asymmetry']
            reduced_expressivity = features['reduced_expressivity']
            saccadic_eye_movement = features['saccadic_eye_movement']

            # Weighted combination based on research
            # Boosted weights for better detection
            ad_probability = (
                    facial_asymmetry * 0.3 +
                    reduced_expressivity * 0.35 +
                    saccadic_eye_movement * 0.35
            )

            # Apply head movement factor
            ad_probability *= head_movement_factor

            # Apply enhanced sensitivity calibration
            ad_probability = max(0.0, min(1.0, (ad_probability - 0.15) * 1.4))

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

            # Display head pose if available
            if result['head_pose'] is not None:
                pitch, yaw, roll = result['head_pose']
                pose_text = f"Head: Pitch {pitch:.1f}, Yaw {yaw:.1f}, Roll {roll:.1f}"
                cv2.putText(frame, pose_text,
                            (frame.shape[1] - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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
            if ms_prob > 0.25:  # Lowered threshold for better visualization
                ms_features = result['ms']['features']
                y_pos += 10
                cv2.putText(frame, "MS Indicators:", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ms_color, 1)
                y_pos += 20

                indicators = []
                if ms_features['facial_paralysis_score'] > 0.3:
                    indicators.append("Facial paralysis detected")
                if ms_features['eye_movement_abnormality'] > 0.3:
                    indicators.append("Abnormal eye movements (nystagmus)")
                if ms_features['eyelid_twitching'] > 0.3:
                    indicators.append("Eyelid twitching (myokymia)")
                if ms_features['hemifacial_spasm'] > 0.3:
                    indicators.append("Hemifacial spasm detected")

                for indicator in indicators:
                    cv2.putText(frame, f"- {indicator}", (20, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ms_color, 1)
                    y_pos += 20

            # Display AD-specific indicators when AD probability is significant
            if ad_prob > 0.25:  # Lowered threshold for better visualization
                ad_features = result['ad']['features']
                y_pos += 10
                cv2.putText(frame, "AD Indicators:", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ad_color, 1)
                y_pos += 20

                indicators = []
                if ad_features['facial_asymmetry'] > 0.3:
                    indicators.append("Facial asymmetry")
                if ad_features['reduced_expressivity'] > 0.5:
                    indicators.append("Reduced facial expressivity")
                if ad_features['saccadic_eye_movement'] > 0.4:
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
                cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 60),
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
