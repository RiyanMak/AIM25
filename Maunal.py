import cv2
import numpy as np
import torch
import torch.nn as nn
import time
import os
import mediapipe as mp

# Initialize MediaPipe Face Mesh with more forgiving parameters
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,  # Set to False for video
    max_num_faces=1,
    min_detection_confidence=0.3)  # Lowered from 0.5 to improve detection rate

# Check CUDA/MPS availability
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
print(f"Using device: {device}")

class FacialActionUnitExtractor:
    def __init__(self):
        # Key AU landmarks for PD detection
        self.au_landmarks = {
            'AU1': [10, 338],   # Inner brow raiser
            'AU2': [65, 295],   # Outer brow raiser
            'AU4': [9, 337],    # Brow lowerer - key for PD
            'AU6': [117, 346],  # Cheek raiser - key for PD
            'AU7': [159, 386],  # Lid tightener
            'AU9': [129, 358],  # Nose wrinkler
            'AU12': [61, 291],  # Lip corner puller (smile) - key for PD
            'AU15': [61, 291],  # Lip corner depressor
            'AU20': [0, 267]    # Lip stretcher
        }
    
    def extract_aus(self, image):
        """Extract facial action units with focus on PD-relevant features"""
        # Convert image to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb_image.shape[:2]  # Image dimensions for correct scaling
        
        # Process the image to find facial landmarks
        results = face_mesh.process(rgb_image)
        
        # Initialize AU values - 9 core AUs + 4 PD-specific measurements
        aus = np.zeros(13)
        
        # Check if face was detected
        if not results.multi_face_landmarks:
            return aus, False  # Return zeros and False for detection failure
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        try:
            # Convert landmarks to numpy array with correct scaling
            points = np.array([(lm.x * w, lm.y * h, lm.z) for lm in landmarks])
            
            # Extract key PD-related AUs
            # 1. AU6 (Cheek Raiser) - key for PD detection
            cheek_raise_left = np.linalg.norm(points[117] - points[123])
            cheek_raise_right = np.linalg.norm(points[346] - points[352])
            aus[0] = (cheek_raise_left + cheek_raise_right) / 2
            
            # 2. AU12 (Lip Corner Puller) - key for PD detection
            mouth_left = points[61]
            mouth_right = points[291]
            mouth_top = points[13]
            mouth_bottom = points[14]
            mouth_center = (mouth_top + mouth_bottom) / 2
            smile_measure = mouth_center[1] - (mouth_left[1] + mouth_right[1])/2
            aus[1] = smile_measure
            
            # 3. AU4 (Brow Lowerer) - key for PD detection
            brow_lower_left = np.linalg.norm(points[9] - points[107])
            brow_lower_right = np.linalg.norm(points[337] - points[336])
            aus[2] = (brow_lower_left + brow_lower_right) / 2
            
            # 4-9. Other AUs (simplified for brevity)
            aus[3] = np.linalg.norm(points[10] - points[338])  # Inner brow raise
            aus[4] = np.linalg.norm(points[65] - points[295])  # Outer brow raise
            aus[5] = np.linalg.norm(points[159] - points[145]) + np.linalg.norm(points[386] - points[374])  # Lid tightener
            aus[6] = np.linalg.norm(points[129] - points[358])  # Nose wrinkle
            aus[7] = np.linalg.norm(points[61] - points[291])  # Lip corner depress
            aus[8] = np.linalg.norm(points[0] - points[267])  # Lip stretch
            
            # PD-specific measurements
            # 10. Smile symmetry - PD often has asymmetrical facial expressions
            left_smile = np.linalg.norm(mouth_left - mouth_top)
            right_smile = np.linalg.norm(mouth_right - mouth_top)
            # Add epsilon to prevent division by zero
            epsilon = 1e-6
            smile_asymmetry = abs(left_smile - right_smile) / (max(left_smile, right_smile) + epsilon)
            aus[9] = smile_asymmetry
            
            # 11. Blink rate approximation (eye openness)
            left_eye_open = np.linalg.norm(points[159] - points[145])
            right_eye_open = np.linalg.norm(points[386] - points[374])
            eye_openness = (left_eye_open + right_eye_open) / 2
            aus[10] = eye_openness
            
            # 12. Overall facial mobility (average of key movements)
            facial_mobility = (max(0, aus[0]) + max(0, abs(aus[1])) + max(0, aus[2])) / 3
            aus[11] = facial_mobility
            
            # 13. Mouth corner resting position (hypomimia indicator)
            mouth_corner_rest = (mouth_left[1] + mouth_right[1]) / 2
            aus[12] = mouth_corner_rest
            
            return aus, True  # Return AUs and True for successful detection
            
        except Exception as e:
            print(f"Error extracting AUs: {e}")
            return aus, False  # Return zeros and False for extraction failure

class SimplePDModel(nn.Module):
    """Simple model for PD detection from facial features"""
    def __init__(self, input_size=13):
        super(SimplePDModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

def run_live_detection():
    """Run live PD detection using webcam with a heuristic approach (no trained models)"""
    # Initialize facial feature extractor
    au_extractor = FacialActionUnitExtractor()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    # For temporal features
    frame_buffer = []
    buffer_size = 30  # Buffer for 30 frames
    
    print("PD Detection started. Press 'q' to quit.")
    
    start_time = time.time()
    frame_count = 0
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
            
        # Resize for processing
        process_frame = cv2.resize(frame, (224, 224))
        
        # Extract AUs
        aus, detected = au_extractor.extract_aus(process_frame)
        
        # Add to buffer
        frame_buffer.append(aus)
        if len(frame_buffer) > buffer_size:
            frame_buffer.pop(0)
            
        # If buffer is full and face detected, make prediction
        if len(frame_buffer) == buffer_size and detected:
            # Calculate variances
            au_variances = np.var(frame_buffer, axis=0)
            
            # Heuristic PD detection (since no trained models)
            # Higher values indicate potential PD signs
            pd_indicators = {
                'smile_asymmetry': au_variances[9] * 5.0,  # Amplify asymmetry importance
                'facial_mobility': 0.1 / (au_variances[11] + 0.001),  # Lower mobility = higher PD indicator
                'expression_variance': 0.1 / (np.mean(au_variances[:3]) + 0.001),  # Lower expression variance = higher PD indicator
                'blink_variance': 0.1 / (au_variances[10] + 0.001)  # Lower blink variance = higher PD indicator
            }
            
            # Calculate combined score
            pd_score = (pd_indicators['smile_asymmetry'] + 
                        pd_indicators['facial_mobility'] + 
                        pd_indicators['expression_variance'] + 
                        pd_indicators['blink_variance']) / 4.0
            
            # Normalize to 0-1 range
            pd_probability = min(1.0, max(0.0, pd_score / 10.0))
            
            # Determine likelihood category
            if pd_probability > 0.7:
                likelihood = "High"
                color = (0, 0, 255)  # Red
            elif pd_probability > 0.3:
                likelihood = "Medium"
                color = (0, 165, 255)  # Orange
            else:
                likelihood = "Low"
                color = (0, 255, 0)  # Green
            
            # Display prediction on frame
            cv2.putText(frame, f"PD Probability: {pd_probability:.2f} ({likelihood})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display key features
            features = {
                'Cheek Raiser': au_variances[0],
                'Lip Corner Puller': au_variances[1],
                'Brow Lowerer': au_variances[2],
                'Smile Asymmetry': au_variances[9],
                'Facial Mobility': au_variances[11]
            }
            
            y_pos = 60
            for name, value in features.items():
                # Normalize feature value for display
                norm_value = min(1.0, max(0.0, value / 0.05))
                feature_text = f"{name}: {value:.4f}"
                cv2.putText(frame, feature_text, (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw bar representation
                bar_length = int(150 * norm_value)
                cv2.rectangle(frame, (200, y_pos-15), (200+bar_length, y_pos-5), color, -1)
                
                y_pos += 25
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
                cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-120, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # Display status when face not detected or buffer not full
            if not detected:
                cv2.putText(frame, "Face not detected", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif len(frame_buffer) < buffer_size:
                cv2.putText(frame, f"Buffering: {len(frame_buffer)}/{buffer_size}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Parkinson\'s Disease Detection', frame)
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run the live detection
if __name__ == "__main__":
    run_live_detection()