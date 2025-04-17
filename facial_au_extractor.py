import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5)

class FacialActionUnitExtractor:
    def __init__(self):
        # Key AU landmarks for PD detection based on the paper
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
        
        # PD-specific facial measurements
        self.pd_measurements = [
            'smile_symmetry',       # Asymmetry in smile - PD indicator
            'blink_rate',           # Reduced blink rate - PD indicator
            'facial_mobility',      # Reduced overall mobility - PD indicator
            'expression_transition' # Slow transitions - PD indicator
        ]
    
    def extract_aus(self, image):
        """Extract facial action units with focus on PD-relevant features"""
        # Convert image to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb_image.shape[:2]  # Image dimensions for correct scaling
        
        # Process the image to find facial landmarks
        results = face_mesh.process(rgb_image)
        
        # Initialize AU values - 9 core AUs + 4 PD-specific measurements
        aus = np.zeros(13)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
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
            
            # 4. AU1 (Inner Brow Raiser)
            inner_brow_raise = np.linalg.norm(points[10] - points[338])
            aus[3] = inner_brow_raise
            
            # 5. AU2 (Outer Brow Raiser)
            outer_brow_raise = np.linalg.norm(points[65] - points[295])
            aus[4] = outer_brow_raise
            
            # 6. AU7 (Lid Tightener)
            lid_tighten_left = np.linalg.norm(points[159] - points[145])
            lid_tighten_right = np.linalg.norm(points[386] - points[374])
            aus[5] = (lid_tighten_left + lid_tighten_right) / 2
            
            # 7. AU9 (Nose Wrinkler)
            nose_wrinkle = np.linalg.norm(points[129] - points[358])
            aus[6] = nose_wrinkle
            
            # 8. AU15 (Lip Corner Depressor)
            lip_corner_depress = np.linalg.norm(points[61] - points[291])
            aus[7] = lip_corner_depress
            
            # 9. AU20 (Lip Stretcher)
            lip_stretch = np.linalg.norm(points[0] - points[267])
            aus[8] = lip_stretch
            
            # PD-specific measurements
            # 10. Smile symmetry - PD often has asymmetrical facial expressions
            left_smile = np.linalg.norm(mouth_left - mouth_top)
            right_smile = np.linalg.norm(mouth_right - mouth_top)
            smile_asymmetry = abs(left_smile - right_smile) / max(left_smile, right_smile)
            aus[9] = smile_asymmetry
            
            # 11. Blink rate approximation (eye openness)
            left_eye_open = np.linalg.norm(points[159] - points[145])
            right_eye_open = np.linalg.norm(points[386] - points[374])
            eye_openness = (left_eye_open + right_eye_open) / 2
            aus[10] = eye_openness
            
            # 12. Overall facial mobility (average movement potential)
            facial_mobility = (aus[0] + aus[1] + aus[2]) / 3  # Average of key AUs
            aus[11] = facial_mobility
            
            # 13. Mouth corner resting position (hypomimia indicator)
            mouth_corner_rest = (mouth_left[1] + mouth_right[1]) / 2
            aus[12] = mouth_corner_rest
            
        return aus