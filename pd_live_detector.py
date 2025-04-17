import cv2
import torch
import numpy as np
import pickle
import time
from facial_au_extractor import FacialActionUnitExtractor
from pd_models import SimplePDModel, TemporalPDModel

def load_models():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    
    print(f"Using device: {device}")
    
    # Load models
    try:
        # Random Forest
        with open('rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        
        # Simple NN model
        simple_model = SimplePDModel(input_size=13)
        simple_model.load_state_dict(torch.load('simple_pd_model.pth', map_location=device))
        simple_model.to(device)
        simple_model.eval()
        
        # Temporal model
        temporal_model = TemporalPDModel(input_size=13, hidden_size=64, num_layers=2)
        temporal_model.load_state_dict(torch.load('temporal_pd_model.pth', map_location=device))
        temporal_model.to(device)
        temporal_model.eval()
        
        return rf_model, simple_model, temporal_model, device
    
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, device

def run_live_detection():
    # Initialize feature extractor
    au_extractor = FacialActionUnitExtractor()
    
    # Load models
    rf_model, simple_model, temporal_model, device = load_models()
    
    if None in (rf_model, simple_model, temporal_model):
        print("Failed to load one or more models. Exiting.")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    # For temporal features
    frame_buffer = []
    buffer_size = 30  # Same as in training
    
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
        aus = au_extractor.extract_aus(process_frame)
        
        # Add to buffer
        frame_buffer.append(aus)
        if len(frame_buffer) > buffer_size:
            frame_buffer.pop(0)
            
        # If buffer is full, make prediction
        if len(frame_buffer) == buffer_size:
            # Calculate variances
            au_variances = np.var(frame_buffer, axis=0)
            
            # Make predictions
            # 1. Random Forest
            rf_pred = rf_model.predict_proba([au_variances])[0, 1]
            
            # 2. Simple Model
            with torch.no_grad():
                simple_input = torch.FloatTensor(au_variances).to(device)
                simple_pred = simple_model(simple_input).item()
            
            # 3. Temporal Model
            with torch.no_grad():
                sequence_np = np.array(frame_buffer)
                temporal_input = torch.FloatTensor(sequence_np).unsqueeze(0).to(device)
                temporal_pred = temporal_model(temporal_input).item()
            
            # Ensemble prediction
            ensemble_pred = (rf_pred * 0.4 + simple_pred * 0.3 + temporal_pred * 0.3)
            
            # Determine likelihood category
            if ensemble_pred > 0.7:
                likelihood = "High"
                color = (0, 0, 255)  # Red
            elif ensemble_pred > 0.3:
                likelihood = "Medium"
                color = (0, 165, 255)  # Orange
            else:
                likelihood = "Low"
                color = (0, 255, 0)  # Green
            
            # Display prediction on frame
            cv2.putText(frame, f"PD Probability: {ensemble_pred:.2f} ({likelihood})", 
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
                norm_value = min(1.0, max(0.0, value / 0.05))  # Adjust denominator as needed
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
        
        # Show frame
        cv2.imshow('Parkinson\'s Disease Detection', frame)
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_detection()