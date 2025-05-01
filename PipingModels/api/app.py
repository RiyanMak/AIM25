from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_ngrok import run_with_ngrok
import cv2
import numpy as np
import base64
import tempfile
import os
import sys
import json

# Add the parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import your original modules
from face_neuro_detector import NeurologicalDisorderDetectionSystem
from pd_detection_system import PDDetectionSystem

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up path to the model
model_path = os.path.join(parent_dir, "PipingModels/best_emotion_model.pth")
if not os.path.exists(model_path):
    print(f"Warning: Model file not found at {model_path}")
    alt_model_path = os.path.join(parent_dir, "best_emotion_model.pth")
    if os.path.exists(alt_model_path):
        print(f"Found model at {alt_model_path}")
        model_path = alt_model_path
    else:
        print("Warning: Model file not found. Will run without emotion recognition.")
        model_path = None

# Initialize detection systems
print("Initializing PD detection system...")
pd_system = PDDetectionSystem(model_path)

print("Initializing neurological disorder detection system...")
neuro_system = NeurologicalDisorderDetectionSystem(model_path)
print("Detection systems initialized successfully.")

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route('/detect', methods=['POST'])
def detect_disorders():
    """Process image and detect neurological disorders"""
    if 'image' not in request.json:
        return jsonify({"error": "No image data provided"}), 400
    
    try:
        base64_image = request.json['image']
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]

        image_data = base64.b64decode(base64_image)

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file.write(image_data)
            temp_file_path = temp_file.name

        frame = cv2.imread(temp_file_path)
        os.unlink(temp_file_path)

        if frame is None:
            return jsonify({"error": "Failed to process image"}), 400

        pd_result = pd_system.process_frame(frame)
        neuro_result = neuro_system.process_frame(frame)

        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj

        pd_result = convert_numpy_types(pd_result)
        neuro_result = convert_numpy_types(neuro_result)

        result = {
            "face_detected": neuro_result['face_detected'],
            "pd": {
                "probability": pd_result['pd_probability'],
                "likelihood": pd_result['pd_likelihood'],
                "features": pd_result.get('features')
            },
            "ms": {
                "probability": neuro_result['ms']['probability'],
                "likelihood": neuro_result['ms']['likelihood'],
                "features": neuro_result['ms'].get('features')
            },
            "ad": {
                "probability": neuro_result['ad']['probability'],
                "likelihood": neuro_result['ad']['likelihood'],
                "features": neuro_result['ad'].get('features')
            }
        }

        print("Received detection request")
        print(request.json.keys())

        return jsonify(result)

    except Exception as e:
        print(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    from flask_ngrok import run_with_ngrok
    run_with_ngrok(app)  # This starts Flask with ngrok
