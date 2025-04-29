import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time
import os
import matplotlib.pyplot as plt
from collections import Counter, deque

# Define the Neurological Disorder Classification Model
class NeurologicalCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(NeurologicalCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # We'll determine the size dynamically during forward pass
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # This will be adjusted in forward
        self.fc2 = nn.Linear(512, num_classes)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten layer
        x = x.view(x.size(0), -1)
        
        # If it's the first pass, adjust the fc1 layer
        if hasattr(self, 'fc1') and self.fc1.in_features != x.shape[1]:
            self.fc1 = nn.Linear(x.shape[1], 512).to(x.device)
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Neurological Disorder Detection System
class NeurologicalDisorderSystem:
    def __init__(self, model_path):
        # Set device for model
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                "mps" if torch.backends.mps.is_available() else 
                                "cpu")
        print(f"Using device: {self.device}")
        
        # Load the trained model
        try:
            self.model = NeurologicalCNN(num_classes=3).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded successfully from: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained model - this is just for demonstration")
            self.model = NeurologicalCNN(num_classes=3).to(self.device)
            self.model.eval()
        
        # Define class names
        self.class_names = {
            0: 'Alzheimers',
            1: 'Parkinsons',
            2: 'Stroke'
        }
        
        # Define transforms for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Store recent predictions for stability
        self.prob_history = deque(maxlen=20)
        
        # Initialize ROI selection
        self.roi_selected = False
        self.roi = None
        
        # Track probabilities over time for visualization
        self.time_series = {class_name: [] for class_name in self.class_names.values()}
        self.max_history_points = 100
        
        # Create plot for visualization
        self.create_plot()
    
    def create_plot(self):
        """Initialize plot for tracking probabilities over time"""
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.fig.tight_layout()
        self.lines = {}
        
        colors = ['blue', 'red', 'orange']
        for i, class_name in enumerate(self.class_names.values()):
            self.lines[class_name], = self.ax.plot([], [], label=class_name, color=colors[i])
        
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(0, self.max_history_points)
        self.ax.set_title('Disorder Probability Over Time')
        self.ax.set_xlabel('Frames')
        self.ax.set_ylabel('Probability')
        self.ax.legend(loc='upper right')
        self.ax.grid(True)
        
        # Convert plot to image
        self.fig.canvas.draw()
        self.plot_img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        self.plot_img = self.plot_img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
    
    def update_plot(self, probabilities):
        """Update the probability plot with new data"""
        # Add new probabilities to time series
        for class_id, prob in enumerate(probabilities):
            class_name = self.class_names[class_id]
            self.time_series[class_name].append(prob)
            
            # Trim to max length
            if len(self.time_series[class_name]) > self.max_history_points:
                self.time_series[class_name].pop(0)
            
            # Update line data
            x_data = list(range(len(self.time_series[class_name])))
            self.lines[class_name].set_data(x_data, self.time_series[class_name])
        
        # Update plot
        if self.time_series[self.class_names[0]]:  # If we have any data
            max_len = len(self.time_series[self.class_names[0]])
            self.ax.set_xlim(0, max(self.max_history_points, max_len))
        
        self.fig.canvas.draw()
        
        # Convert to image
        self.plot_img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        self.plot_img = self.plot_img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        
        # Convert to BGR for OpenCV
        self.plot_img = cv2.cvtColor(self.plot_img, cv2.COLOR_RGB2BGR)
        
        return self.plot_img
    
    def process_frame(self, frame):
        """Process a single frame for disorder classification"""
        # Initialize result dictionary
        result = {
            'detected': False,
            'class_name': None,
            'class_id': None,
            'probabilities': None,
            'relative_confidence': 0.0
        }
        
        # Convert frame to RGB (for PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use the entire frame if ROI not selected
        if not self.roi_selected:
            image = Image.fromarray(rgb_frame)
        else:
            # Extract the ROI
            x, y, w, h = self.roi
            roi_rgb = rgb_frame[y:y+h, x:x+w]
            image = Image.fromarray(roi_rgb)
        
        # Apply transformations
        try:
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                predicted_class = torch.argmax(outputs, dim=1).item()
            
            # Add to probability history for smoothing
            self.prob_history.append(probabilities)
            
            # Calculate smoothed probabilities (average over recent frames)
            if self.prob_history:
                avg_probs = np.mean(np.array(self.prob_history), axis=0)
                smoothed_class = np.argmax(avg_probs)
                
                # Calculate relative confidence (difference between highest and second highest)
                sorted_probs = np.sort(avg_probs)[::-1]
                relative_confidence = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
                
                # Fill result dictionary
                result['detected'] = True
                result['class_name'] = self.class_names[smoothed_class]
                result['class_id'] = smoothed_class
                result['probabilities'] = avg_probs
                result['relative_confidence'] = relative_confidence
            
            # Update the plot
            self.plot_img = self.update_plot(probabilities)
            
            return result
        
        except Exception as e:
            print(f"Error processing frame: {e}")
            return result
    
    def select_roi(self, frame):
        """Allow user to select region of interest"""
        print("Select region of interest and press ENTER when done (ESC to cancel)")
        roi = cv2.selectROI("Select Region of Interest", frame, False)
        cv2.destroyWindow("Select Region of Interest")
        
        if roi[2] > 0 and roi[3] > 0:
            self.roi = roi
            self.roi_selected = True
            print(f"ROI selected: {roi}")
        else:
            self.roi_selected = False
            print("No ROI selected, using full frame")

# Function to run live detection
def run_neurological_classification(model_path="neurological_cnn_model.pth"):
    """Run real-time neurological disorder classification using webcam"""
    # Create detection system
    detector = NeurologicalDisorderSystem(model_path)
    
    # Open webcam
    cap = cv2.VideoCapture(0)  # Try camera 0 first
    if not cap.isOpened():
        print("Could not open camera 0, trying camera 1...")
        cap = cv2.VideoCapture(1)  # Try camera 1 as fallback
        if not cap.isOpened():
            print("Could not open any webcam")
            return
    
    print("Neurological Disorder Classification started.")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 'r' to select ROI")
    print("- Press 's' to save a screenshot")
    
    # For FPS calculation
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Create output directory for screenshots
    os.makedirs("disorder_screenshots", exist_ok=True)
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Process frame for classification
        result = detector.process_frame(frame)
        
        # Draw ROI if selected
        if detector.roi_selected:
            x, y, w, h = detector.roi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Create a composite frame with the plot
        if hasattr(detector, 'plot_img') and detector.plot_img is not None:
            plot_img = detector.plot_img
            
            # Resize plot to fit on screen
            h, w = frame.shape[:2]
            plot_h, plot_w = plot_img.shape[:2]
            target_h = h // 3
            target_w = int(plot_w * (target_h / plot_h))
            
            plot_img = cv2.resize(plot_img, (target_w, target_h))
            
            # Create composite image
            composite_h = h
            composite_w = max(w, target_w)
            composite = np.zeros((composite_h, composite_w, 3), dtype=np.uint8)
            
            # Add frame
            composite[:h, :w] = frame
            
            # Add plot at the bottom
            plot_y = h - target_h
            composite[plot_y:plot_y+target_h, :target_w] = plot_img
            
            # Use composite as our display frame
            frame = composite
        
        # Display results on frame
        if result['detected']:
            # Get class name and probabilities
            class_name = result['class_name']
            probabilities = result['probabilities']
            relative_confidence = result['relative_confidence']
            
            # Different colors for different conditions
            if class_name == 'Alzheimers':
                color = (255, 0, 0)  # Blue
            elif class_name == 'Parkinsons':
                color = (0, 0, 255)  # Red
            else:  # Stroke
                color = (0, 255, 255)  # Yellow
            
            # Display main classification with confidence
            confidence_level = "High" if relative_confidence > 0.5 else "Medium" if relative_confidence > 0.2 else "Low"
            cv2.putText(frame, f"Classification: {class_name} ({confidence_level} confidence)", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add disclaimer
            cv2.putText(frame, "RESEARCH ONLY - NOT FOR DIAGNOSIS", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display probabilities for each class
            y_pos = 90
            for i, class_id in enumerate(detector.class_names):
                class_prob = probabilities[class_id]
                class_text = f"{detector.class_names[class_id]}: {class_prob:.3f}"
                
                # Use same color for the class that's detected
                if class_id == result['class_id']:
                    text_color = color
                else:
                    text_color = (255, 255, 255)
                
                cv2.putText(frame, class_text, (10, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                
                # Draw probability bar
                bar_length = int(150 * class_prob)
                bar_color = color if class_id == result['class_id'] else (255, 255, 255)
                cv2.rectangle(frame, (200, y_pos-15), (200+bar_length, y_pos-5), bar_color, -1)
                
                y_pos += 30
                
            # Display some key signs based on detected condition
            if class_name == 'Alzheimers':
                signs = ["- May show memory impairment indicators",
                         "- May exhibit confusion or disorientation",
                         "- Potential language difficulties"]
            elif class_name == 'Parkinsons':
                signs = ["- May show reduced facial expressions",
                         "- Potential for facial rigidity or masked face",
                         "- Possible asymmetrical features"]
            else:  # Stroke
                signs = ["- May show facial asymmetry",
                         "- Potential drooping on one side",
                         "- Possible slurred speaking patterns"]
                
            y_pos += 10
            cv2.putText(frame, "Potential Research Indicators:", (10, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y_pos += 25
            
            for sign in signs:
                cv2.putText(frame, sign, (20, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_pos += 25
        else:
            # If no valid detection
            cv2.putText(frame, "Processing... (Make sure face is visible)", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
            
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-120, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display instructions
        instructions = [
            "q: Quit",
            "r: Select ROI",
            "s: Save screenshot"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, 
                      (frame.shape[1]-120, 60 + i*20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Neurological Disorder Classification', frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.select_roi(frame)
        elif key == ord('s'):
            # Save screenshot
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"disorder_screenshots/screenshot_{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved to {filename}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    plt.close(detector.fig)

# Function to create a simple GUI for the application
def create_gui():
    """Create a simple GUI for the application"""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        
        def select_model():
            model_path = filedialog.askopenfilename(
                title="Select Model File",
                filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
            )
            if model_path:
                model_entry.delete(0, tk.END)
                model_entry.insert(0, model_path)
        
        def start_detection():
            model_path = model_entry.get()
            if not model_path or not os.path.exists(model_path):
                if messagebox.askyesno("Model Not Found", 
                                      "Model file not found. Run with demo mode?"):
                    model_path = ""
                else:
                    return
            
            root.destroy()
            run_neurological_classification(model_path)
        
        # Create main window
        root = tk.Tk()
        root.title("Neurological Disorder Classification")
        root.geometry("500x300")
        
        # Create frame
        frame = tk.Frame(root, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(frame, text="Neurological Disorder Classification", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Model selection
        model_frame = tk.Frame(frame)
        model_frame.pack(fill=tk.X, pady=(0, 20))
        
        model_label = tk.Label(model_frame, text="Model Path:")
        model_label.pack(side=tk.LEFT, padx=(0, 10))
        
        model_entry = tk.Entry(model_frame, width=30)
        model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        model_entry.insert(0, "neurological_cnn_model.pth")
        
        browse_button = tk.Button(model_frame, text="Browse", command=select_model)
        browse_button.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Disclaimer
        disclaimer_text = (
            "IMPORTANT: This application is for RESEARCH PURPOSES ONLY.\n"
            "It is not intended for medical diagnosis or treatment decisions.\n"
            "Consult with healthcare professionals for any medical concerns."
        )
        disclaimer = tk.Label(frame, text=disclaimer_text, fg="red", 
                             wraplength=400, justify=tk.CENTER)
        disclaimer.pack(pady=(0, 20))
        
        # Start button
        start_button = tk.Button(frame, text="Start Detection", 
                               command=start_detection, 
                               bg="#4CAF50", fg="white", 
                               font=("Arial", 12, "bold"),
                               padx=20, pady=10)
        start_button.pack()
        
        # Run the GUI
        root.mainloop()
        
    except ImportError:
        print("tkinter not available. Starting detection directly.")
        run_neurological_classification()

# Main execution
if __name__ == "__main__":
    try:
        create_gui()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        print("Starting detection directly...")
        run_neurological_classification()