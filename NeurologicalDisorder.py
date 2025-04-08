import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import time
import copy
import cv2
import mediapipe as mp

# Set device to MPS for M2 Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Check if MPS is available and working
if device.type == "mps":
    # Create a small test tensor to verify MPS is working
    x = torch.ones(1, device=device)
    print(f"Test tensor on MPS: {x}")
    print("MPS is working correctly!")
else:
    print("MPS is not available. Using CPU instead.")
    print("To enable MPS, ensure you have:")
    print("1. macOS 12.3 or later")
    print("2. PyTorch 1.12 or later with MPS support")

# Rest of the model definition remains the same
class NeurologicalDisorderDetector(nn.Module):
    def __init__(self, num_disorders=5, pretrained=True):
        super(NeurologicalDisorderDetector, self).__init__()
        
        # Use a pretrained model for better feature extraction
        self.feature_extractor = models.resnet50(pretrained=pretrained)
        # Replace the final fully connected layer
        in_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()
        
        # Temporal modeling with bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )
        
        # Attention mechanism for focusing on important temporal features
        self.attention = nn.Sequential(
            nn.Linear(512 * 2, 128),  # 512*2 for bidirectional
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_disorders)
        )
        
        # Additional branch for processing facial landmarks
        self.landmark_processor = nn.Sequential(
            nn.Linear(68 * 2, 256),  # 68 landmarks with x,y coordinates
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # Final fusion layer
        self.fusion = nn.Linear(512 * 2 + 128, 512)
    
    def forward(self, frames_sequence, landmarks=None):
        batch_size, seq_len, channels, height, width = frames_sequence.size()
        
        # Process each frame with CNN
        features = []
        for t in range(seq_len):
            frame_t = frames_sequence[:, t, :, :, :]
            feat = self.feature_extractor(frame_t)
            features.append(feat)
        
        # Stack features to create sequence
        features = torch.stack(features, dim=1)  # [batch_size, seq_len, features]
        
        # Process sequence with LSTM
        lstm_out, _ = self.lstm(features)  # [batch_size, seq_len, 2*hidden_size]
        
        # Apply attention
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Process landmarks if available
        if landmarks is not None:
            landmark_features = self.landmark_processor(landmarks)
            # Combine with visual features
            combined = torch.cat([context_vector, landmark_features], dim=1)
            fusion_output = self.fusion(combined)
        else:
            fusion_output = context_vector
        
        # Classification
        output = self.classifier(fusion_output)
        
        return output

# Function to extract facial landmarks from video
def extract_landmarks_from_video(video_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    landmarks_sequence = []
    frames = []
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        # Convert to RGB and process
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        # Save the frame
        frames.append(Image.fromarray(image_rgb))
        
        # Extract landmarks
        if results.multi_face_landmarks:
            landmarks = []
            for landmark in results.multi_face_landmarks[0].landmark:
                landmarks.extend([landmark.x, landmark.y])
            landmarks_sequence.append(landmarks)
        else:
            # If no face detected, use zeros (or previous landmarks if available)
            if landmarks_sequence:
                landmarks_sequence.append(landmarks_sequence[-1])
            else:
                landmarks_sequence.append([0.0] * (68 * 2))
    
    cap.release()
    return frames, np.array(landmarks_sequence)

# Custom dataset for video sequences
class NeurologicalDisorderDataset(Dataset):
    def __init__(self, root_dir, label_file, transform=None, seq_length=16):
        """
        Args:
            root_dir (string): Directory with all the videos
            label_file (string): Path to the label file (CSV with video_name, disorder_label)
            transform (callable, optional): Optional transform to be applied on frames
            seq_length (int): Number of frames to sample from each video
        """
        self.root_dir = root_dir
        self.transform = transform
        self.seq_length = seq_length
        
        # Read the label file
        self.labels_df = pd.read_csv(label_file)
        
        # Process videos and extract data
        self.samples = []
        for index, row in self.labels_df.iterrows():
            video_path = os.path.join(root_dir, row['video_name'])
            label = row['disorder_label']
            
            # Extract frames and landmarks
            print(f"Processing video: {video_path}")
            try:
                frames, landmarks = extract_landmarks_from_video(video_path)
                
                # Sample seq_length frames evenly from the video
                if len(frames) >= seq_length:
                    indices = np.linspace(0, len(frames) - 1, seq_length, dtype=int)
                    sampled_frames = [frames[i] for i in indices]
                    sampled_landmarks = landmarks[indices]
                    
                    self.samples.append((sampled_frames, sampled_landmarks, label))
                else:
                    print(f"Warning: Video {video_path} has fewer than {seq_length} frames")
            except Exception as e:
                print(f"Error processing video {video_path}: {e}")
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        frames, landmarks, label = self.samples[idx]
        
        # Apply transform to frames
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        # Stack frames to create a sequence tensor
        frames_sequence = torch.stack(frames)
        
        # Convert landmarks to tensor (use mean landmarks across sequence for simplicity)
        # For better results, you could use temporal landmark features as well
        mean_landmarks = torch.FloatTensor(np.mean(landmarks, axis=0))
        
        return frames_sequence, mean_landmarks, label

# MPS-optimized training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()
    
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Set up performance monitor for M2 chip
    try:
        import psutil
        monitor_performance = True
    except ImportError:
        monitor_performance = False
        print("psutil not installed. Install with 'pip install psutil' to monitor CPU/Memory.")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        # Monitoring
        if monitor_performance:
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            print(f"CPU Usage: {cpu_percent}% | Memory Usage: {memory_info.percent}%")
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, landmarks, labels in dataloaders[phase]:
                # Move tensors to the MPS device
                inputs = inputs.to(device)
                landmarks = landmarks.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    outputs = model(inputs, landmarks)
                    loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Ensure we synchronize with MPS device to get accurate stats
                if device.type == "mps":
                    torch.mps.synchronize()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

# Real-time detection function optimized for M2
def detect_disorder_realtime(model, disorder_labels, seq_length=16):
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5
    )
    
    # Initialize frame buffer
    frame_buffer = []
    landmark_buffer = []
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    model.eval()
    
    # For performance monitoring
    frame_times = []
    last_time = time.time()
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        # Calculate FPS
        current_time = time.time()
        frame_times.append(current_time - last_time)
        last_time = current_time
        
        # Only keep the last 30 frame times
        if len(frame_times) > 30:
            frame_times.pop(0)
            
        avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        # Convert to PIL for transforms
        pil_image = Image.fromarray(image_rgb)
        
        # Apply transform
        transformed_image = transform(pil_image).unsqueeze(0)
        
        # Extract landmarks
        if results.multi_face_landmarks:
            landmarks = []
            for landmark in results.multi_face_landmarks[0].landmark:
                landmarks.extend([landmark.x, landmark.y])
            landmarks = np.array(landmarks)
        else:
            # If no face detected, use zeros
            landmarks = np.zeros(68 * 2)
        
        # Add to buffers
        frame_buffer.append(transformed_image)
        landmark_buffer.append(landmarks)
        
        # Keep only the last seq_length frames
        if len(frame_buffer) > seq_length:
            frame_buffer.pop(0)
            landmark_buffer.pop(0)
        
        # If we have enough frames, make a prediction
        if len(frame_buffer) == seq_length:
            with torch.no_grad():
                # Prepare input
                frames_sequence = torch.cat(frame_buffer, dim=0).unsqueeze(0).to(device)
                mean_landmarks = torch.FloatTensor(np.mean(landmark_buffer, axis=0)).unsqueeze(0).to(device)
                
                # Get prediction
                outputs = model(frames_sequence, mean_landmarks)
                
                # Ensure synchronization with MPS device for accurate results
                if device.type == "mps":
                    torch.mps.synchronize()
                    
                _, preds = torch.max(outputs, 1)
                
                # Display result
                predicted_label = disorder_labels[preds.item()]
                cv2.putText(image, f"Prediction: {predicted_label}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display FPS
        cv2.putText(image, f"FPS: {avg_fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Neurological Disorder Detection', image)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main function with MPS optimizations
def main():
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Set your data directory
    data_dir = '/path/to/neurological_disorder_dataset'
    
    # Create datasets
    train_dataset = NeurologicalDisorderDataset(
        root_dir=os.path.join(data_dir, 'train'),
        label_file=os.path.join(data_dir, 'train_labels.csv'),
        transform=data_transforms['train']
    )
    
    val_dataset = NeurologicalDisorderDataset(
        root_dir=os.path.join(data_dir, 'val'),
        label_file=os.path.join(data_dir, 'val_labels.csv'),
        transform=data_transforms['val']
    )
    
    # Create dataloaders - adjust batch size based on M2 memory capacity
    # M2 chips can typically handle larger batch sizes than the default
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    }
    
    # Define your disorder labels (example)
    disorder_labels = ['Normal', 'Parkinson', 'Alzheimer', 'Stroke', 'Epilepsy']
    
    # Initialize model
    model = NeurologicalDisorderDetector(num_disorders=len(disorder_labels))
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Use a learning rate scheduler for better training on M2
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Train model
    model, val_acc_history = train_model(
        model, dataloaders, criterion, optimizer, num_epochs=25
    )
    
    # Save model
    torch.save(model.state_dict(), 'neurological_disorder_detector.pth')
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(val_acc_history)), [acc.cpu().numpy() for acc in val_acc_history])
    plt.title('Validation Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('training_history.png')
    
    # Real-time detection demo
    detect_disorder_realtime(model, disorder_labels)

if __name__ == "__main__":
    main()