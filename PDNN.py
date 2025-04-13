import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import time
import copy
import cv2
import mediapipe as mp



# Print diagnostic information
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Set device (MPS for M1/M2, CUDA for NVIDIA, CPU otherwise)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using MPS device: {device}")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {device}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Performance optimization
torch.backends.cudnn.benchmark = True

# Helper function to free memory
def empty_cache():
    """Empty GPU cache to free memory"""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

# Initialize MediaPipe Face Mesh for facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5)

# Define the PD Detection model
class PDDetectionModel(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=2):
        super(PDDetectionModel, self).__init__()
        
        # Based on the paper, we'll focus on facial action unit variances
        # The LSTM processes sequences of AU variance values
        self.lstm = nn.LSTM(
            input_size=input_size,  # 9 action units as in the paper
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),  # Binary classification (PD or non-PD)
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        last_time_step = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        output = self.fc_layers(last_time_step)
        return output

# Facial Action Unit extraction class
class FacialActionUnitExtractor:
    def __init__(self):
        # Define AU-related facial landmarks (based on MediaPipe's 468 points)
        # These mappings are approximations since MediaPipe doesn't directly provide AUs
        self.au_landmarks = {
            'AU1': [10, 338],  # Inner brow raiser
            'AU2': [65, 295],  # Outer brow raiser
            'AU4': [9, 337],   # Brow lowerer
            'AU6': [117, 346], # Cheek raiser (around eye corners)
            'AU7': [159, 386], # Lid tightener
            'AU9': [129, 358], # Nose wrinkler
            'AU12': [61, 291], # Lip corner puller
            'AU15': [61, 291], # Lip corner depressor
            'AU20': [0, 267]   # Lip stretcher
        }
    
    def extract_aus(self, image):
        """Extract facial action units from an image"""
        # Convert image to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image to find facial landmarks
        results = face_mesh.process(rgb_image)
        
        # Initialize AU values
        aus = np.zeros(9)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Extract approximate AU values based on landmark distances/positions
            # For simplicity, we'll use the distance between key points as a proxy for AU activation
            
            # Convert landmarks to numpy array for easier processing
            points = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
            
            # Extract AU values (simplified approach)
            au_idx = 0
            for au, landmark_ids in self.au_landmarks.items():
                # Calculate distance or movement for the AU
                if len(landmark_ids) >= 2:
                    distance = np.linalg.norm(points[landmark_ids[0]] - points[landmark_ids[1]])
                    aus[au_idx] = distance
                au_idx += 1
            # For smiling detection (AU12), you might want to add:
        if results.multi_face_landmarks:
            # Calculate average mouth corner height vs mouth center
            left_corner = points[61]
            right_corner = points[291]
            mouth_center = (points[13] + points[14]) / 2  # Upper and lower lip center
            
            # Enhanced smile detection (AU12)
            smile_measure = mouth_center[1] - (left_corner[1] + right_corner[1])/2
            aus[6] = smile_measure  # Store in AU12 position

        return aus

# Custom dataset for Parkinson's Disease detection
class PDVideoDataset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None, seq_length=30, target_size=(224, 224)):
        """
        Args:
            root_dir (string): Directory with all the videos
            annotations_file (string): Path to the CSV file with annotations
            transform (callable, optional): Optional transform to be applied on frames
            seq_length (int): Number of frames to extract from each video
            target_size (tuple): Target size for frames
        """
        self.root_dir = root_dir
        self.transform = transform
        self.seq_length = seq_length
        self.target_size = target_size
        self.au_extractor = FacialActionUnitExtractor()
        
        # Read annotations file
        self.annotations = pd.read_csv(annotations_file)
        
        # Check if we have real video files or need to use images instead
        self.use_images = False
        if 'video_file' not in self.annotations.columns:
            print("Video files not specified in annotations. Using image files instead.")
            self.use_images = True
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Get label (PD or non-PD)
        has_pd = self.annotations.iloc[idx].get('has_pd', 0)
        
        # Get filename
        if self.use_images:
            # For RAF-DB conversion: use image files instead of videos
            img_path = os.path.join(self.root_dir, self.annotations.iloc[idx]['image_file'])
            
            # Load image and extract AUs
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Could not find image: {img_path}")
            
            img = cv2.resize(img, self.target_size)
            aus = self.au_extractor.extract_aus(img)
            
            # Generate a sequence by repeating AUs with small random variations
            # This simulates a temporal sequence when only still images are available
            au_sequence = []
            for _ in range(self.seq_length):
                # Add small random variations to simulate movement
                variation = np.random.normal(0, 0.02, size=aus.shape)
                au_sequence.append(aus + variation)
            
            # Calculate variance of AUs over the simulated sequence
            au_variances = np.var(au_sequence, axis=0)
            
        else:
            # Process actual video
            video_path = os.path.join(self.root_dir, self.annotations.iloc[idx]['video_file'])
            
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open video file: {video_path}")
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Determine frame indices to extract
            if frame_count <= self.seq_length:
                # If video has fewer frames than needed, repeat frames
                indices = np.linspace(0, frame_count-1, self.seq_length, dtype=int)
            else:
                # Extract evenly spaced frames
                indices = np.linspace(0, frame_count-1, self.seq_length, dtype=int)
            
            # Extract AUs from selected frames
            au_sequence = []
            for frame_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, self.target_size)
                    aus = self.au_extractor.extract_aus(frame)
                    au_sequence.append(aus)
                else:
                    # If frame reading fails, use previous frame's AUs
                    if au_sequence:
                        au_sequence.append(au_sequence[-1])
                    else:
                        # If no previous frame, use zeros
                        au_sequence.append(np.zeros(9))
            
            cap.release()
            
            # Calculate variance of AUs over the sequence
            au_variances = np.var(au_sequence, axis=0)
        
        # Convert to tensors
        au_sequence_tensor = torch.FloatTensor(au_sequence)
        au_variances_tensor = torch.FloatTensor(au_variances)
        label_tensor = torch.FloatTensor([has_pd])
        
        return au_sequence_tensor, au_variances_tensor, label_tensor

# Training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()
    
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            all_preds = []
            all_labels = []
            
            # Iterate over data
            for au_sequences, au_variances, labels in dataloaders[phase]:
                au_sequences = au_sequences.to(device)
                au_variances = au_variances.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(au_sequences)
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item() * au_sequences.size(0)
                
                # Store predictions and labels for AUC calculation
                all_preds.extend(outputs.detach().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
            # Calculate AUC
            all_preds = np.array(all_preds).flatten()
            all_labels = np.array(all_labels).flatten()
            epoch_auc = roc_auc_score(all_labels, all_preds)
            
            # Calculate accuracy at threshold 0.5
            binary_preds = (all_preds >= 0.5).astype(int)
            epoch_acc = np.mean(binary_preds == all_labels)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}')
            
            # Save loss history
            if phase == 'train':
                train_loss_history.append(epoch_loss)
            else:
                val_loss_history.append(epoch_loss)
            
            # deep copy the model if best validation AUC
            if phase == 'val' and epoch_auc > best_auc:
                best_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val AUC: {best_auc:.4f}')
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    # Return training history
    history = {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'val_acc': val_acc_history,
        'best_auc': best_auc
    }
    
    return model, history

# Function to convert RAF-DB to PD dataset format
def convert_rafdb_to_pd_format(rafdb_root, emotion_map_file, output_annotations):
    """
    Convert RAF-DB emotion dataset to a format suitable for PD detection
    
    Args:
        rafdb_root: Root directory of RAF-DB dataset
        emotion_map_file: Optional file mapping emotions to PD probability
        output_annotations: Path to save the output annotations CSV
    """
    # Find all image files
    image_files = []
    for root, _, files in os.walk(os.path.join(rafdb_root, 'DATASET')):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                rel_path = os.path.relpath(os.path.join(root, file), rafdb_root)
                image_files.append(rel_path)
    
    print(f"Found {len(image_files)} image files")
    
    # Create annotations dataframe
    data = {
        'image_file': image_files,
        'has_pd': [0] * len(image_files)  # Default to non-PD
    }
    
    # If we have an emotion map file, use it to assign PD probabilities
    if emotion_map_file and os.path.exists(emotion_map_file):
        emotion_map = pd.read_csv(emotion_map_file)
        
        # Extract emotion from folder structure (RAF-DB organizes by emotion category)
        emotions = []
        for image_file in image_files:
            parts = image_file.split('/')
            # In RAF-DB, the emotion category is typically a number in the path
            for part in parts:
                if part.isdigit():
                    emotions.append(int(part))
                    break
            else:
                emotions.append(0)  # Default if no category found
        
        data['emotion'] = emotions
        
        # Map emotions to PD probability and sample based on probability
        pd_probs = []
        for emotion in emotions:
            if emotion in emotion_map['emotion'].values:
                prob = emotion_map.loc[emotion_map['emotion'] == emotion, 'pd_probability'].iloc[0]
                pd_probs.append(prob)
            else:
                pd_probs.append(0.1)  # Default probability
        
        # Generate binary PD labels based on probabilities
        has_pd = np.random.binomial(1, pd_probs)
        data['has_pd'] = has_pd
    
    # Save annotations
    annotations_df = pd.DataFrame(data)
    annotations_df.to_csv(output_annotations, index=False)
    print(f"Saved annotations to {output_annotations}")
    
    return annotations_df

# Main function
def main():
    # Data directory
    data_dir = '/Users/riyan/Desktop/archive'  # Base directory for the dataset
    
    # Check if we're using RAF-DB or a dedicated PD dataset
    use_rafdb = True  # Set to False if you have a dedicated PD video dataset
    
    if use_rafdb:
        # Convert RAF-DB to PD format
        print("Converting RAF-DB dataset to PD detection format...")
        
        # Create a simple emotion mapping (based on the paper findings)
        # The paper suggests that PD patients have less variance in facial expressions,
        # particularly in smiling (which corresponds to "Happy" in RAF-DB)
        # Updated probabilities based on hypomimia evidence
        emotion_map = {
            'emotion': [1, 2, 3, 4, 5, 6, 7],
            'emotion_name': ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral'],
            'pd_probability': [0.15, 0.12, 0.18, 0.35, 0.15, 0.18, 0.40]
        }
        
        # Save emotion map
        emotion_map_df = pd.DataFrame(emotion_map)
        emotion_map_file = os.path.join(data_dir, 'emotion_pd_map.csv')
        emotion_map_df.to_csv(emotion_map_file, index=False)
        
        # Convert dataset
        annotations_file = os.path.join(data_dir, 'pd_annotations.csv')
        annotations = convert_rafdb_to_pd_format(data_dir, emotion_map_file, annotations_file)
        
        # Split into train/val sets (80/20 split)
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(annotations, test_size=0.2, stratify=annotations['has_pd'])
        
        # Save train/val splits
        train_df.to_csv(os.path.join(data_dir, 'pd_train.csv'), index=False)
        val_df.to_csv(os.path.join(data_dir, 'pd_val.csv'), index=False)
    
    # Data transformations (for image processing if needed)
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Create datasets
    train_dataset = PDVideoDataset(
        root_dir=data_dir,
        annotations_file=os.path.join(data_dir, 'pd_train.csv'),
        transform=data_transforms['train']
    )
    
    val_dataset = PDVideoDataset(
        root_dir=data_dir,
        annotations_file=os.path.join(data_dir, 'pd_val.csv'),
        transform=data_transforms['val']
    )
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    }
    
    # Initialize model
    model = PDDetectionModel(input_size=9)  # 9 AUs as input
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    model, history = train_model(
        model, dataloaders, criterion, optimizer, num_epochs=25
    )
    
    # Save model
    torch.save(model.state_dict(), 'pd_detection_model.pth')
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(len(history['train_loss'])), history['train_loss'], label='Train Loss')
    plt.plot(range(len(history['val_loss'])), history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(len(history['val_acc'])), history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    print("Training complete. Model saved to pd_detection_model.pth")

if __name__ == "__main__":
    main()