import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import time
import copy
import cv2
import mediapipe as mp
import pickle

# ===== SETUP =====

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

# ===== FACIAL LANDMARK EXTRACTION =====

# Initialize MediaPipe Face Mesh for facial landmarks
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

# ===== DATASET PREPARATION =====

class PDDataset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None, seq_length=30, target_size=(224, 224),
                inspect_features=False, balance_classes=True):
        """
        Dataset for Parkinson's Disease detection
        
        Args:
            root_dir: Directory with all the images/videos
            annotations_file: Path to CSV file with annotations
            transform: Optional transform to be applied
            seq_length: Number of frames to extract
            target_size: Target size for frames
            inspect_features: Whether to save feature statistics for inspection
            balance_classes: Whether to balance classes in the dataset
        """
        self.root_dir = root_dir
        self.transform = transform
        self.seq_length = seq_length
        self.target_size = target_size
        self.au_extractor = FacialActionUnitExtractor()
        self.inspect_features = inspect_features
        
        # Read annotations
        self.annotations = pd.read_csv(annotations_file)
        
        # Balance classes if requested
        if balance_classes:
            pd_samples = self.annotations[self.annotations['has_pd'] == 1]
            non_pd_samples = self.annotations[self.annotations['has_pd'] == 0]
            
            # Downsample majority class or upsample minority class
            if len(pd_samples) < len(non_pd_samples):
                # Downsample non-PD
                non_pd_samples = non_pd_samples.sample(n=len(pd_samples), random_state=42)
                self.annotations = pd.concat([pd_samples, non_pd_samples])
            elif len(pd_samples) > len(non_pd_samples):
                # Upsample PD (with replacement)
                pd_samples = pd_samples.sample(n=len(non_pd_samples), random_state=42, replace=True)
                self.annotations = pd.concat([pd_samples, non_pd_samples])
        
        # Shuffle
        self.annotations = self.annotations.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Report class distribution
        pd_count = sum(self.annotations['has_pd'])
        total = len(self.annotations)
        print(f"Dataset loaded: {total} samples")
        print(f"Class distribution: PD={pd_count} ({pd_count/total*100:.2f}%), Non-PD={total-pd_count} ({(total-pd_count)/total*100:.2f}%)")
        
        # For feature inspection
        if inspect_features:
            self.pd_features = []
            self.non_pd_features = []
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Get label (PD or non-PD)
        has_pd = self.annotations.iloc[idx].get('has_pd', 0)
        
        # Get filename - handle both image and video datasets
        if 'video_file' in self.annotations.columns:
            file_path = os.path.join(self.root_dir, self.annotations.iloc[idx]['video_file'])
            is_video = True
        else:
            file_path = os.path.join(self.root_dir, self.annotations.iloc[idx]['image_file'])
            is_video = False
        
        # Extract features
        if is_video:
            # Process video file
            sequence, variances = self._process_video(file_path)
        else:
            # Process image file (with simulated sequence)
            sequence, variances = self._process_image(file_path)
        
        # Store features for inspection if needed
        if self.inspect_features:
            if has_pd == 1:
                self.pd_features.append(variances.numpy())
            else:
                self.non_pd_features.append(variances.numpy())
        
        # Return tensors
        label_tensor = torch.FloatTensor([has_pd])
        
        return sequence, variances, label_tensor
    
    def _process_video(self, video_path):
        """Process a video file to extract AU sequence and variances"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")
        
        # Get frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Select frames
        if frame_count <= self.seq_length:
            indices = np.linspace(0, frame_count-1, self.seq_length, dtype=int)
        else:
            indices = np.linspace(0, frame_count-1, self.seq_length, dtype=int)
        
        # Extract AUs
        au_sequence = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, self.target_size)
                aus = self.au_extractor.extract_aus(frame)
                au_sequence.append(aus)
            else:
                if au_sequence:
                    au_sequence.append(au_sequence[-1])
                else:
                    au_sequence.append(np.zeros(13))
        
        cap.release()
        
        # Calculate variances (key feature according to the paper)
        au_variances = np.var(au_sequence, axis=0)
        
        # Convert to tensors
        sequence_np = np.array(au_sequence)
        au_sequence_tensor = torch.FloatTensor(sequence_np)
        au_variances_tensor = torch.FloatTensor(au_variances)
        
        return au_sequence_tensor, au_variances_tensor
    
    def _process_image(self, image_path):
        """Process an image file with simulated sequence for temporal features"""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not find image: {image_path}")
        
        img = cv2.resize(img, self.target_size)
        aus = self.au_extractor.extract_aus(img)
        
        # Generate a sequence with subtle variations to simulate a video
        au_sequence = []
        for i in range(self.seq_length):
            # Create variations that decrease as frames progress (simulating expression change)
            decay_factor = (self.seq_length - i) / self.seq_length
            variation = np.random.normal(0, 0.02 * decay_factor, size=aus.shape)
            au_sequence.append(aus + variation)
        
        # Calculate variances
        au_variances = np.var(au_sequence, axis=0)
        
        # Convert to tensors
        sequence_np = np.array(au_sequence)
        au_sequence_tensor = torch.FloatTensor(sequence_np)
        au_variances_tensor = torch.FloatTensor(au_variances)
        
        return au_sequence_tensor, au_variances_tensor
    
    def get_feature_statistics(self):
        """Get statistics of features for PD and non-PD samples"""
        if not self.inspect_features:
            raise ValueError("Feature inspection was not enabled for this dataset")
        
        pd_features = np.array(self.pd_features)
        non_pd_features = np.array(self.non_pd_features)
        
        pd_means = np.mean(pd_features, axis=0)
        non_pd_means = np.mean(non_pd_features, axis=0)
        
        pd_std = np.std(pd_features, axis=0)
        non_pd_std = np.std(non_pd_features, axis=0)
        
        feature_names = [
            "AU6 (Cheek Raiser)", 
            "AU12 (Lip Corner Puller)",
            "AU4 (Brow Lowerer)",
            "AU1 (Inner Brow Raiser)",
            "AU2 (Outer Brow Raiser)",
            "AU7 (Lid Tightener)",
            "AU9 (Nose Wrinkler)",
            "AU15 (Lip Corner Depressor)",
            "AU20 (Lip Stretcher)",
            "Smile Asymmetry",
            "Eye Openness",
            "Facial Mobility",
            "Mouth Corner Rest"
        ]
        
        stats = {
            "feature_names": feature_names,
            "pd_means": pd_means,
            "non_pd_means": non_pd_means,
            "pd_std": pd_std,
            "non_pd_std": non_pd_std,
            "differences": pd_means - non_pd_means,
            "t_statistics": (pd_means - non_pd_means) / np.sqrt(pd_std**2/len(pd_features) + non_pd_std**2/len(non_pd_features))
        }
        
        return stats

# ===== FEATURE VALIDATION FUNCTIONS =====

def validate_features(train_dataset):
    """Validate whether features can distinguish between PD and non-PD"""
    print("Validating features...")
    
    # Extract feature statistics
    stats = train_dataset.get_feature_statistics()
    
    # Print differences
    print("\nFeature Differences (PD vs non-PD):")
    print("-" * 50)
    for i, name in enumerate(stats["feature_names"]):
        diff = stats["differences"][i]
        t_stat = stats["t_statistics"][i]
        pd_val = stats["pd_means"][i]
        non_pd_val = stats["non_pd_means"][i]
        
        significance = ""
        if abs(t_stat) > 2.58:
            significance = "*** (p<0.01)"
        elif abs(t_stat) > 1.96:
            significance = "** (p<0.05)"
        elif abs(t_stat) > 1.65:
            significance = "* (p<0.1)"
        
        print(f"{name}: PD={pd_val:.5f}, Non-PD={non_pd_val:.5f}, Diff={diff:.5f}, t={t_stat:.2f} {significance}")
    
    # Plot key differences
    plt.figure(figsize=(12, 6))
    
    # 1. Bar chart of means
    plt.subplot(1, 2, 1)
    x = np.arange(len(stats["feature_names"]))
    width = 0.35
    plt.bar(x - width/2, stats["pd_means"], width, label='PD')
    plt.bar(x + width/2, stats["non_pd_means"], width, label='Non-PD')
    plt.xlabel('Features')
    plt.ylabel('Mean Value')
    plt.title('Feature Means Comparison')
    plt.xticks(x, [f"F{i+1}" for i in range(len(stats["feature_names"]))], rotation=45)
    plt.legend()
    
    # 2. T-statistics
    plt.subplot(1, 2, 2)
    t_stats = stats["t_statistics"]
    colors = ['red' if abs(t) > 1.96 else 'gray' for t in t_stats]
    plt.bar(x, t_stats, color=colors)
    plt.axhline(y=1.96, color='black', linestyle='--', alpha=0.7, label='p=0.05')
    plt.axhline(y=-1.96, color='black', linestyle='--', alpha=0.7)
    plt.xlabel('Features')
    plt.ylabel('t-statistic')
    plt.title('Feature Significance (t-statistics)')
    plt.xticks(x, [f"F{i+1}" for i in range(len(stats["feature_names"]))], rotation=45)
    
    plt.tight_layout()
    plt.savefig('feature_validation.png')
    print("Feature validation plot saved to 'feature_validation.png'")
    
    return stats

def test_with_classical_ml(train_dataset, val_dataset):
    """Test if a simple ML model can classify the data"""
    print("\nTesting with classical ML models...")
    
    # Extract features and labels
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    
    for _, variance, label in train_dataset:
        X_train.append(variance.numpy())
        y_train.append(int(label.item()))
    
    for _, variance, label in val_dataset:
        X_val.append(variance.numpy())
        y_val.append(int(label.item()))
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    print(f"Training data shape: {X_train.shape}")
    
    # Train a Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    train_acc = rf.score(X_train, y_train)
    val_acc = rf.score(X_val, y_val)
    
    # Get predictions
    train_preds = rf.predict(X_train)
    val_preds = rf.predict(X_val)
    
    # Calculate additional metrics
    train_auc = roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1])
    val_auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
    
    print(f"Random Forest results:")
    print(f"  Train accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}")
    print(f"  Validation accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")
    
    # Feature importance
    feature_names = [
        "AU6 (Cheek Raiser)", 
        "AU12 (Lip Corner Puller)",
        "AU4 (Brow Lowerer)",
        "AU1 (Inner Brow Raiser)",
        "AU2 (Outer Brow Raiser)",
        "AU7 (Lid Tightener)",
        "AU9 (Nose Wrinkler)",
        "AU15 (Lip Corner Depressor)",
        "AU20 (Lip Stretcher)",
        "Smile Asymmetry",
        "Eye Openness",
        "Facial Mobility",
        "Mouth Corner Rest"
    ]
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature ranking (Random Forest):")
    for f in range(X_train.shape[1]):
        print(f"{f+1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")
    
    # Print classification report
    print("\nValidation Set Classification Report:")
    print(classification_report(y_val, val_preds))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_val, val_preds)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Non-PD', 'PD'])
    plt.yticks(tick_marks, ['Non-PD', 'PD'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('rf_confusion_matrix.png')
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png')
    
    # Save the model
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    
    print("Random Forest model saved to 'rf_model.pkl'")
    
    return {
        'model': rf,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_auc': train_auc,
        'val_auc': val_auc,
        'importances': importances,
        'indices': indices
    }

# ===== MODEL DEFINITIONS =====

class SimplePDModel(nn.Module):
    """Simple model operating on AU variances directly"""
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

class TemporalPDModel(nn.Module):
    """More complex model that uses the full AU sequence"""
    def __init__(self, input_size=13, hidden_size=64, num_layers=2):
        super(TemporalPDModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: [batch_size, seq_len, hidden_size]
        
        # Attention weights
        attn_weights = self.attention(lstm_out)
        # attn_weights shape: [batch_size, seq_len, 1]
        
        # Apply attention
        context = torch.sum(attn_weights * lstm_out, dim=1)
        # context shape: [batch_size, hidden_size]
        
        # Final prediction
        output = self.fc(context)
        # output shape: [batch_size, 1]
        
        return output

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

# ===== TRAINING FUNCTIONS =====

def train_simple_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, scheduler=None):
    """Train a simple model that uses AU variances directly"""
    print("\nTraining simple model...")
    since = time.time()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_auc': [],
        'val_auc': []
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader
            
            running_loss = 0.0
            all_labels = []
            all_preds = []
            
            # Iterate over batches
            for _, variances, labels in data_loader:
                variances = variances.to(device)
                labels = labels.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(variances)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize (training only)
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * variances.size(0)
                
                # Store predictions and labels
                all_preds.extend(outputs.detach().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            epoch_loss = running_loss / len(data_loader.dataset)
            all_preds = np.array(all_preds).flatten()
            all_labels = np.array(all_labels).flatten()
            
            # Calculate AUC
            try:
                epoch_auc = roc_auc_score(all_labels, all_preds)
            except:
                epoch_auc = 0.5  # Default if only one class
            
            # Calculate accuracy
            epoch_acc = np.mean(((all_preds > 0.5) == all_labels).astype(float))
            
            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
                history['train_auc'].append(epoch_auc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                history['val_auc'].append(epoch_auc)
                
                # Save best model
                if epoch_auc > best_auc:
                    best_auc = epoch_auc
                    best_model_wts = copy.deepcopy(model.state_dict())
            
            # Print metrics
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}')
            
            # If validation phase, print confusion matrix
            if phase == 'val':
                # Convert to binary predictions
                binary_preds = (all_preds > 0.5).astype(int)
                
                # Confusion matrix
                cm = confusion_matrix(all_labels, binary_preds)
                print(f"Confusion Matrix: \n{cm}")
                
                # Sensitivity and specificity
                if cm.shape == (2, 2):
                    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
                    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
                    print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
        
        # Step scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(history['val_loss'][-1])
            else:
                scheduler.step()
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val AUC: {best_auc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def train_temporal_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, scheduler=None):
    """Train a temporal model that uses the full AU sequence"""
    print("\nTraining temporal model...")
    since = time.time()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_auc': [],
        'val_auc': []
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader
            
            running_loss = 0.0
            all_labels = []
            all_preds = []
            
            # Iterate over batches
            for sequences, _, labels in data_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize (training only)
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * sequences.size(0)
                
                # Store predictions and labels
                all_preds.extend(outputs.detach().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            epoch_loss = running_loss / len(data_loader.dataset)
            all_preds = np.array(all_preds).flatten()
            all_labels = np.array(all_labels).flatten()
            
            # Calculate AUC
            try:
                epoch_auc = roc_auc_score(all_labels, all_preds)
            except:
                epoch_auc = 0.5  # Default if only one class
            
            # Calculate accuracy
            epoch_acc = np.mean(((all_preds > 0.5) == all_labels).astype(float))
            
            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
                history['train_auc'].append(epoch_auc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                history['val_auc'].append(epoch_auc)
                
                # Save best model
                if epoch_auc > best_auc:
                    best_auc = epoch_auc
                    best_model_wts = copy.deepcopy(model.state_dict())
            
            # Print metrics
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}')
            
            # If validation phase, print confusion matrix
            if phase == 'val':
                # Convert to binary predictions
                binary_preds = (all_preds > 0.5).astype(int)
                
                # Confusion matrix
                cm = confusion_matrix(all_labels, binary_preds)
                print(f"Confusion Matrix: \n{cm}")
                
                # Sensitivity and specificity
                if cm.shape == (2, 2):
                    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
                    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
                    print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
        
        # Step scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(history['val_loss'][-1])
            else:
                scheduler.step()
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val AUC: {best_auc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def plot_training_history(history, model_name):
    """Plot training history"""
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot AUC
    plt.subplot(1, 3, 3)
    plt.plot(history['train_auc'], label='Train AUC')
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.title('AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_history.png')
    print(f"Training history saved to '{model_name}_history.png'")

# ===== MAIN FUNCTION =====

def main():
    # Data directory
    data_dir = '/path/to/your/dataset'  # CHANGE THIS to your dataset path
    
    # Create synthetic data for testing if needed
    create_synthetic = False
    if create_synthetic:
        create_synthetic_dataset(data_dir)
    
    # Step 1: Create dataset from RAF-DB
    # Convert RAF-DB to PD format with balanced classes
    print("Converting RAF-DB dataset to PD detection format...")
    
    # Define the mapping from emotions to PD probabilities based on the paper's findings
    emotion_map = {
        'emotion': [1, 2, 3, 4, 5, 6, 7],
        'emotion_name': ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral'],
        'pd_probability': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # 50/50 for each emotion
    }
    
    # Save emotion map
    emotion_map_df = pd.DataFrame(emotion_map)
    emotion_map_file = os.path.join(data_dir, 'emotion_pd_map_balanced.csv')
    emotion_map_df.to_csv(emotion_map_file, index=False)
    
    # Convert dataset
    annotations_file = os.path.join(data_dir, 'pd_annotations_balanced.csv')
    
    # If annotations file doesn't exist, create it (simplified for this example)
    if not os.path.exists(annotations_file):
        # Find all image files (simplified)
        image_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    rel_path = os.path.join(root, file)
                    image_files.append(rel_path)
        
        # Create 50/50 balanced dataset
        n_samples = len(image_files)
        n_pd = n_samples // 2
        has_pd = [1] * n_pd + [0] * (n_samples - n_pd)
        np.random.shuffle(has_pd)
        
        # Create DataFrame
        annotations = pd.DataFrame({
            'image_file': image_files,
            'has_pd': has_pd
        })
        
        # Save annotations
        annotations.to_csv(annotations_file, index=False)
        print(f"Created balanced annotations with {n_pd} PD samples out of {n_samples} total")
    else:
        # Load existing annotations
        annotations = pd.read_csv(annotations_file)
        print(f"Loaded existing annotations with {sum(annotations['has_pd'])} PD samples out of {len(annotations)} total")
    
    # Split into train/val sets (80/20 split)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(annotations, test_size=0.2, stratify=annotations['has_pd'], random_state=42)
    
    # Save train/val splits
    train_df.to_csv(os.path.join(data_dir, 'pd_train_balanced.csv'), index=False)
    val_df.to_csv(os.path.join(data_dir, 'pd_val_balanced.csv'), index=False)
    
    # Step 2: Create datasets with feature inspection enabled
    train_dataset = PDDataset(
        root_dir=data_dir,
        annotations_file=os.path.join(data_dir, 'pd_train_balanced.csv'),
        inspect_features=True,
        balance_classes=True
    )
    
    val_dataset = PDDataset(
        root_dir=data_dir,
        annotations_file=os.path.join(data_dir, 'pd_val_balanced.csv'),
        inspect_features=True,
        balance_classes=True
    )
    
    # Step 3: Validate features - see if they can distinguish PD from non-PD
    feature_stats = validate_features(train_dataset)
    
    # Step 4: Try classical ML approach first
    ml_results = test_with_classical_ml(train_dataset, val_dataset)
    
    # Step 5: Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Step 6: Train simple model
    simple_model = SimplePDModel(input_size=13).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(simple_model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    trained_simple_model, simple_history = train_simple_model(
        simple_model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=15, scheduler=scheduler
    )
    
    # Save simple model
    torch.save(trained_simple_model.state_dict(), 'simple_pd_model.pth')
    print("Simple model saved to 'simple_pd_model.pth'")
    
    # Plot simple model history
    plot_training_history(simple_history, 'simple_model')
    
    # Step 7: Train temporal model
    temporal_model = TemporalPDModel(input_size=13, hidden_size=64, num_layers=2).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(temporal_model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    trained_temporal_model, temporal_history = train_temporal_model(
        temporal_model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=15, scheduler=scheduler
    )
    
    # Save temporal model
    torch.save(trained_temporal_model.state_dict(), 'temporal_pd_model.pth')
    print("Temporal model saved to 'temporal_pd_model.pth'")
    
    # Plot temporal model history
    plot_training_history(temporal_history, 'temporal_model')
    
    # Step 8: Compare models
    print("\nModel Comparison:")
    print("-" * 50)
    print(f"Random Forest Validation AUC: {ml_results['val_auc']:.4f}")
    print(f"Simple Model Validation AUC: {max(simple_history['val_auc']):.4f}")
    print(f"Temporal Model Validation AUC: {max(temporal_history['val_auc']):.4f}")
    
    best_model = "Random Forest" if ml_results['val_auc'] > max(simple_history['val_auc']) and ml_results['val_auc'] > max(temporal_history['val_auc']) else \
                "Simple Model" if max(simple_history['val_auc']) > max(temporal_history['val_auc']) else \
                "Temporal Model"
    
    print(f"\nBest model: {best_model}")
    
    # Create ensemble prediction function
    def ensemble_predict(image_path, rf_model, simple_model, temporal_model):
        """Make an ensemble prediction on a single image"""
        # Load and process image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not find image: {image_path}")
        
        img = cv2.resize(img, (224, 224))
        au_extractor = FacialActionUnitExtractor()
        aus = au_extractor.extract_aus(img)
        
        # Generate sequence
        au_sequence = []
        for i in range(30):  # 30 frames
            decay_factor = (30 - i) / 30
            variation = np.random.normal(0, 0.02 * decay_factor, size=aus.shape)
            au_sequence.append(aus + variation)
        
        # Calculate variances
        au_variances = np.var(au_sequence, axis=0)
        
        # Make predictions
        # 1. Random Forest
        rf_pred = rf_model.predict_proba([au_variances])[0, 1]
        
        # 2. Simple Model
        simple_model.eval()
        with torch.no_grad():
            simple_input = torch.FloatTensor(au_variances).to(device)
            simple_pred = simple_model(simple_input).item()
        
        # 3. Temporal Model
        temporal_model.eval()
        with torch.no_grad():
            sequence_np = np.array(au_sequence)
            temporal_input = torch.FloatTensor(sequence_np).unsqueeze(0).to(device)  # Add batch dimension
            temporal_pred = temporal_model(temporal_input).item()
        
        # Ensemble prediction (weighted average)
        ensemble_pred = (rf_pred * 0.4 + simple_pred * 0.3 + temporal_pred * 0.3)
        
        return {
            'rf_pred': rf_pred,
            'simple_pred': simple_pred,
            'temporal_pred': temporal_pred,
            'ensemble_pred': ensemble_pred,
            'pd_likelihood': "High" if ensemble_pred > 0.7 else "Medium" if ensemble_pred > 0.3 else "Low",
            'features': {
                'au_variances': au_variances,
                'key_features': {
                    'cheek_raiser': au_variances[0],
                    'lip_corner_puller': au_variances[1],
                    'brow_lowerer': au_variances[2],
                    'smile_asymmetry': au_variances[9],
                    'facial_mobility': au_variances[11]
                }
            }
        }
    
    # Save ensemble prediction function
    with open('ensemble_predict.py', 'w') as f:
        f.write("""import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
import pickle

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

class SimplePDModel(nn.Module):
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

class TemporalPDModel(nn.Module):
    def __init__(self, input_size=13, hidden_size=64, num_layers=2):
        super(TemporalPDModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: [batch_size, seq_len, hidden_size]
        
        # Attention weights
        attn_weights = self.attention(lstm_out)
        # attn_weights shape: [batch_size, seq_len, 1]
        
        # Apply attention
        context = torch.sum(attn_weights * lstm_out, dim=1)
        # context shape: [batch_size, hidden_size]
        
        # Final prediction
        output = self.fc(context)
        # output shape: [batch_size, 1]
        
        return output

def predict_pd_probability(image_path):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    
    # Load models
    # 1. Random Forest
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    # 2. Simple Model
    simple_model = SimplePDModel(input_size=13)
    simple_model.load_state_dict(torch.load('simple_pd_model.pth', map_location=device))
    simple_model.to(device)
    simple_model.eval()
    
    # 3. Temporal Model
    temporal_model = TemporalPDModel(input_size=13, hidden_size=64, num_layers=2)
    temporal_model.load_state_dict(torch.load('temporal_pd_model.pth', map_location=device))
    temporal_model.to(device)
    temporal_model.eval()
    
    # Load and process image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not find image: {image_path}")
    
    img = cv2.resize(img, (224, 224))
    au_extractor = FacialActionUnitExtractor()
    aus = au_extractor.extract_aus(img)
    
    # Generate sequence
    au_sequence = []
    for i in range(30):  # 30 frames
        decay_factor = (30 - i) / 30
        variation = np.random.normal(0, 0.02 * decay_factor, size=aus.shape)
        au_sequence.append(aus + variation)
    
    # Calculate variances
    au_variances = np.var(au_sequence, axis=0)
    
    # Make predictions
    # 1. Random Forest
    rf_pred = rf_model.predict_proba([au_variances])[0, 1]
    
    # 2. Simple Model
    with torch.no_grad():
        simple_input = torch.FloatTensor(au_variances).to(device)
        simple_pred = simple_model(simple_input).item()
    
    # 3. Temporal Model
    with torch.no_grad():
        sequence_np = np.array(au_sequence)
        temporal_input = torch.FloatTensor(sequence_np).unsqueeze(0).to(device)  # Add batch dimension
        temporal_pred = temporal_model(temporal_input).item()
    
    # Ensemble prediction (weighted average)
    ensemble_pred = (rf_pred * 0.4 + simple_pred * 0.3 + temporal_pred * 0.3)
    
    return {
        'rf_pred': rf_pred,
        'simple_pred': simple_pred,
        'temporal_pred': temporal_pred,
        'ensemble_pred': ensemble_pred,
        'pd_likelihood': "High" if ensemble_pred > 0.7 else "Medium" if ensemble_pred > 0.3 else "Low",
        'features': {
            'au_variances': au_variances,
            'key_features': {
                'cheek_raiser': au_variances[0],
                'lip_corner_puller': au_variances[1],
                'brow_lowerer': au_variances[2],
                'smile_asymmetry': au_variances[9],
                'facial_mobility': au_variances[11]
            }
        }
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    result = predict_pd_probability(sys.argv[1])
    print(f"PD Probability: {result['ensemble_pred']:.4f} ({result['pd_likelihood']} likelihood)")
    print(f"Individual Models: RF={result['rf_pred']:.4f}, Simple={result['simple_pred']:.4f}, Temporal={result['temporal_pred']:.4f}")
    print("\\nKey Features:")
    for name, value in result['features']['key_features'].items():
        print(f"  {name}: {value:.4f}")
""")
    print("Prediction script saved to 'ensemble_predict.py'")
    print("Usage: python ensemble_predict.py <image_path>")

# Create synthetic dataset function for testing
def create_synthetic_dataset(data_dir):
    """Create a synthetic dataset for testing"""
    print("Creating synthetic dataset...")
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    
    # Create annotations DataFrame
    n_samples = 1000
    n_pd = n_samples // 2
    has_pd = [1] * n_pd + [0] * (n_samples - n_pd)
    
    # Image filenames
    image_files = [f"synthetic_{i}.jpg" for i in range(n_samples)]
    
    # Create DataFrame
    annotations = pd.DataFrame({
        'image_file': image_files,
        'has_pd': has_pd
    })
    
    # Save annotations
    annotations_file = os.path.join(data_dir, 'pd_annotations_synthetic.csv')
    annotations.to_csv(annotations_file, index=False)
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(annotations, test_size=0.2, stratify=annotations['has_pd'], random_state=42)
    
    # Save train/val splits
    train_df.to_csv(os.path.join(data_dir, 'pd_train_synthetic.csv'), index=False)
    val_df.to_csv(os.path.join(data_dir, 'pd_val_synthetic.csv'), index=False)
    
    print(f"Created synthetic dataset with {n_pd} PD samples out of {n_samples} total")

if __name__ == "__main__":
    main()