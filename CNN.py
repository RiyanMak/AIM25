import os
import torch
import torch.nn as nn
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

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the CNN-LSTM model as we previously created
class CNNLSTMNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):  # RAF-DB has 7 emotion classes
        super(CNNLSTMNet, self).__init__()
        
        # CNN layers for feature extraction
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate CNN output size (assuming 100x100 input images as in RAF-DB)
        self.cnn_output_size = 64 * 25 * 25  # For 100x100 input images after 2 max pooling layers
        
        # LSTM layer for temporal processing
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        
        # Layer for static features (facial landmarks)
        self.static_feature_size = 68 * 2  # 68 landmarks with x,y coordinates
        self.static_features_fc = nn.Linear(self.static_feature_size, 64)
        
        # Feature fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 64),  # LSTM output + static features
            nn.ReLU()
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(64, num_classes)  # Changed to 7 classes for RAF-DB
        )
    
    def forward(self, frames_sequence, facial_landmarks):
        batch_size, seq_len, channels, height, width = frames_sequence.size()
        
        # Process each frame with CNN
        cnn_output = []
        for t in range(seq_len):
            # Extract frames at time t for all samples in the batch
            frame_t = frames_sequence[:, t, :, :, :]
            
            # Apply CNN to extract features
            features = self.cnn_layers(frame_t)
            
            # Flatten the CNN output
            features = features.view(batch_size, -1)
            
            cnn_output.append(features)
        
        # Stack the CNN outputs to create a sequence
        cnn_sequence = torch.stack(cnn_output, dim=1)
        
        # Process the sequence with LSTM
        lstm_out, _ = self.lstm(cnn_sequence)
        
        # Take the last output from LSTM
        lstm_out = lstm_out[:, -1, :]
        
        # Process static features (facial landmarks)
        static_features = self.static_features_fc(facial_landmarks)
        
        # Join features
        combined_features = torch.cat((lstm_out, static_features), dim=1)
        fused_features = self.fusion(combined_features)
        
        # Generate output
        output = self.output_layer(fused_features)
        
        return output

# Create a custom dataset for RAF-DB
class RAFDBDataset(Dataset):
    def __init__(self, root_dir, label_file, transform=None, is_train=True, seq_length=4):
        """
        Args:
            root_dir (string): Directory with all the images
            label_file (string): Path to the label file
            transform (callable, optional): Optional transform to be applied on an image
            is_train (bool): Whether this is training set or test set
            seq_length (int): Number of consecutive frames to use for each sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.seq_length = seq_length
        
        # Read the label file
        # Format is expected to be: image_name emotion_label
        self.labels_df = pd.read_csv(label_file, sep=' ', header=None, 
                                     names=['image_name', 'emotion'])
        
        # Subset to training or testing
        subset = 'train' if is_train else 'test'
        self.labels_df = self.labels_df[self.labels_df['image_name'].str.contains(subset)]
        
        # Generate frame sequences
        # For simplicity, we'll repeat the same image to create a sequence
        # In a real application, you would use consecutive video frames
        self.sequences = []
        for index, row in self.labels_df.iterrows():
            image_name = row['image_name']
            emotion = row['emotion'] - 1  # Convert 1-indexed to 0-indexed
            self.sequences.append((image_name, emotion))
            
        # For landmark features, we would load them from a file
        # Since we don't have actual landmark data, we'll generate random ones for demonstration
        self.landmark_features = np.random.rand(len(self.sequences), 68 * 2)
            
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        image_name, emotion = self.sequences[idx]
        img_path = os.path.join(self.root_dir, 'aligned', image_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform if defined
        if self.transform:
            image = self.transform(image)
        
        # Create a sequence by repeating the same image (for demonstration)
        # In real application, you would load consecutive frames
        sequence = image.unsqueeze(0).repeat(self.seq_length, 1, 1, 1)
        
        # Get landmarks (random for demonstration)
        landmarks = torch.FloatTensor(self.landmark_features[idx])
        
        return sequence, landmarks, emotion

# Define training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()
    
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
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
            running_corrects = 0
            
            # Iterate over data
            for inputs, landmarks, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                landmarks = landmarks.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, landmarks)
                    loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
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

# Main function to run the training pipeline
def main():
    # Data transformation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    

    data_dir = '/Users/riyan/Desktop/archive'  # Base directory for the dataset
    label_file = os.path.join(data_dir, 'list_partition_label.txt')  # Label file path
    
    # Create datasets
    train_dataset = RAFDBDataset(
        root_dir=data_dir,
        label_file=label_file,
        transform=data_transforms['train'],
        is_train=True
    )
    
    val_dataset = RAFDBDataset(
        root_dir=data_dir,
        label_file=label_file,
        transform=data_transforms['val'],
        is_train=False
    )
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    }
    
    # Initialize model
    model = CNNLSTMNet(in_channels=3, num_classes=7)  # RAF-DB has 7 emotion classes
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    model, val_acc_history = train_model(
        model, dataloaders, criterion, optimizer, num_epochs=25
    )
    
    # Save model
    torch.save(model.state_dict(), 'cnn_lstm_raf_db.pth')
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(val_acc_history)), [acc.cpu().numpy() for acc in val_acc_history])
    plt.title('Validation Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('training_history.png')
    
    # Evaluate model on test set
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, landmarks, labels in dataloaders['val']:
            inputs = inputs.to(device)
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs, landmarks)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print confusion matrix and classification report
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # RAF-DB emotion labels
    emotion_labels = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=emotion_labels))

if __name__ == "__main__":
    main()
