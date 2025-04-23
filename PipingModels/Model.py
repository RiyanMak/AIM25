import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from PIL import Image
import pandas as pd
import os
import numpy as np

class FacialExpressionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(FacialExpressionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Use adaptive pooling to get fixed size output regardless of input dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Calculate the flattened size: 128 channels * 6 * 6 = 4608
        self.fc1 = nn.Linear(4608, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # Add adaptive pooling to ensure fixed output size
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class RAFDataset(Dataset):
    def __init__(self, image_dir, label_file=None, transform=None):
        """
        RAF Dataset class
        Args:
            image_dir (string): Path to the image directory
            label_file (string): Path to label list file (optional)
            transform (callable, optional): Optional transform to apply
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # RAF-DB has 7 emotion classes
        self.emotion_classes = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        
        # Create image paths and labels lists
        self.image_paths = []
        self.labels = []
        
        # Try to load from directory structure
        try:
            for i in range(1, 8):  # Assuming folder names 1 to 7 for classes
                class_folder = os.path.join(image_dir, str(i))
                if os.path.exists(class_folder):
                    for image_name in os.listdir(class_folder):
                        if image_name.endswith('.jpg') or image_name.endswith('.png'):
                            self.image_paths.append(os.path.join(class_folder, image_name))
                            self.labels.append(i-1)  # Mapping 1-7 to 0-6
            
            if not self.image_paths:  # If directory structure didn't work, try flat directory
                emotion_mapping = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6}
                for image_name in os.listdir(image_dir):
                    if image_name.endswith('.jpg') or image_name.endswith('.png'):
                        # Try to extract emotion from filename
                        for emotion_code, emotion_idx in emotion_mapping.items():
                            if f"_{emotion_code}_" in image_name:
                                self.image_paths.append(os.path.join(image_dir, image_name))
                                self.labels.append(emotion_idx)
                                break
        
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Handle empty dataset gracefully
            if not self.image_paths:
                print("Warning: No images found. Please check your directory structure.")
        
        print(f"Loaded {len(self.image_paths)} images")
        if self.labels:
            # Print class distribution
            unique, counts = np.unique(self.labels, return_counts=True)
            distribution = dict(zip([self.emotion_classes[i] for i in unique], counts))
            print(f"Class distribution: {distribution}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            # Get label
            label = self.labels[idx]
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder in case of error
            if self.transform:
                return torch.zeros((1, 48, 48)), 0
            else:
                return Image.new('L', (48, 48)), 0

# Function to save model
def save_model(model, path, accuracy=None):
    """Save the trained model to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model state dictionary
    torch.save(model.state_dict(), path)
    
    # Also save with accuracy in filename if provided
    if accuracy is not None:
        accuracy_path = path.replace('.pth', f'_acc{accuracy:.2f}.pth')
        torch.save(model.state_dict(), accuracy_path)
        
    print(f"Model saved to {path}")
    if accuracy is not None:
        print(f"Model with accuracy info saved to {accuracy_path}")

def train_model():
    # Define paths for dataset
    base_dir = '/Users/riyan/Desktop/archive/DATASET'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    
    # Create model directory in the requested location
    model_dir = '/Users/riyan/Desktop/AIM Spring 2025/PipingModels'
    os.makedirs(model_dir, exist_ok=True)
    
    # Define model paths
    model_path = os.path.join(model_dir, 'emotion_model.pth')
    best_model_path = os.path.join(model_dir, 'best_emotion_model.pth')
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = RAFDataset(image_dir=train_dir, transform=train_transform)
    test_dataset = RAFDataset(image_dir=test_dir, transform=test_transform)
    
    # Check if datasets are not empty
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("Error: One or both datasets are empty. Please check your directory paths.")
        return
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                       "mps" if torch.backends.mps.is_available() else 
                       "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = FacialExpressionCNN(num_classes=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training variables
    num_epochs = 15
    best_accuracy = 0.0
    history = {
        'train_loss': [],
        'val_accuracy': []
    }
    
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        # Calculate epoch loss
        epoch_loss = running_loss / len(train_dataset)
        history['train_loss'].append(epoch_loss)
        
        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        history['val_accuracy'].append(accuracy)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, best_model_path, accuracy)
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(model_dir, f'emotion_model_epoch_{epoch+1}.pth')
            save_model(model, checkpoint_path)
    
    # Save final model
    save_model(model, model_path, accuracy)
    
    # Final evaluation
    f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds)
    
    print("\nFinal Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(report)
    
    print(f"\nTraining complete. Final model saved to: {model_path}")
    print(f"Best model saved to: {best_model_path}")
    
    return model, best_accuracy

if __name__ == "__main__":
    train_model()