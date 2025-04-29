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
import matplotlib.pyplot as plt
import numpy as np

class NeurologicalCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(NeurologicalCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Changed to 3 input channels for RGB
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

class NeurologicalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset class for neurological condition images
        Args:
            root_dir (string): Path to the image directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Class mapping
        self.class_names = {
            'Alzimers': 0,
            'Parkenson': 1, 
            'Stroke': 2
        }
        
        # Create a list of all image file paths and their corresponding labels
        self.image_paths = []
        self.labels = []
        
        # Walk through all folders and get images
        for class_name in self.class_names.keys():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for image_name in os.listdir(class_dir):
                    if image_name.endswith('.jpg') or image_name.endswith('.png'):
                        self.image_paths.append(os.path.join(class_dir, image_name))
                        self.labels.append(self.class_names[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')  # Convert to RGB
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
                
            # Get label
            label = self.labels[idx]
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder if image loading fails
            if self.transform:
                placeholder = torch.zeros((3, 224, 224))
                return placeholder, self.labels[idx]
            else:
                return None, self.labels[idx]

def plot_training_history(train_losses, val_accuracies):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define dataset path
    data_dir = '/Users/riyan/Downloads/Test_Data'
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to standard size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Create dataset
    full_dataset = NeurologicalDataset(root_dir=data_dir, transform=transform)
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Print dataset information
    print(f"Total samples: {len(full_dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Check class distribution
    class_counts = [0, 0, 0]
    for _, label in full_dataset:
        class_counts[label] += 1
    
    print("Class distribution:")
    for class_name, count in zip(full_dataset.class_names.keys(), class_counts):
        print(f"{class_name}: {count} images")
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the model
    model = NeurologicalCNN(num_classes=3).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 20
    
    # Lists to store metrics
    train_losses = []
    val_accuracies = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                # Store predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        val_accuracies.append(acc)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {acc:.4f}")
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Final Evaluation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate final metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, 
                                  target_names=list(full_dataset.class_names.keys()))
    
    print("\nFinal Results:")
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(report)
    
    # Plot training history
    plot_training_history(train_losses, val_accuracies)
    
    # Save the model
    torch.save(model.state_dict(), 'neurological_cnn_model.pth')
    print("Model saved successfully.")

# Run the program
if __name__ == "__main__":
    main()