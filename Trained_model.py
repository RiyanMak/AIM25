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

class FacialExpressionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(FacialExpressionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(18432, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        print(x.shape)  # Optional: For debugging
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FERDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        """
        FER2013 Dataset class (Modified for your directory structure)
        Args:
            image_dir (string): Path to the image directory (train/test).
            label_file (string): Path to the CSV file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.label_data = pd.read_csv(label_file)
        self.transform = transform

        # FER dataset has 7 emotion classes
        self.class_names = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }

        # Create a list of all image file paths and their corresponding labels
        self.image_paths = []
        self.labels = []
        for i in range(1, 8):  # Assuming folder names 1 to 7 for classes
            class_folder = os.path.join(image_dir, str(i))
            if os.path.exists(class_folder):
                for image_name in os.listdir(class_folder):
                    if image_name.endswith('.jpg') or image_name.endswith('.png'):  # Ensure image file types
                        self.image_paths.append(os.path.join(class_folder, image_name))
                        self.labels.append(i-1)  # Assuming class numbers 1 to 7, mapping to 0 to 6

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Get label (assumed to be in self.labels)
        label = self.labels[idx]
        return image, label

def main():
    # Define dataset paths
    train_dir = '/Users/kazangue/CS-Files/Projects-Clubs/Nuero_cam/AIM25/NueroCam/RAF_Dataset/DATASET/train'  # Modify to your actual path
    test_dir = '/Users/kazangue/CS-Files/Projects-Clubs/Nuero_cam/AIM25/NueroCam/RAF_Dataset/DATASET/test'    # Modify to your actual path
    train_labels_csv = '/Users/kazangue/CS-Files/Projects-Clubs/Nuero_cam/AIM25/NueroCam/RAF_Dataset/train_labels.csv'  # Modify to your actual path
    test_labels_csv = '/Users/kazangue/CS-Files/Projects-Clubs/Nuero_cam/AIM25/NueroCam/RAF_Dataset/test_labels.csv'    # Modify to your actual path

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = FERDataset(image_dir=train_dir, label_file=train_labels_csv, transform=transform)
    test_dataset = FERDataset(image_dir=test_dir, label_file=test_labels_csv, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FacialExpressionCNN(num_classes=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

    torch.save(model, 'facial_cnn_full.pth')
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

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds)

    print("Validation Accuracy:", acc)
    print("F1 Score:", f1)
    print("Classification Report:")
    print(report)

# Run the program
if __name__ == "__main__":
    main()
