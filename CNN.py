import torch
import torch.nn as nn
import torchvision.transforms as transforms

class CNNLSTMNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
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
        
        # Calculate CNN output size (assuming 64x64 input images)
        self.cnn_output_size = 64 * 16 * 16  # Adjust based on your input dimensions
        
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
            nn.Linear(64, num_classes),
            nn.Sigmoid()
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

# Usage example
def main():
    # Model parameters
    in_channels = 3  # RGB images
    num_classes = 1  # Binary classification (has neurological disorder or not)
    
    # Create model
    model = CNNLSTMNet(in_channels, num_classes)
    
    # Example input
    batch_size = 4
    seq_len = 4  # 4 frames sequence
    height = 64
    width = 64
    
    frames = torch.randn(batch_size, seq_len, in_channels, height, width)
    landmarks = torch.randn(batch_size, 68 * 2)  # 68 landmarks with x,y coordinates
    
    # Forward pass
    output = model(frames, landmarks)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()


