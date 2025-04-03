import torch
import torch.nn as nn

class TemporalFusionModule(nn.Module):
    def __init__(self, cnn_output_size=64 * 25 * 25, static_feature_size=68 * 2, lstm_hidden_size=128, num_classes=7):
        super(TemporalFusionModule, self).__init__()

        # LSTM for processing CNN features over time
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Static feature processing (e.g., MediaPipe facial landmarks)
        self.static_fc = nn.Sequential(
            nn.Linear(static_feature_size, 64),
            nn.ReLU()
        )

        # Feature fusion layer (LSTM output + static features)
        self.fusion = nn.Sequential(
            nn.Linear(lstm_hidden_size + 64, 64),
            nn.ReLU()
        )

        # Output layer with sigmoid activation
        self.output = nn.Sequential(
            nn.Linear(64, num_classes),
            nn.Sigmoid()  
        )

    def forward(self, cnn_sequence, static_features):
        # cnn_sequence: shape (batch_size, seq_len, cnn_output_size)
        # static_features: shape (batch_size, static_feature_size)

        lstm_out, _ = self.lstm(cnn_sequence)      
        lstm_final = lstm_out[:, -1, :]                

        static_out = self.static_fc(static_features)    # Project static features

        combined = torch.cat((lstm_final, static_out), dim=1)
        fused = self.fusion(combined)

        return self.output(fused)