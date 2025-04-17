import torch
import torch.nn as nn

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