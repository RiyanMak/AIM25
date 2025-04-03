import torch
from CNN import CNNLSTMNet
import numpy as np

# simulate facial landmark features from week 3
def load_sample_landmarks(batch_size=4):
    landmarks = np.random.rand(batch_size, 68, 3)[:, :, :2] 
    return landmarks.reshape(batch_size, -1).astype(np.float32) 

# Load a sample frame sequence
def load_sample_frame_sequences(batch_size=4, seq_len=4):
    return torch.randn(batch_size, seq_len, 3, 100, 100)

# Test the forward pass
def run_forward_pass(model, frames_tensor, landmarks_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(frames_tensor, landmarks_tensor)
    return outputs

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CNNLSTMNet().to(device)

    batch_size = 4
    # Load real or simulated input tensors
    landmarks = load_sample_landmarks(batch_size=batch_size)
    landmarks_tensor = torch.tensor(landmarks).to(device)

    frames_tensor = load_sample_frame_sequences(batch_size=batch_size).to(device)

    # Run forward pass
    outputs = run_forward_pass(model, frames_tensor, landmarks_tensor)
    print("Output shape:", outputs.shape)
    print("Raw model output:")
    print(outputs)

    # Simulate validation metric
    predicted = torch.argmax(outputs, dim=1).cpu().numpy()
    print("Predicted emotion classes:", predicted)

if __name__ == "__main__":
    main()
