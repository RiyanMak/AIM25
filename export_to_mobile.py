import torch
from Trained_model import FacialExpressionCNN

model = FacialExpressionCNN(num_classes=7)
model.load_state_dict(torch.load("facial_cnn.pth", map_location=torch.device('cpu')))
model.eval()

# Example input for tracing the model
example_input = torch.rand(1, 1, 48, 48)

# Convert to TorchScript via tracing
traced_model = torch.jit.trace(model, example_input)

# Save the model
traced_model.save("facial_expression_model.pt")

print("Model exported as TorchScript to 'facial_expression_model.pt'")