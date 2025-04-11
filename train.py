from utils import (
    load_and_prepare_data,
    select_and_clean_features,
    encode_categorical,
    normalize_numeric,
    split_and_tensorize
)

from model.model import DiabetesPredictor

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.onnx

# STEP 1: Load + Preprocess
df = load_and_prepare_data("data/diabetes_dataset.csv")
df = select_and_clean_features(df)
df = encode_categorical(df)
df = normalize_numeric(df)

X_train, X_test, y_train, y_test = split_and_tensorize(df)

input_size = X_train.shape[1]  # Number of features

# STEP 2: Init Model
model = DiabetesPredictor(input_size)

# Binary classification → use Binary Cross Entropy Loss
criterion = nn.BCELoss()

# Optimizer: Adam usually works well
optimizer = optim.Adam(model.parameters(), lr=0.01)

# STEP 3: Training Loop
epochs = 100

for epoch in range(epochs):
    model.train()

    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")



# === STEP 4: EVALUATE ON TEST SET ===
model.eval()  # Set model to evaluation mode (turns off dropout, etc.)

with torch.no_grad():  # Don't track gradients during evaluation
    predictions = model(X_test)
    predicted_classes = (predictions >= 0.5).float()  # Convert probabilities to 0 or 1

    correct = (predicted_classes == y_test).sum().item()
    total = y_test.size(0)
    accuracy = correct / total

    print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")


dummy_input = torch.randn(1, X_train.shape[1])
torch.onnx.export(
    model,
    dummy_input,
    "diabetes_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print("ONNX model exported.")