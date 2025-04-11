from utils import (
    load_and_prepare_data_advanced,
    encode_categorical,
    normalize_numeric,
    split_and_tensorize
)
from sklearn.metrics import classification_report

from model.advanced_model import DiabetesAdvancedModel

import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Load and prepare full-feature dataset
df = load_and_prepare_data_advanced("data/diabetes_dataset.csv")
df = encode_categorical(df)
df = normalize_numeric(df)
X_train, X_test, y_train, y_test = split_and_tensorize(df)

input_size = X_train.shape[1]

# Step 2: Define model
model = DiabetesAdvancedModel(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 3: Train loop
epochs = 100
for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Step 4: Evaluate
model.eval()
with torch.no_grad():
    preds = model(X_test)
    pred_classes = (preds >= 0.5).float()
    correct = (pred_classes == y_test).sum().item()
    accuracy = correct / y_test.size(0)
    print(f"\n✅ Advanced Model Accuracy: {accuracy * 100:.2f}%")

