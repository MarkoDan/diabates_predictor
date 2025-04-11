import torch 
import torch.nn as nn

class DiabetesAdvancedModel(nn.Module):
    def __init__(self, input_size):
        super(DiabetesAdvancedModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)