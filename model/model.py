import torch
import torch.nn as nn

class DiabetesPredictor(nn.Module):
    def __init__(self, input_size):
        super(DiabetesPredictor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 16), # input -> hidden
            nn.ReLU(),                 # activation
            nn.Linear(16, 8),          # hidden -> smaller hidden
            nn.ReLU(),
            nn.Linear(8, 1),           #hidden -> output
            nn.Sigmoid()               #probability output

        )
    
    def forward(self, x): 
        return self.network(x)