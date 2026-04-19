import torch
import torch.nn as nn

class NCA(nn.Module):
    """
    Neural Constraint Archive model.
    Takes values and mask as input and reconstructs valid data.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        inp = torch.cat([x, mask], dim=1)
        return self.net(inp)