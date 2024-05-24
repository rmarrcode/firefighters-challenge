import torch
import torch.nn as nn

class PointNetPolylineEncode(nn.Module):
    def __init__(self, in_channels, hidden_dim, output_dims):
        super().__init__()
        self.mlp = nn.Sequential([
            nn.Linear(in_channels, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dims)
        ])
    def forward(self, polylines):
        x = self.mlp(polylines)
        x = x.max(dim=0)