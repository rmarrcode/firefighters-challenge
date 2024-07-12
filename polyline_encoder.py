import torch
import torch.nn as nn

class PointNetPolylineEncode(nn.Module):
    def __init__(self, in_channels, hidden_dim, output_dims, pool_size):
        super().__init__()
        self.pool_size = pool_size
        self.mlp = nn.Sequential([
            nn.Linear(in_channels, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dims)
        ])
    def forward(self, polylines):
        x = self.mlp(polylines)
        x = nn.MaxPool(self.pool_size, stride=1)
        return x