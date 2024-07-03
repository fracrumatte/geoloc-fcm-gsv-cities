import torch
import torch.nn.functional as F
import torch.nn as nn

class AvgPool(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.AAP = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.AAP(x)
        x = torch.flatten(x, 1)  # Flatten to [batch_size, num_features]
        return F.normalize(x, p=2, dim=1)