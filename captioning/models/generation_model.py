import torch
import torch.nn as nn
from config import TARGET_IMAGE_SIZE

class SimpleConditionalAutoencoder(nn.Module):
    def __init__(self, condition_dim=10):  # Adjust condition dim as needed.
        super(SimpleConditionalAutoencoder, self).__init__()
        self.condition_dim = condition_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (H/2, W/2, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (H/4, W/4, 64)
            nn.ReLU()
        )

        self.fc_condition = nn.Linear(condition_dim, 64)

            self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> (H/2, W/2, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # -> (H, W, 3)
            nn.Sigmoid()
        )

    def forward(self, x, condition):
        # Encode image
        enc = self.encoder(x)  # shape: (batch, 64, H/4, W/4)
        # Process condition and reshape to add to the bottleneck
        batch_size, _, h, w = enc.size()
        cond = self.fc_condition(condition).unsqueeze(2).unsqueeze(3)
        cond = cond.expand(batch_size, 64, h, w)
        # Fuse encoded image and condition
        fused = enc + cond
        # Decode fused representation
        out = self.decoder(fused)
        return out

def build_generation_model(condition_dim=10):
    model = SimpleConditionalAutoencoder(condition_dim=condition_dim)
    return model
