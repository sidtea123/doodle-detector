import torch
import torch.nn as nn

class DoodleIdentifier(nn.Module):
    def __init__(self, in_channels, num_categories):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(10, num_categories)
        )

    def forward(self, x):
        return self.network(x)

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DoodleIdentifier(1,1).to(device)