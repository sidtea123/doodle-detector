import torch
import torch.nn as nn

from data_manager import *
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class DoodleIdentifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),

            nn.Flatten(start_dim=1),
            nn.Linear(in_features=128*7*7, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=NUM_CATEGORIES)
        )

    def forward(self, x):
        return self.network(x)

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    model = DoodleIdentifier().to(device, non_blocking=True)

    images, labels = read_files()
    data_loader = generate_dataset_and_loader(images, labels, BATCH_SIZE)

    epochs = 15
    lr = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(data_loader) * epochs, eta_min=1e-6)

    print(f'\nfinished preparing model, now running on {device}...\n')

    for epoch in range(1, epochs + 1):
        mismatches = 0
        total_loss = 0
        for x, y in data_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_hat = model(x)

            loss = loss_function(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            mismatches += (torch.argmax(y_hat, dim=1) != y).sum().item()
            total_loss += loss.item() * y.size(0) 

        print(f'\nProgress at epoch {epoch}:')
        print(f'training accuracy: {((len(labels)-mismatches)/len(labels) * 100):.2f}%')
        print(f'loss: {total_loss / TOTAL_LENGTH}')
        print(f'lr: {optimizer.param_groups[0]['lr']}')
        torch.save(model.state_dict(), MODEL_URL)
        print(f'saved current model state to {MODEL_URL}.\n')

    print('\nfinished!')
    print(f'now saving model to {MODEL_URL}...\n')
    torch.save(model.state_dict(), MODEL_URL)
    print('save complete!')