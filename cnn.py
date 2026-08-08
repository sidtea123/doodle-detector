import torch
import torch.nn as nn

from data_manager import *

class DoodleIdentifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=784, out_features=NUM_CATEGORIES),
            nn.BatchNorm1d(num_features=NUM_CATEGORIES)
        )

    def forward(self, x):
        return self.network(x)

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DoodleIdentifier().to(device, non_blocking=True)

    images, labels = read_files()
    data_loader = generate_dataset_and_loader(images, labels, BATCH_SIZE)

    epochs = 40
    lr = 0.00001

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    print(f'\nfinished preparing model, now running on {device}...\n')

    for epoch in range(1, epochs + 1):
        for x, y in data_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_hat = model(x)
            loss = loss_function(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch % 5 == 0):
            print(f'loss at epoch = {epoch} is {loss}.\nlr is now {optimizer.param_groups[0]['lr']}.\nnow saving.\n')
            torch.save(model.state_dict(), MODEL_URL)
            print(f'save at epoch {epoch} successful.')

        scheduler.step(loss.item())

    print(f'\n{'=' * 20}\n\nfinished training, now saving model to {MODEL_URL}...\n')
    torch.save(model.state_dict(), MODEL_URL)