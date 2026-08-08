import torch
import torch.nn as nn

from data_manager import *

class DoodleIdentifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.1),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.1),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.Dropout(p=0.2),

            nn.Flatten(start_dim=1),
            nn.Linear(in_features=128*7*7, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=23)
        )

    def forward(self, x):
        return self.network(x)

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DoodleIdentifier().to(device, non_blocking=True)

    images, labels = read_files()
    data_loader = generate_dataset_and_loader(images, labels, BATCH_SIZE)

    epochs = 20
    lr = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(data_loader) * epochs, eta_min=1e-6)

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
            scheduler.step()

        predictions = torch.argmax(y_hat, dim=1)
        mismatches = (predictions != y).sum().item()
        print(f'\nProgress at epoch {epoch}:')
        print(f'training accuracy: {((len(y)-mismatches)/len(y) * 100):.2f}%')
        print(f'loss: {loss.item()}')
        print(f'lr: {optimizer.param_groups[0]['lr']}')
        torch.save(model.state_dict(), MODEL_URL)
        print(f'saved current model state to {MODEL_URL}.\n')

    print(f'\n\nfinished training with final accuracy: {((len(y)-mismatches)/len(y) * 100):.2f}%')
    print(f'now saving model to {MODEL_URL}...\n')
    torch.save(model.state_dict(), MODEL_URL)
    print('save complete!')