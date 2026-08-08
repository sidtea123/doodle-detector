import torch
from data_manager import *
from cnn import DoodleIdentifier

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DoodleIdentifier()
    model.load_state_dict(torch.load(MODEL_URL, weights_only=True))
    model.eval()
    images, labels = get_testing_data()
    x, y = prepare_tensors(images, labels)

    with torch.no_grad():
        y_hat = model(x)

        predictions = torch.argmax(y_hat, dim=1)
        mismatches = (predictions != y).sum().item()

        print(f'test accuracy: {((len(y)-mismatches)/len(y) * 100):.2f}%')