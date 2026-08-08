import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class DoodleDataset(Dataset):
    def __init__(self, images, labels):
        self.data, self.labels = prepare_tensors(images, labels)

        self.data = self.data.share_memory_()
        self.labels = self.labels.share_memory_()
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def generate_dataset_and_loader(images, labels, batch_size):
    dataset = DoodleDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

def prepare_tensors(images, labels):
     return torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

categories = [
    'airplane',
    'bicycle',
    'computer',
    'dog',
    'elephant',
    'fish',
    'guitar',
    'helicopter',
    'ice cream',
    'jail',
    'keyboard',
    'lantern',
    'mountain',
    'nose',
    'octopus',
    'pig',
    'rainbow',
    'snail',
    'triangle',
    'umbrella',
    'violin',
    'whale',
    'zebra'
]

CATEGORY_SIZE = 7500
TEST_CATEGORY_SIZE = 500
BATCH_SIZE = 128
NUM_CATEGORIES = len(categories)
IMAGE_WIDTH = 28
TOTAL_LENGTH = CATEGORY_SIZE * NUM_CATEGORIES
MODEL_URL = 'CNN_Model.pt'

def read_files():
    print('\nnow parsing data...\n')
    images = np.zeros((TOTAL_LENGTH, 1, IMAGE_WIDTH, IMAGE_WIDTH))
    labels = np.zeros(TOTAL_LENGTH)

    for i in range(NUM_CATEGORIES):
        # only loads in first chunk of actual data
        c_url = f'doodle_data/full_numpy_bitmap_{categories[i]}.npy'
        images[i*CATEGORY_SIZE:(i + 1)*CATEGORY_SIZE] = np.expand_dims(np.load(c_url)[:CATEGORY_SIZE].reshape((CATEGORY_SIZE,28,28)), axis=1) / 255.0
        labels[i*CATEGORY_SIZE:(i + 1)*CATEGORY_SIZE] = i
        # print(f'finished {categories[i]}')
    
    print('\nfinished parsing data...\n')
    return images, labels

def get_testing_data():
    print('\nnow grabbing testing data...\n')
    images = np.zeros((TEST_CATEGORY_SIZE * NUM_CATEGORIES, 1, IMAGE_WIDTH, IMAGE_WIDTH))
    labels = np.zeros(TEST_CATEGORY_SIZE * NUM_CATEGORIES)

    for i in range(NUM_CATEGORIES):
        # loads in some chunk of unused training data
        c_url = f'doodle_data/full_numpy_bitmap_{categories[i]}.npy'
        images[i*TEST_CATEGORY_SIZE:(i + 1)*TEST_CATEGORY_SIZE] = np.expand_dims(np.load(c_url)[CATEGORY_SIZE:CATEGORY_SIZE + TEST_CATEGORY_SIZE].reshape((TEST_CATEGORY_SIZE,28,28)), axis=1) / 255.0
        labels[i*TEST_CATEGORY_SIZE:(i + 1)*TEST_CATEGORY_SIZE] = i

    print('\nfinished grabbing testing data...\n')
    return images, labels

if __name__ == '__main__':
    images, _ = get_testing_data()
    print(images[0])