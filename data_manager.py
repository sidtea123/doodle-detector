import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

class DoodleDataset(Dataset):
    def __init__(self, images, labels):
        self.data, self.labels = prepare_tensors(images, labels)

        self.data = self.data.share_memory_()
        self.labels = self.labels.share_memory_()

        self.transforms = v2.Compose([
            v2.RandomRotation(degrees=10),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomResizedCrop(size=(28,28), scale=(0.7,1.0), ratio=(0.9, 1.1), antialias=True)
        ])
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        return self.transforms(self.data[idx]), self.labels[idx]

def generate_dataset_and_loader(images, labels, batch_size):
    dataset = DoodleDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

def prepare_tensors(images, labels):
     return torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

categories = [
    "aircraft carrier", "airplane", "alarm clock", "ambulance", "angel", 
    "animal migration", "ant", "anvil", "apple", "arm", "asparagus", "axe", 
    "backpack", "banana", "bandage", "barn", "baseball", "baseball bat", 
    "basket", "basketball", "bat", "bathtub", "beach", "bear", "beard", "bed", 
    "bee", "belt", "bench", "bicycle", "binoculars", "bird", "birthday cake", 
    "blackberry", "blueberry", "book", "boomerang", "bottlecap", "bowtie", 
    "bracelet", "brain", "bread", "bridge", "broccoli", "broom", "bucket", 
    "bulldozer", "bus", "bush", "butterfly", "cactus", "cake", "calculator", 
    "calendar", "camel", "camera", "camouflage", "campfire", "candle", "cannon", 
    "canoe", "car", "carrot", "castle", "cat", "ceiling fan", "cello", 
    "cell phone", "chair", "chandelier", "church", "circle", "clarinet", "clock", 
    "cloud", "coffee cup", "compass", "computer", "cookie", "cooler", "couch", 
    "cow", "crab", "crayon", "crocodile", "crown", "cruise ship", "cup", 
    "diamond", "dishwasher", "diving board", "dog", "dolphin", "donut", "door", 
    "dragon", "dresser", "drill", "drums", "duck", "dumbbell", "ear", "elbow", 
    "elephant", "envelope", "eraser", "eye", "eyeglasses", "face", "fan", 
    "feather", "fence", "finger", "fire hydrant", "fireplace", "firetruck", 
    "fish", "flamingo", "flashlight", "flip flops", "floor lamp", "flower", 
    "flying saucer", "foot", "fork", "frog", "frying pan", "garden", 
    "garden hose", "giraffe", "goatee", "golf club", "grapes", "grass", "guitar", 
    "hamburger", "hammer", "hand", "harp", "hat", "headphones", "hedgehog", 
    "helicopter", "helmet", "hexagon", "hockey puck", "hockey stick", "horse", 
    "hospital", "hot air balloon", "hot dog", "hot tub", "hourglass", "house", 
    "house plant", "hurricane", "ice cream", "jacket", "jail", "kangaroo", 
    "key", "keyboard", "knee", "knife", "ladder", "lantern", "laptop", "leaf", 
    "leg", "light bulb", "lighter", "lighthouse", "lightning", "line", "lion", 
    "lipstick", "lobster", "lollipop", "mailbox", "map", "marker", "matches", 
    "megaphone", "mermaid", "microphone", "microwave", "monkey", "moon", 
    "mosquito", "motorbike", "mountain", "mouse", "moustache", "mouth", "mug", 
    "mushroom", "nail", "necklace", "nose", "ocean", "octagon", "octopus", 
    "onion", "oven", "owl", "paintbrush", "paint can", "palm tree", "panda", 
    "pants", "paper clip", "parachute", "parrot", "passport", "peanut", "pear", 
    "peas", "pencil", "penguin", "piano", "pickup truck", "picture frame", 
    "pig", "pillow", "pineapple", "pizza", "pliers", "police car", "pond", 
    "pool", "popsicle", "postcard", "potato", "power outlet", "purse", "rabbit", 
    "raccoon", "radio", "rain", "rainbow", "rake", "remote control", 
    "rhinoceros", "rifle", "river", "roller coaster", "rollerskates", 
    "sailboat", "sandwich", "saw", "saxophone", "school bus", "scissors", 
    "scorpion", "screwdriver", "sea turtle", "see saw", "shark", "sheep", 
    "shoe", "shorts", "shovel", "sink", "skateboard", "skull", "skyscraper", 
    "sleeping bag", "smiley face", "snail", "snake", "snorkel", "snowflake", 
    "snowman", "soccer ball", "sock", "speedboat", "spider", "spoon", 
    "spreadsheet", "square", "squiggle", "squirrel", "stairs", "star", "steak", 
    "stereo", "stethoscope", "stitches", "stop sign", "stove", "strawberry", 
    "streetlight", "string bean", "submarine", "suitcase", "sun", "swan", 
    "sweater", "swing set", "sword", "syringe", "table", "teapot", "teddy-bear", 
    "telephone", "television", "tennis racquet", "tent", "The Eiffel Tower", 
    "The Great Wall of China", "The Mona Lisa", "tiger", "toaster", "toe", 
    "toilet", "tooth", "toothbrush", "toothpaste", "tornado", "tractor", 
    "traffic light", "train", "tree", "triangle", "trombone", "truck", 
    "trumpet", "t-shirt", "umbrella", "underwear", "van", "vase", "violin", 
    "washing machine", "watermelon", "waterslide", "whale", "wheel", "windmill", 
    "wine bottle", "wine glass", "wristwatch", "yoga", "zebra", "zigzag"
]

# total size ~145,000
CATEGORY_SIZE = 1000
TEST_CATEGORY_SIZE = 10
BATCH_SIZE = 64
NUM_CATEGORIES = len(categories)
IMAGE_WIDTH = 28
TOTAL_LENGTH = CATEGORY_SIZE * NUM_CATEGORIES
MODEL_URL = 'models/CNN_Model.pt'

def read_files():
    print('\nnow parsing data...\n')
    images = np.zeros((TOTAL_LENGTH, 1, IMAGE_WIDTH, IMAGE_WIDTH))
    labels = np.zeros(TOTAL_LENGTH)
    character = ''

    for i in range(NUM_CATEGORIES):
        if categories[i][0] != character:
            print(f'starting {categories[i][0]}')
            character = categories[i][0]

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