import csv
import numpy as np

def parse_image(image):
    output = np.zeros((255,255))

    for pixel in image:
        pass

    return output

def read_files():
    print('now parsing data...')
    images = []
    labels = []
    with open('doodle_data/master_doodle_dataframe.csv') as file:
        data = csv.reader(file)

        # skipping first row
        next(data)

        for row in data:
            images.append(row[1])
            labels.append(row[4])

    print(labels[1])

if __name__ == '__main__':
    read_files()