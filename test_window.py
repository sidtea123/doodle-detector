import tkinter as tk
import numpy as np
import torch
from data_manager import *
from cnn import DoodleIdentifier
import random

def clear():
    global pixels
    pixels = np.zeros((28, 28))
    draw_pixels()
    evaluate()

def random_test():
    global pixels
    pixels = random.choice(images).copy() * 255
    draw_pixels()
    evaluate()

def evaluate():
    x = torch.unsqueeze(torch.unsqueeze(torch.tensor(pixels / 255., dtype=torch.float32), 0), 0)
    with torch.no_grad():
        y_hat = model(x)
        dist = torch.softmax(torch.squeeze(y_hat), dim=0)
        # don't even ask, I don't know how it works either
        output = dict(zip(categories, dist.tolist()))
        sorted_data = dict(sorted(output.items(), key=lambda item: item[1], reverse=True))
        prediction_label.config(text=f'{'_' * 20}\n\nPredictions:\n{"\n".join(f'{k}: {v*100:0.2f}%' for k, v in sorted_data.items())}\n{'_' * 20}')

def draw(event):
    p_w = int(canvas.cget('width')) // 28
    x, y = event.x // p_w, event.y // p_w

    if x < 0 or x > 27 or y < 0 or y > 27:
        return
    
    if DO_ERASER:
        draw_rect(x, y, 0)
    else:
        draw_rect(x, y, 255)
        if not DO_BLUR:
            return
        if x > 0:
            draw_rect(x - 1, y, light_gray)
        if y > 0:
            draw_rect(x, y - 1, light_gray)
        if x < 27:
            draw_rect(x + 1, y, light_gray)
        if y < 27:
            draw_rect(x, y + 1, light_gray)

def get_color(grayscale):
    return f"#{grayscale:02x}{grayscale:02x}{grayscale:02x}"

def eraser():
    global DO_ERASER
    if DO_ERASER:
        DO_ERASER = False
        eraser_btn.config(text='Eraser')
    else:
        DO_ERASER = True
        eraser_btn.config(text='Pen')

def draw_pixels():
    x = 0
    y = 0
    for row in pixels:
        for col in row:
            canvas.delete(f'{x},{y}')
            draw_rect(x, y, col)
            x += 1
        x = 0
        y += 1

def draw_rect(x, y, col):
    if pixels[y,x] == 255 and col == light_gray:
        return
    if pixels[y,x] == light_gray and col == light_gray:
        col = gray
    sX, sY = x * p_w, y * p_w
    canvas.create_rectangle(sX, sY, sX + p_w, sY + p_w, fill=get_color(int(col)), tags=f'{x},{y}')
    if col != pixels[y,x]:
        pixels[y,x] = col
        evaluate()

images, _ = get_testing_data()
images = np.squeeze(images, axis=1)
model = DoodleIdentifier()
model.load_state_dict(torch.load(MODEL_URL, weights_only=True))
model.eval()

root = tk.Tk()
root.geometry('630x570')

canvas = tk.Canvas(root, width=476, height=476)
canvas.place(anchor='n', x=250, y=(500-476)/2 + 35)

canvas.bind('<Button-1>', draw)
canvas.bind('<B1-Motion>', draw)

pixels = np.zeros((28, 28))
p_w = int(canvas.cget('width')) // 28
gray = 80
light_gray = 60
DO_ERASER = False
DO_BLUR = False

draw_pixels()

title = tk.Label(root, text='Drawing Classifier (CNN)', font=("Helvetica", 20, "bold"))
title.place(anchor='n', x=630//2, y=5)

clear_btn = tk.Button(root, text='Clear', command=clear)
clear_btn.place(anchor='n', x=100, y=535)

random_btn = tk.Button(root, text='Random Image', command=random_test)
random_btn.place(anchor='n', x=250, y=535)

eraser_btn = tk.Button(root, text="Eraser", command=eraser)
eraser_btn.place(anchor='n', x=400, y=535)

prediction_label = tk.Label(root, text='No Prediction...', justify='left')
prediction_label.place(anchor='nw', x=500, y=(500-476)/2 + 35)

evaluate()

root.mainloop()