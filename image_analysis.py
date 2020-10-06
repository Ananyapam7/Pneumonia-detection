import os
import matplotlib.pyplot as plt
import cv2
import glob
import random
from multiprocessing import Pool
import seaborn as sns
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio


BASE_FOLDER = "../chest_xray"

TRAIN_DIR = os.path.join(BASE_FOLDER, 'train')
TEST_DIR = os.path.join(BASE_FOLDER, 'test')
VAL_DIR = os.path.join(BASE_FOLDER, 'val')

sets = ["train", "test", "val"]

FOLDERS = [TRAIN_DIR, TEST_DIR, VAL_DIR]

PNEUMONIA_IMAGES = []
NORMAL_IMAGES = []

for folder in FOLDERS:
    normal = glob.glob(os.path.join(folder, "NORMAL/*.jpeg"))
    pneumonia = glob.glob(os.path.join(folder, "PNEUMONIA/*.jpeg"))
    NORMAL_IMAGES.extend(normal)
    PNEUMONIA_IMAGES.extend(pneumonia)

print(f"Total Pneumonia Images: {len(PNEUMONIA_IMAGES)}")
print(f"Total Normal Images: {len(NORMAL_IMAGES)}")

random.shuffle(NORMAL_IMAGES)
random.shuffle(PNEUMONIA_IMAGES)
images = NORMAL_IMAGES[:50] + PNEUMONIA_IMAGES[:50]

#Viewing the images

fig=plt.figure(figsize=(15, 10))
columns = 4; rows = 2
for i in range(1, columns*rows +1):
    img = cv2.imread(images[i])
    img = cv2.resize(img, (128, 128))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.axis(False)

#Ben Graham's method  of using grayscale images and then applying Gaussian Blur to them.

fig=plt.figure(figsize=(15, 10))
columns = 4; rows = 2
for i in range(1, columns*rows +1):
    img = cv2.imread(images[i])
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.addWeighted (img, 4, cv2.GaussianBlur(img, (0,0), 512/10), -4, 128)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.axis(False)

#Background Subtraction Method

fgbg = cv2.createBackgroundSubtractorMOG2()

fig=plt.figure(figsize=(15, 10))
columns = 4; rows = 2
for i in range(1, columns*rows +1):
    img = cv2.imread(images[i])
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = fgbg.apply(img)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.axis(False)

# Canny Edge Detection

fig=plt.figure(figsize=(15, 10))
columns = 4; rows = 2
for i in range(1, columns*rows +1):
    img = cv2.imread(images[i])
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 80, 100)
    fig.add_subplot(rows, columns, i)
    plt.imshow(edges)
    plt.axis(False)


fig=plt.figure(figsize=(15, 10))
columns = 4; rows = 2
for i in range(1, columns*rows +1):
    img = cv2.imread(images[i])
    fig.add_subplot(rows, columns, i)
    plt.hist(img.ravel(),256,[0,256])
    plt.axis(False)

plt.show() 