import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load images


def load_images(folder_path):
    images = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            year = filename.split('_')[1].split('.')[0]
            img = cv2.imread(os.path.join(folder_path, filename))
            images[year] = img
    return images

# Crop to area of interest


def crop_images(images, x, y, width, height):
    cropped = {}
    for year, img in images.items():
        cropped[year] = img[y:y+height, x:x+width]
    return cropped

# Simple difference detection


def compare_images(img1, img2):
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate difference
    diff = cv2.absdiff(gray1, gray2)

    # Threshold to highlight significant changes
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    return diff, thresh
