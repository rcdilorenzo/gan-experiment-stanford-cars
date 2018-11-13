import os
import numpy as np
from matplotlib.image import imread
from PIL import Image
from toolz.curried import *
from funcy import rpartial

# ===================================================
# Determine basic image dimension statistics
# ===================================================
#
# https://rcd.ai/image-resize-prep-for-training/


PATH = '~/workspaces/data/stanford-cars/'

folder_path = partial(os.path.join, os.path.expanduser(PATH))
is_color = lambda data: len(data.shape) == 3

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def normalize_image(x):
    return x.astype(np.float32) / 255.0

def standardize(image):
    return rgb2gray(image) if is_color(image) else image

def x_data(set_type):
    pipeline = compose(
        standardize,
        normalize_image,
        imread,
        partial(os.path.join, folder_path(set_type))
    )
    file_paths = pipe(
        set_type,
        folder_path,
        os.listdir,
        list
    )
    return np.array([pipeline(path) for path in file_paths])

def load_data(subfolder='resized_extra'):
    return x_data('cars_train/' + subfolder), x_data('cars_test/' + subfolder)
