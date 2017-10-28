#!/usr/bin/env python
from PIL import Image
import numpy as np
import tensorflow as tf
import random
import os

"""
basic functions to get uniform inputs through batches
"""

def get_pix(image, width, height):
    """
    return array of pixel red + array of pixel green + array of pixel blue, of size width * height * 3
    """
    image_resized = Image.open(image).resize((width, height))
    return np.array(image_resized).T.flatten()

def get_batch(size, width, height, dataset='train'):
    """
    return a random batch of size length
    each contains feature (pixelarray of r + g + b) and one hot vector target (0 for dog or 1 for cat) from train datatset
    """
    features = []
    targets = []
    for _ in range(size):
        type = 'dog' if random.randint(0,1) == 1 else 'cat'
        filename = '{dataset}/{type}.{nb}.jpg'.format(dataset=dataset, type=type, nb=random.randint(0, 12400))
        features.append(get_pix(filename, width, height))
        targets.append(np.eye(2)[0 if type == "dog" else 1])
    return features, targets

if __name__ == '__main__':
    if os.path.isdir("./train/") == False:
        print "miss folder for train data"
        exit (1)

    # hyper parameters

    # images will all be resized to 224 * 224 pixels
    width = 224
    height = 224

    # initialize variables

    # train
    for _ in range(1):
        batch_xs, batch_ys = get_batch(1, width, height)
        
    # Test trained model
