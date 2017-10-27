#!/usr/bin/env python
from PIL import Image
import numpy as np
import tensorflow as tf
import random
import os

def get_pix(im):
    """
    takes image of file jpg
    return array of pixel red + array of pixel green + array of pixel blue
    """
    pix = list(im.getdata())
    width, height = im.size
    pixArray = lambda pix, rank: [pix[y * width + x][rank] for x in range(width) for y in range(height)]
    return pixArray(pix, 0) + pixArray(pix, 1) + pixArray(pix, 2)

def get_batch(size, dataset='train'):
    """
    return a random batch of size length
    each contains feature (pixelarray of r + g + b) and one hot vector target (0 for dog or 1 for cat) from train datatset
    """
    features = []
    targets = []
    for _ in range(size):
        type = 'dog' if random.randint(0,1) == 1 else 'cat'
        nb = random.randint(0, 12400)
        filename = '{type}.{nb}.jpg'.format(type=type, nb=nb)
        im = Image.open('{dataset}/{filename}'.format(dataset=dataset, filename=filename))
        features.append(get_pix(im))
        targets.append(np.eye(2)[0 if type == "dog" else 1])
    return features, targets

if __name__ == '__main__':
    if os.path.isdir("./train/") == False:
        print "miss folder for train data"
        exit (1)
        
    # initialize variables

    # train
    for _ in range(50):
        batch_xs, batch_ys = get_batch(20)

    # Test trained model
