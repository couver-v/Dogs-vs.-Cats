#!/usr/bin/env python
from PIL import Image
import numpy as np
import random

IMG_SIZE = int(64)
NB_PIX = IMG_SIZE * IMG_SIZE
NB_CLASSES = int(2)
NB_CHANNELS = int(3)


def show_image():
    data_folder = 'data-{}'.format(IMG_SIZE)
    id = random.randint(0, 1)
    type = 'dog' if id == 1 else 'cat'
    nb = random.randint(0, 7500)
    filename = '{path}/{type}.{nb}.npy'.format(path=data_folder, type=type, nb=nb)
    fullsize_image = 'train/{type}.{nb}.jpg'.format(path=data_folder, type=type, nb=nb)
    array = np.load(filename)
    array = array.astype('uint8')
    Image.fromarray(array).show()
    Image.open(fullsize_image).show()


if __name__ == '__main__':
    show_image()
