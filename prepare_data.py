from PIL import Image
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random
import tempfile
import os

IMG_SIZE = 128
NB_PIX = IMG_SIZE * IMG_SIZE
NB_CLASSES = 2
NB_CHANNELS = 3

def get_img_tensor(filename):
    im = Image.open(filename)
    width, height = im.size
    size = max(height, width)
    new_im = Image.new('RGB', (size, size))
    new_im.paste(im, ((size - width) // 2, (size - height) // 2))
    new_im = new_im.resize((IMG_SIZE, IMG_SIZE))
    return new_im


def process(path='train'):
    data_folder = 'data-{}'.format(IMG_SIZE)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    for type in ['dog', 'cat']:
        print('Process {}s'.format(type))
        for i in tqdm(range(1, 12500)):
            output_file = '{path}/{type}.{nb}.npy'.format(path=data_folder, type=type, nb=i)
            if not os.path.exists(output_file):
                filename = '{path}/{type}.{nb}.jpg'.format(path=path, type=type, nb=i)
                img = np.array(get_img_tensor(filename))
                np.save(output_file, img)

process()
