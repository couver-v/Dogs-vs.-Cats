from PIL import Image
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random
import tempfile
import os

IMG_SIZE = 64
NB_PIX = IMG_SIZE * IMG_SIZE
NB_CLASSES = 2
NB_CHANNELS = 3

def process_img(src, new_size=64, pad=True):
    height, width, _ = src.shape
    dst = src
    if height != width:
        if pad:
            size = max(height, width)
            oh = (size - height) // 2
            ow = (size - width) // 2
            dst = tf.image.pad_to_bounding_box(image=dst, offset_height=oh, offset_width=ow, target_height=size, target_width=size)
        else:
            size = min(height, width)
            oh = (height - size) // 2
            ow = (width - size) // 2
            dst = tf.image.crop_to_bounding_box(image=dst, offset_height=oh, offset_width=ow, target_height=size, target_width=size)

    assert(dst.shape[0] == dst.shape[1])

    size, _, _ = dst.shape
    if size > new_size:
        method = tf.image.ResizeMethod.AREA
    else:
        method = tf.image.ResizeMethod.BICUBIC
    if size != new_size:
        size_tensor = tf.constant([new_size, new_size], dtype=tf.int32)
        dst = tf.image.resize_images(images=dst, size=size_tensor, method=method)
    return dst

def get_img_tensor(sess, filename):
        im = Image.open(filename)
        pixels = list(im.getdata())
        width, height = im.size
        p_img = tf.placeholder(tf.float32)
        img = tf.reshape(p_img, [height, width, NB_CHANNELS])
        return sess.run(process_img(img, new_size=IMG_SIZE), feed_dict={p_img: pixels})

def process(path='train'):
    data_folder = 'data-{}'.format(IMG_SIZE)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for type in ['dog', 'cat']:
            print('Process {}s'.format(type))
            for i in tqdm(range(1, 12501)):
                output_file = '{path}/{type}.{nb}.npy'.format(path=data_folder, type=type, nb=i)
                if not os.path.exists(output_file):
                    filename = '{path}/{type}.{nb}.jpg'.format(path=path, type=type, nb=i)
                    img = np.array(get_img_tensor(sess, filename))
                    np.save(output_file, img)

process()
