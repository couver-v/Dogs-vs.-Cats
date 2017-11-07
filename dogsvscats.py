from PIL import Image
import tensorflow as tf
import numpy as np
import random
import tempfile
import sys
from tqdm import tqdm

IMG_SIZE = 128
TF_IMG = False
NB_PIX = IMG_SIZE * IMG_SIZE
NB_CLASSES = 2
NB_CHANNELS = 3

def get_batch(size, dataset='train', path='train'):
    features = []
    targets = []
    data_folder = 'data-{}{}'.format('tf-' if TF_IMG else '', IMG_SIZE)
    if dataset == 'train':
        data_range = (1, 7500)
    if dataset == 'test':
        data_range = (7500, 8000)
    if dataset == 'validation':
        data_range = (10000, 12500)
    for _ in range(size):
        id = random.randint(0,1)
        type = 'dog' if id == 1 else 'cat'
        nb = random.randint(*data_range)
        filename = '{path}/{type}.{nb}.npy'.format(path=data_folder, type=type, nb=nb)
        img = np.load(filename)
        features.append(img)
        targets.append(np.eye(NB_CLASSES)[id])
    features = np.array(features)
    return features, targets

def init_graph_layers(x_image, training, keep_prob):
    initializer = tf.contrib.layers.xavier_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(0.001)

    conv1 = tf.layers.conv2d(
        inputs=x_image,
        filters=32,
        kernel_size=[3, 3],
        padding='SAME',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        use_bias=True,
        bias_initializer=initializer,
        bias_regularizer=regularizer,
        activation=tf.nn.relu)
    conv1 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=[3, 3],
        padding='SAME',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        use_bias=True,
        bias_initializer=initializer,
        bias_regularizer=regularizer,
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[2, 2], strides=(2, 2))
    
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding='SAME',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        use_bias=True,
        bias_initializer=initializer,
        bias_regularizer=regularizer,
        activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(
        inputs=conv2,
        filters=64,
        kernel_size=[3, 3],
        padding='SAME',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        use_bias=True,
        bias_initializer=initializer,
        bias_regularizer=regularizer,
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=[2, 2], strides=(2, 2))

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[3, 3],
        padding='SAME',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        use_bias=True,
        bias_initializer=initializer,
        bias_regularizer=regularizer,
        activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(
        inputs=conv3,
        filters=64,
        kernel_size=[3, 3],
        padding='SAME',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        use_bias=True,
        bias_initializer=initializer,
        bias_regularizer=regularizer,
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3, pool_size=[2, 2], strides=(2, 2))
    flatten = tf.reshape(pool3, [-1, int(IMG_SIZE / 8) * int(IMG_SIZE / 8) * 64])

    fc1 = tf.layers.dense(
        inputs=flatten,
        units=256,
        activation=tf.nn.relu,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        use_bias=True,
        bias_initializer=initializer,
        bias_regularizer=regularizer)
    fc1 = tf.layers.dropout(
        inputs=fc1,
        rate=keep_prob,
        training=training)
    fc2 = tf.layers.dense(
        inputs=fc1,
        units=256,
        activation=tf.nn.relu,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        use_bias=True,
        bias_initializer=initializer,
        bias_regularizer=regularizer)
    fc2 = tf.layers.dropout(
        inputs=fc2,
        rate=keep_prob,
        training=training)
    logits = tf.layers.dense(inputs=fc2, units=2)

    return logits

def init_graph(x_image):

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, NB_CHANNELS, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        # W_conv1 = weight_variable([5, 5, 32, 32])
        # b_conv1 = bias_variable([32])
        # h_conv1 = tf.nn.relu(conv2d(h_conv1, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        # W_conv2 = weight_variable([5, 5, 64, 64])
        # b_conv2 = bias_variable([64])
        # h_conv2 = tf.nn.relu(conv2d(h_conv2, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(tf.nn.relu(h_conv2))

    # with tf.name_scope('conv3'):
    #     W_conv3 = weight_variable([5, 5, 64, 64])
    #     b_conv3 = bias_variable([64])
    #     h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv2) + b_conv2)
    #     # W_conv3 = weight_variable([5, 5, 64, 64])
    #     # b_conv3 = bias_variable([64])
    #     # h_conv3 = tf.nn.relu(conv2d(h_conv3, W_conv2) + b_conv2)

    # with tf.name_scope('pool3'):
    #     h_pool3 = max_pool_2x2(h_conv3)

    # print(h_pool2)
    with tf.name_scope('fc1'):
        length = (IMG_SIZE / 4) * (IMG_SIZE / 4) * 64
        W_fc1 = weight_variable([length, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, length])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, NB_CLASSES])
        b_fc2 = bias_variable([NB_CLASSES])
        preactivation = tf.matmul(h_fc1, W_fc2) + b_fc2

    return preactivation, keep_prob


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.03)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


features = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, NB_CHANNELS])
targets = tf.placeholder(tf.float32, [None, NB_CLASSES])
training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)

preactivation = init_graph_layers(features, training, keep_prob)

with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=preactivation)
    cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(preactivation, 1), tf.argmax(targets, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

graph_location = './logs'
print('Saving graph to: %s' % graph_location)
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('loss', cross_entropy)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(graph_location)

with tf.Session() as sess:
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    test_features, test_targets = get_batch(500, dataset='test')
    for i in range(10000):
        batch_features, batch_targets = get_batch(40)
        for j in tqdm(range(25)):
            train_step.run(feed_dict={features: batch_features, targets: batch_targets, keep_prob: 0.5, training: True})
        summary = sess.run(merged, feed_dict={features: test_features, targets: test_targets, keep_prob: 1, training: False})
        writer.add_summary(summary, i)
        train_accuracy, loss = sess.run([accuracy, cross_entropy], feed_dict={features: test_features, targets: test_targets, keep_prob: 0.5, training: True})
        print('step {i}, cost: {loss} training accuracy {accuracy}'.format(i=i, loss=loss, accuracy=train_accuracy))
    print('test accuracy %g' % accuracy.eval(feed_dict={features: batch_features, targets: batch_targets, keep_prob: 1, training: False}))
