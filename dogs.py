#!/usr/bin/env python
from PIL import Image
import numpy as np
import tensorflow as tf
import random
import os
import tempfile

def get_pix(image, width, height):
    """
    return array of pixel rgb, of size width * height * 3
    """
    image_resized = Image.open(image).resize((width, height))
    return np.array(image_resized).T.flatten()

def get_batch(size, width, height, dataset='train'):
    """
    return a random batch of size length
    each contains a feature pixel array and a one hot vector target (0 for dog or 1 for cat)
    from images 0 to 1000 in datatset
    """
    features = []
    targets = []
    for _ in range(size):
        type = 'dog' if random.randint(0,1) == 1 else 'cat'
        filename = '{dataset}/{type}.{nb}.jpg'.format(dataset=dataset, type=type, nb=random.randint(0, 9999))
        features.append(get_pix(filename, width, height))
        targets.append(np.eye(2)[0 if type == "dog" else 1])
    return np.array(features), np.array(targets)

def get_test(width, height, dataset='train'):
    """
    return complete list of features and targets for test
    """
    features = []
    targets = []
    for i in range(10000, 10500):
        for t in range(2):
            filename = '{dataset}/{type}.{nb}.jpg'.format(dataset=dataset, type="dog" if t == 0 else "cat", nb=i)
            features.append(get_pix(filename, width, height))
            targets.append(np.eye(2)[t])
    return features, targets

def graph(input_data):
    """
    build the graph
    """
    print "conv 32 layers"
    with tf.variable_scope('conv1') as scope:
        weights = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.03), name=scope.name)
        bias = tf.Variable(tf.truncated_normal([32]), name=scope.name)
        conv = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
        conv += bias
        conv = tf.nn.relu(conv, name=scope.name)

    print "pool max 2"
    with tf.variable_scope('pool1') as scope:
        pool = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=scope.name)

    print "conv 64 layers"
    with tf.variable_scope('conv2') as scope:
        weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.03), name=scope.name)
        bias = tf.Variable(tf.truncated_normal([64]), name=scope.name)
        conv = tf.nn.conv2d(pool, weights, [1, 1, 1, 1], padding='SAME')
        conv += bias
        conv = tf.nn.relu(pool, name=scope.name)

    print "pool max 2"
    with tf.variable_scope('pool2') as scope:
        pool = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=scope.name)

    print "linear layer 1"
    with tf.variable_scope('layer1') as scope:
        flattened = tf.reshape(pool, [-1, (width / 4) * (height / 4) * 32]) # pool is 32 layers of w * h array
        wd1 = tf.Variable(tf.truncated_normal([(width / 4) * (height / 4) * 32, 1000], stddev=0.03), name=scope.name)
        bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name=scope.name)
        dense_layer1 = tf.matmul(flattened, wd1) + bd1
        dense_layer1 = tf.nn.relu(dense_layer1)

    print "linear layer 2"
    with tf.variable_scope('layer2') as scope:
        wd2 = tf.Variable(tf.truncated_normal([1000, 2], stddev=0.03), name=scope.name)
        bd2 = tf.Variable(tf.truncated_normal([2], stddev=0.01), name=scope.name)
        dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
        y_ = tf.nn.softmax(dense_layer2)
        print "shape ", dense_layer2.shape

    return dense_layer2, y_

if __name__ == '__main__':

    if os.path.isdir("./train/") == False:
        print "miss folder for train data"
        exit (1)

    print "hyperparameters"
    width, height = 112, 112
    learning_rate = 0.0001
    epochs = 80
    batch_size = 40

    print "test dataset"
    test_features, test_targets = get_test(width, height)

    print "train variables"
    x = tf.placeholder(tf.float32, [None, width * height * 3])
    x_shaped = tf.reshape(x, [-1, width, height, 3])
    y = tf.placeholder(tf.float32, [None, 2])

    print "build graph"
    train_logits, y_ = graph(x_shaped)

    print "config network"
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_logits, labels=y))
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        total_batch = int(1000 / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = get_batch(batch_size, width, height)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
                print i

            test_acc = sess.run(accuracy, feed_dict={x: test_features, y: test_targets})
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))

        print("\nTraining complete!")
        print(sess.run(accuracy, feed_dict={x: test_features, y: test_targets}))

