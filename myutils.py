from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import numpy as np
import tensorflow as tf


from myconfig import *

# supports numpy arrays and tensorflow tensors
def NDHWC_to_NCHWD(arr):
    try:
        return tf.transpose(arr, perm=(0,4,2,3,1))
    except:
        return arr.transpose((0,4,2,3,1))
def NCHWD_to_NDHWC(arr):
    try:
        return tf.transpose(arr, perm=(0,4,2,3,1))
    except:
        return arr.transpose((0,4,2,3,1))

def dataset_input_from_tfrecords(filenames, batch_size=1, num_epochs=1, shuffle=True):
    # filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)
    
    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    def parser(record):
        keys_to_features = {
            'case_name': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        
        case_name = parsed['case_name']
        # Perform additional preprocessing on the parsed data.
        image = tf.decode_raw(parsed['image_raw'], tf.uint8)
        image = tf.reshape(image, [DEPTH, HEIGHT, WIDTH, 2]) 
        label = tf.decode_raw(parsed['label_raw'], tf.uint8)
        label = tf.reshape(label, [DEPTH, HEIGHT, WIDTH, 2]) 
        image = tf.cast(image, tf.float32)
        
        return image, label, case_name
    
    # Use `Dataset.map()` to build a pair of a feature dictionary and a label 
    # tensor for each example.
    dataset = dataset.map(parser)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=128)
    dataset = dataset.batch(batch_size)
    # dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    
    features, labels, case_names = iterator.get_next()
    return features, labels, case_names

def dice_tf(logits, labels, threshold=0.5, axis=[1,2,3], smooth=1e-5):
    logits = tf.cast(logits, dtype=tf.float32)
    labels = tf.cast(labels, dtype=tf.float32)
    logits = tf.cast(logits > threshold, dtype=tf.float32)
    labels = tf.cast(labels > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(logits, labels), axis=axis)
    l = tf.reduce_sum(logits, axis=axis)
    r = tf.reduce_sum(labels, axis=axis)
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice

def dice_loss1(logits, labels, threshold=0.5, axis=[1,2,3], smooth=1e-5):
    # exponential_map = tf.exp(logits)
    # sum_exp = tf.reduce_sum(exponential_map, 4, keep_dims=True)
    # tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, 1, tf.shape(logits)[4]]))
    # prediction = tf.div(exponential_map,tensor_sum_exp)
    prediction = tf.nn.softmax(logits, axis=4)
    
    labels = tf.expand_dims(tf.cast(labels, dtype=tf.float32), axis=4)
    labels_expand = tf.concat([1-labels, labels], axis=4)
    eps = 1e-5
    intersection = tf.reduce_sum(prediction * labels_expand, axis=[1,2,3,4])
    union =  eps + tf.reduce_sum(prediction, axis=[1,2,3,4]) + tf.reduce_sum(labels_expand, axis=[1,2,3,4])
    loss = tf.reduce_mean(1.0 - (2 * intersection/ (union)))
    return loss

def dice_loss(logits, labels, threshold=0.5, axis=[1,2,3], smooth=1e-5):
    labels = tf.cast(labels, tf.float32)
    prediction = tf.nn.softmax(logits, axis=4)
    eps = 1e-5
    intersection = tf.reduce_sum(prediction[...,1] * labels, axis=[1,2,3])
    union =  eps + tf.reduce_sum(prediction[...,1], axis=[1,2,3]) + tf.reduce_sum(labels, axis=[1,2,3])
    loss = tf.reduce_mean(1.0 - (2 * intersection/ (union)))
    return loss

def computeDice(y_true, y_pred):
    y_true_f = y_true.flatten()>0.5
    y_pred_f = y_pred.flatten()>0.5
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-5) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-5)

def focal_loss1(labels, logits, gamma=1.0, alpha=1.0):    
    epsilon = 1e-9
    prediction = tf.nn.softmax(logits, axis=4)
    model_out = tf.add(prediction, epsilon)
    labels = tf.expand_dims(tf.cast(labels, dtype=tf.float32), axis=4)
    labels_expand = tf.concat([1-labels, labels], axis=4)
    ce = tf.multiply(labels_expand, -tf.log(model_out))
    weight = tf.multiply(labels_expand, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_mean(tf.reduce_max(fl, axis=[4]))
    return reduced_fl

def focal_loss(labels, logits, alpha=0.25, gamma=2.0):   
    predictions = tf.nn.sigmoid(logits)
    labels = tf.expand_dims(tf.cast(labels, dtype=tf.float32), axis=4)
    onehot_labels = tf.concat([1.-labels, labels], axis=4)
    predictions_pt = tf.where(tf.equal(onehot_labels, 1.0), predictions, 1.-predictions)
    # add small value to avoid 0
    epsilon = 1e-8
    alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
    alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
    losses = tf.reduce_sum(-alpha_t * tf.pow(1. - predictions_pt, gamma) * tf.log(predictions_pt+epsilon), axis=4)
    return tf.reduce_mean(losses)

def binary_cross_entropy(labels, logits):
    labels = tf.expand_dims(tf.cast(labels, dtype=tf.float32), axis=4)
    onehot_labels = tf.concat([1.-labels, labels], axis=4)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=onehot_labels, logits=logits))
    return loss