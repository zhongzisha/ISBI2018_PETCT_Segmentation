from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import SimpleITK as sitk
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import time
import itertools
import random
import sys
import glob
import pickle
from datetime import datetime

from tensorflow.python.saved_model import loader

from dltk.io.augmentation import add_gaussian_noise, flip, extract_class_balanced_example_array, elastic_transform
from dltk.io.preprocessing import whitening

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


def parse_args():

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--net_type", type=str, default='myunet_bn')
    parser.add_argument("--use_bn", type=int, default=1)
    parser.add_argument("--use_crf", type=int, default=0)
    parser.add_argument("--restore_ckpt_meta", type=str, default='')
    parser.add_argument("--feat_index", type=int, default=0) # 0: ct, 1: pt
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--test_filenames', type=str, default='testForTest0.csv')
    parser.add_argument('--norm_type', type=str, default='nonorm')
    
    
    args = parser.parse_args()

    return args

def main():
    
    args = parse_args()
    
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)
    
    subsets = ['train', 'val', 'test']
    image_mean = IMAGE_MEAN
    image_std  = IMAGE_STD
    image_mean_tensor = tf.constant(image_mean, dtype=tf.float32)
    image_std_tensor = tf.constant(image_std, dtype=tf.float32)
    
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    global_step = tf.Variable(0, trainable=False) 
    image_node = tf.placeholder(tf.float32, shape=[None, DEPTH, HEIGHT, WIDTH, 2])
    label_node = tf.placeholder(tf.int32, shape=[None, DEPTH, HEIGHT, WIDTH, 2])
    
    if args.norm_type == 'nonorm':
        image_node_new = image_node
    elif args.norm_type == 'globalnorm_mean':
        image_node_new = image_node - image_mean_tensor
    elif args.norm_type == 'globalnorm_meanstd':
        image_node_new = image_node - image_mean_tensor
        image_node_new /= image_std_tensor
    elif args.norm_type == 'instancenorm_mean':
        image_node_new = tf.map_fn(lambda frame: frame - tf.reduce_mean(frame, axis=[0,1,2], keep_dims=True), image_node)
    elif args.norm_type == 'instancenorm_meanstd':
        batch_mean, batch_var = tf.nn.moments(image_node, axes=[1,2,3], keep_dims=True)
        image_node_new = (image_node - batch_mean) / tf.sqrt(batch_var + 1e-6)
        
    if args.net_type == 'myunet3d_bn_crf':
        from myunet3d_basic import myunet3d_bn_crf
        net_output_ops = myunet3d_bn_crf(
            name='ct' if args.feat_index==0 else 'pt',
            inputs=image_node_new[...,args.feat_index][...,tf.newaxis],
            num_classes=NUM_CLASSES,
            phase_train=phase_train,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(value=0.1),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            bias_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            use_crf=args.use_crf,
            args=args)
    
    pred_op = net_output_ops['y_']
    prob_op = net_output_ops['y_prob']
    print('pred_op shape: ', pred_op.shape)
    print('prob_op shape: ', prob_op.shape)
    
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True 
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    restore_ckpt = args.restore_ckpt_meta[:-5]
    print(restore_ckpt)
    save_dir = restore_ckpt + '_results'
    
    saver1 = tf.train.import_meta_graph('{}.meta'.format(restore_ckpt))
    saver1.restore(sess, '{}'.format(restore_ckpt))

    if save_dir != '' and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    dices = []
    for idx, filename in enumerate(args.test_filenames.split(',')):
        test_filenames = pd.read_csv(
            DATA_ROOT + '/' + filename,
            dtype=object,
            keep_default_na=False,
            na_values=[]).as_matrix()
    
        dice_val = []
        for f in test_filenames:
            subject_id = f[0]
            img_fn = f[1]
            case_name = img_fn.split('/')[-1]
    
            ct_sitk = sitk.ReadImage(str(os.path.join(img_fn, 'InputCT_ROI.nii.gz')))
            ct = sitk.GetArrayFromImage(ct_sitk).astype((np.float32))
            pt_sitk = sitk.ReadImage(str(os.path.join(img_fn, 'InputPET_SUV_ROI.nii.gz')))
            pt = sitk.GetArrayFromImage(pt_sitk).astype((np.float32))
            lbl_ct = sitk.GetArrayFromImage(sitk.ReadImage(str(os.path.join(
                img_fn, 'GTV_Primary_ROI_CT{}.nii.gz'.format(GT_POSTFIX))))).astype(np.uint8)
            lbl_pt = sitk.GetArrayFromImage(sitk.ReadImage(str(os.path.join(
                img_fn, 'GTV_Primary_ROI_PET{}.nii.gz'.format(GT_POSTFIX))))).astype(np.uint8)
                
            ct[ct>200.] = 200.
            ct[ct<-500.] = -500.
            ct = 255*(ct+500)/(700.)
            
            pt[pt<0.01]=0.01
            pt[pt>20.]=20.
            pt = 255*(pt-0.01)/(19.99)
            
            image = np.concatenate([ct[...,np.newaxis],pt[...,np.newaxis]], axis=3)
            label = np.concatenate([lbl_ct[...,np.newaxis],lbl_pt[...,np.newaxis]], axis=3)
    
            pred, prob = sess.run([pred_op, prob_op], 
                            feed_dict={image_node: image[np.newaxis,...],
                                       label_node: label[np.newaxis,...],
                                       phase_train: False})
            dice_val_ = computeDice(label[...,args.feat_index], pred[0])
            dice_val.append(dice_val_)
            
            if save_dir != '':
                case_save_dir = '{}/{}'.format(save_dir, case_name)
                if not os.path.exists(case_save_dir):
                    os.makedirs(case_save_dir)
                
                new_sitk_ct = sitk.GetImageFromArray(pred[0].astype(np.int32))
                new_sitk_ct.CopyInformation(ct_sitk)
                sitk.WriteImage(new_sitk_ct, str('{}/crf0_pred_{}.nii.gz'.format(case_save_dir,
                                                                                   'ct' if args.feat_index==0 else 'pt')))
                new_sitk_ct = sitk.GetImageFromArray(prob[0][...,1].astype(np.float32))
                new_sitk_ct.CopyInformation(pt_sitk)
                sitk.WriteImage(new_sitk_ct, str('{}/crf0_prob_{}.nii.gz'.format(case_save_dir,
                                                                                   'ct' if args.feat_index==0 else 'pt')))
        dices.append(np.mean(np.array(dice_val)))
    print('FINAL_TEST {:.4f} {:.4f} {:.4f}'.format(dices[0],dices[1],dices[2]))

if __name__ == '__main__':
    main()
