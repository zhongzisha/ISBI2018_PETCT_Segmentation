from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import SimpleITK as sitk
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
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

# from crfrnn3d_tf_fortest import *

from myconfig import *

# supports numpy arrays and tensorflow tensors
def NDHWC_to_NCHWD(arr):
    try:
        return tf.compat.v1.transpose(arr, perm=(0,4,2,3,1))
    except:
        return arr.transpose((0,4,2,3,1))
def NCHWD_to_NDHWC(arr):
    try:
        return tf.compat.v1.transpose(arr, perm=(0,4,2,3,1))
    except:
        return arr.transpose((0,4,2,3,1))

def dataset_input_from_tfrecords(filenames, batch_size=1, num_epochs=1, shuffle=True):
    # filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    dataset = tf.compat.v1.data.TFRecordDataset(filenames)
    
    # Use `tf.compat.v1.parse_single_example()` to extract data from a `tf.compat.v1.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    def parser(record):
        keys_to_features = {
            'case_name': tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
            'image_raw': tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
            'label_raw': tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
        }
        parsed = tf.compat.v1.parse_single_example(record, keys_to_features)
        
        case_name = parsed['case_name']
        # Perform additional preprocessing on the parsed data.
        image = tf.compat.v1.decode_raw(parsed['image_raw'], tf.compat.v1.uint8)
        image = tf.compat.v1.reshape(image, [DEPTH, HEIGHT, WIDTH, 2]) 
        label = tf.compat.v1.decode_raw(parsed['label_raw'], tf.compat.v1.uint8)
        label = tf.compat.v1.reshape(label, [DEPTH, HEIGHT, WIDTH, 2]) 
        image = tf.compat.v1.cast(image, tf.compat.v1.float32)
        
        return image, label, case_name
    
    # Use `Dataset.map()` to build a pair of a feature dictionary and a label 
    # tensor for each example.
    dataset = dataset.map(parser)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=128)
    dataset = dataset.batch(batch_size)
    # dataset = dataset.apply(tf.compat.v1.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    
    features, labels, case_names = iterator.get_next()
    return features, labels, case_names

def dice_tf(logits, labels, threshold=0.5, axis=[1,2,3], smooth=1e-5):
    logits = tf.compat.v1.cast(logits, dtype=tf.compat.v1.float32)
    labels = tf.compat.v1.cast(labels, dtype=tf.compat.v1.float32)
    logits = tf.compat.v1.cast(logits > threshold, dtype=tf.compat.v1.float32)
    labels = tf.compat.v1.cast(labels > threshold, dtype=tf.compat.v1.float32)
    inse = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(logits, labels), axis=axis)
    l = tf.compat.v1.reduce_sum(logits, axis=axis)
    r = tf.compat.v1.reduce_sum(labels, axis=axis)
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    hard_dice = tf.compat.v1.reduce_mean(hard_dice)
    return hard_dice

def dice_loss1(logits, labels, threshold=0.5, axis=[1,2,3], smooth=1e-5):
    # exponential_map = tf.compat.v1.exp(logits)
    # sum_exp = tf.compat.v1.reduce_sum(exponential_map, 4, keep_dims=True)
    # tensor_sum_exp = tf.compat.v1.tile(sum_exp, tf.compat.v1.stack([1, 1, 1, 1, tf.compat.v1.shape(logits)[4]]))
    # prediction = tf.compat.v1.div(exponential_map,tensor_sum_exp)
    prediction = tf.compat.v1.nn.softmax(logits, axis=4)
    
    labels = tf.compat.v1.expand_dims(tf.compat.v1.cast(labels, dtype=tf.compat.v1.float32), axis=4)
    labels_expand = tf.compat.v1.concat([1-labels, labels], axis=4)
    eps = 1e-5
    intersection = tf.compat.v1.reduce_sum(prediction * labels_expand, axis=[1,2,3,4])
    union =  eps + tf.compat.v1.reduce_sum(prediction, axis=[1,2,3,4]) + tf.compat.v1.reduce_sum(labels_expand, axis=[1,2,3,4])
    loss = tf.compat.v1.reduce_mean(1.0 - (2 * intersection/ (union)))
    return loss

def dice_loss(logits, labels, threshold=0.5, axis=[1,2,3], smooth=1e-5):
    labels = tf.compat.v1.cast(labels, tf.compat.v1.float32)
    prediction = tf.compat.v1.nn.softmax(logits, axis=4)
    eps = 1e-5
    intersection = tf.compat.v1.reduce_sum(prediction[...,1] * labels, axis=[1,2,3])
    union =  eps + tf.compat.v1.reduce_sum(prediction[...,1], axis=[1,2,3]) + tf.compat.v1.reduce_sum(labels, axis=[1,2,3])
    loss = tf.compat.v1.reduce_mean(1.0 - (2 * intersection/ (union)))
    return loss

def computeDice(y_true, y_pred):
    y_true_f = y_true.flatten()>0.5
    y_pred_f = y_pred.flatten()>0.5
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-5) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-5)

def focal_loss1(labels, logits, gamma=1.0, alpha=1.0):    
    epsilon = 1e-9
    prediction = tf.compat.v1.nn.softmax(logits, axis=4)
    model_out = tf.compat.v1.add(prediction, epsilon)
    labels = tf.compat.v1.expand_dims(tf.compat.v1.cast(labels, dtype=tf.compat.v1.float32), axis=4)
    labels_expand = tf.compat.v1.concat([1-labels, labels], axis=4)
    ce = tf.compat.v1.multiply(labels_expand, -tf.compat.v1.log(model_out))
    weight = tf.compat.v1.multiply(labels_expand, tf.compat.v1.pow(tf.compat.v1.subtract(1., model_out), gamma))
    fl = tf.compat.v1.multiply(alpha, tf.compat.v1.multiply(weight, ce))
    reduced_fl = tf.compat.v1.reduce_mean(tf.compat.v1.reduce_max(fl, axis=[4]))
    return reduced_fl

def focal_loss(labels, logits, alpha=0.25, gamma=2.0):   
    predictions = tf.compat.v1.nn.sigmoid(logits)
    labels = tf.compat.v1.expand_dims(tf.compat.v1.cast(labels, dtype=tf.compat.v1.float32), axis=4)
    onehot_labels = tf.compat.v1.concat([1.-labels, labels], axis=4)
    predictions_pt = tf.compat.v1.where(tf.compat.v1.equal(onehot_labels, 1.0), predictions, 1.-predictions)
    # add small value to avoid 0
    epsilon = 1e-8
    alpha_t = tf.compat.v1.scalar_mul(alpha, tf.compat.v1.ones_like(onehot_labels, dtype=tf.compat.v1.float32))
    alpha_t = tf.compat.v1.where(tf.compat.v1.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
    losses = tf.compat.v1.reduce_sum(-alpha_t * tf.compat.v1.pow(1. - predictions_pt, gamma) * tf.compat.v1.log(predictions_pt+epsilon), axis=4)
    return tf.compat.v1.reduce_mean(losses)

def binary_cross_entropy(labels, logits):
    labels = tf.compat.v1.expand_dims(tf.compat.v1.cast(labels, dtype=tf.compat.v1.float32), axis=4)
    onehot_labels = tf.compat.v1.concat([1.-labels, labels], axis=4)
    loss = tf.compat.v1.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(labels=onehot_labels, logits=logits))
    return loss


def parse_args():

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--net_type", type=str, default='myunet_bn')
    parser.add_argument("--use_bn", type=int, default=1)
    parser.add_argument("--use_crf", type=int, default=0)
    parser.add_argument("--restore_ckpt_meta", type=str, default='')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--test_filenames', type=str, default='testForTest0.csv')
    parser.add_argument('--norm_type', type=str, default='nonorm')
    
    
    args = parser.parse_args()

    return args

def main():
    
    args = parse_args()
    
    np.random.seed(args.random_seed)
    tf.compat.v1.set_random_seed(args.random_seed)
    
    subsets = ['train', 'val', 'test']
    image_mean = IMAGE_MEAN
    image_std  = IMAGE_STD
    image_mean_tensor = tf.compat.v1.constant(image_mean, dtype=tf.compat.v1.float32)
    image_std_tensor = tf.compat.v1.constant(image_std, dtype=tf.compat.v1.float32)
    
    phase_train = tf.compat.v1.placeholder(tf.compat.v1.bool, name='phase_train')
    global_step = tf.compat.v1.Variable(0, trainable=False) 
    image_node = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, DEPTH, HEIGHT, WIDTH, 2])
    label_node = tf.compat.v1.placeholder(tf.compat.v1.int32, shape=[None, DEPTH, HEIGHT, WIDTH, 2])
    
    if args.norm_type == 'nonorm':
        image_node_new = image_node
    elif args.norm_type == 'globalnorm_mean':
        image_node_new = image_node - image_mean_tensor
    elif args.norm_type == 'globalnorm_meanstd':
        image_node_new = image_node - image_mean_tensor
        image_node_new /= image_std_tensor
    elif args.norm_type == 'instancenorm_mean':
        image_node_new = tf.compat.v1.map_fn(lambda frame: frame - tf.compat.v1.reduce_mean(frame, axis=[0,1,2], keep_dims=True), image_node)
    elif args.norm_type == 'instancenorm_meanstd':
        batch_mean, batch_var = tf.compat.v1.nn.moments(image_node, axes=[1,2,3], keep_dims=True)
        image_node_new = (image_node - batch_mean) / tf.compat.v1.sqrt(batch_var + 1e-6)
        
    if args.net_type == 'myfusionunet2_bn':
        from myunet3d_basic import myfusionunet2_bn
        net_output_ops = myfusionunet2_bn(
            name='ctpt',
            inputs=image_node_new,
            num_classes=NUM_CLASSES,
            phase_train=phase_train,
            use_bias=True,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.compat.v1.constant_initializer(value=0.1),
            kernel_regularizer=tf.compat.v1.keras.regularizers.l2(1e-4),
            bias_regularizer=tf.compat.v1.keras.regularizers.l2(1e-4),
            use_crf=args.use_crf,
            args=args)

    pred_ct_op = net_output_ops['y_ct']
    pred_pt_op = net_output_ops['y_pt']
    prob_ct_op = net_output_ops['y_prob_ct']
    prob_pt_op = net_output_ops['y_prob_pt']
    
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=0)
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True 
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())
    
    restore_ckpt = args.restore_ckpt_meta[:-5]
    print(restore_ckpt)
    save_dir = restore_ckpt + '_results'
    
    saver1 = tf.compat.v1.train.import_meta_graph('{}.meta'.format(restore_ckpt))
    saver1.restore(sess, '{}'.format(restore_ckpt))

    if save_dir != '' and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, filename in enumerate(args.test_filenames.split(',')):
        test_filenames = pd.read_csv(
            DATA_ROOT + '/' + filename,
            dtype=object,
            keep_default_na=False,
            na_values=[]).values
    
        dice_val_cts = []
        dice_val_pts = []
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
    
            pred_ct, pred_pt, \
            prob_ct, prob_pt = sess.run([pred_ct_op, pred_pt_op, 
                                         prob_ct_op, prob_pt_op], 
                            feed_dict={image_node: image[np.newaxis,...],
                                       label_node: label[np.newaxis,...],
                                       phase_train: False})
            dice_val_ct = computeDice(label[...,0], pred_ct[0])
            dice_val_pt = computeDice(label[...,1], pred_pt[0])
            dice_val_cts.append(dice_val_ct)
            dice_val_pts.append(dice_val_pt)
            
            if save_dir != '':
                case_save_dir = '{}/{}'.format(save_dir, case_name)
                if not os.path.exists(case_save_dir):
                    os.makedirs(case_save_dir)
                
                new_sitk_ct = sitk.GetImageFromArray(pred_ct[0].astype(np.int32))
                new_sitk_ct.CopyInformation(ct_sitk)
                sitk.WriteImage(new_sitk_ct, str('{}/crf0_fusion2_pred_ct.nii.gz'.format(case_save_dir)))
                new_sitk_ct = sitk.GetImageFromArray(prob_ct[0][...,1].astype(np.float32))
                new_sitk_ct.CopyInformation(pt_sitk)
                sitk.WriteImage(new_sitk_ct, str('{}/crf0_fusion2_prob_ct.nii.gz'.format(case_save_dir)))
                new_sitk_ct = sitk.GetImageFromArray(pred_pt[0].astype(np.int32))
                new_sitk_ct.CopyInformation(ct_sitk)
                sitk.WriteImage(new_sitk_ct, str('{}/crf0_fusion2_pred_pt.nii.gz'.format(case_save_dir)))
                new_sitk_ct = sitk.GetImageFromArray(prob_pt[0][...,1].astype(np.float32))
                new_sitk_ct.CopyInformation(pt_sitk)
                sitk.WriteImage(new_sitk_ct, str('{}/crf0_fusion2_prob_pt.nii.gz'.format(case_save_dir)))

        print(dice_val_cts, np.mean(np.array(dice_val_cts)))
        print(dice_val_pts, np.mean(np.array(dice_val_pts)))

if __name__ == '__main__':
    main()