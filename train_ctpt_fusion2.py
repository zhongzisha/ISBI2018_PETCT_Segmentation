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

# from crfrnn3d_tf_fortest import *

from myconfig import *
from myutils import *

def parse_args():

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--action", type=str, default='train')
    parser.add_argument("--net_type", type=str, default='myunet_bn')
    parser.add_argument("--use_bn", type=int, default=1)
    parser.add_argument("--use_crf", type=int, default=0)
    parser.add_argument("--restore_ckpt", type=str, default='')
    parser.add_argument("--log_dir", type=str, default='logs3d_nonorm')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--lr_policy', type=str, default='constant')
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--max_steps', type=int, default=2000000)
    parser.add_argument("--opt_type", type=str, default='gd')
    parser.add_argument('--reg_coef', type=float, default=1.0)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--restore_from_saved_model', type=str, default='')
    parser.add_argument('--save_prob_dir', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--test_filenames', type=str, default='testForTest0.csv')
    
    parser.add_argument('--norm_type', type=str, default='nonorm')
    parser.add_argument('--loss_type', type=str, default='ce') # ce or dice
    
    parser.add_argument('--with_aug', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=21)
    parser.add_argument('--decay_epochs', type=int, default=5)
    
    args = parser.parse_args()

    return args

def main():
    
    args = parse_args()
    
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)
    
    use_crf = args.use_crf
    print('Use CRF: ', use_crf)
    BATCH_SIZE = args.batch_size
    
    print('Setting up...')
    train_filenames = pd.read_csv(
        TRAIN_FILENAME,
        dtype=object,
        keep_default_na=False,
        na_values=[]).as_matrix()
    train0_filenames = pd.read_csv(
        TRAIN0_FILENAME,
        dtype=object,
        keep_default_na=False,
        na_values=[]).as_matrix()
    val_filenames = pd.read_csv(
        VAL_FILENAME,
        dtype=object,
        keep_default_na=False,
        na_values=[]).as_matrix()
    test_filenames = pd.read_csv(
        TEST_FILENAME,
        dtype=object,
        keep_default_na=False,
        na_values=[]).as_matrix()
    subsets = ['train', 'val', 'test']
    image_mean = IMAGE_MEAN
    image_std  = IMAGE_STD
    image_mean_tensor = tf.constant(image_mean, dtype=tf.float32)
    image_std_tensor = tf.constant(image_std, dtype=tf.float32)
    
    print('num_train: ', NUM_TRAIN_SAMPLES)
    print('num_train0: ', NUM_TRAIN0_SAMPLES)
    #if args.with_aug==1:
    STEPS_ONE_EPOCH = int(NUM_TRAIN_SAMPLES / BATCH_SIZE)
    #else:
    #    STEPS_ONE_EPOCH = int(NUM_TRAIN0_SAMPLES / BATCH_SIZE)
    max_steps = STEPS_ONE_EPOCH * args.num_epochs
    DECAY_STEPS = STEPS_ONE_EPOCH * args.decay_epochs
    print('steps one epoch: ', STEPS_ONE_EPOCH)
    print('decay steps: ', DECAY_STEPS)
    
    if args.with_aug==1:
        train_filenames_tfrecords = [f[1]+'/data2.tfrecords' for f in train_filenames]
    else:
        train_filenames_tfrecords = [f[1]+'/data2.tfrecords' for f in train0_filenames]
    val_filenames_tfrecords = [f[1]+'/data2.tfrecords' for f in val_filenames]
    train_images, train_labels, train_case_names = dataset_input_from_tfrecords(train_filenames_tfrecords, 
                                                                                batch_size=BATCH_SIZE, 
                                                                                num_epochs=50000, 
                                                                                shuffle=True)
    val_images, val_labels, val_case_names = dataset_input_from_tfrecords(val_filenames_tfrecords, 
                                                                          batch_size=BATCH_SIZE, 
                                                                          num_epochs=50000, 
                                                                          shuffle=False)
    
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
        
    if args.net_type == 'myfusionunet2_bn':
        from myunet3d_basic import myfusionunet2_bn
        net_output_ops = myfusionunet2_bn(
            name='ctpt',
            inputs=image_node_new,
            num_classes=NUM_CLASSES,
            phase_train=phase_train,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(value=0.1),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            bias_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            use_crf=args.use_crf,
            args=args)

    pred_ct_op = net_output_ops['y_ct']
    pred_pt_op = net_output_ops['y_pt']

    # 2. set up a loss function    # regularization loss
    reg_constant = tf.constant(args.reg_coef, dtype=tf.float32)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if len(reg_losses)!=0:
        loss_reg_op = tf.add_n(reg_losses)
    else:
        loss_reg_op = tf.constant(0, dtype=tf.float32)
    tf.summary.scalar('loss_reg_ctpt2', loss_reg_op)
    # 2. set up a loss function
    if args.loss_type == 'ce':
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=net_output_ops['logits_ct'],
            labels=label_node[...,0])
        ce2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=net_output_ops['logits_pt'],
            labels=label_node[...,1])
        loss_op = tf.reduce_mean(ce) + tf.reduce_mean(ce2)
    elif args.loss_type == 'dice':
        loss_op = dice_loss(labels=label_node[...,0], logits=net_output_ops['logits_ct']) + \
                  dice_loss(labels=label_node[...,1], logits=net_output_ops['logits_pt'])
    elif args.loss_type == 'focal':
        loss_op = focal_loss(labels=label_node[...,0], logits=net_output_ops['logits_ct']) + \
                  focal_loss(labels=label_node[...,1], logits=net_output_ops['logits_pt'])
    elif args.loss_type == 'bce':
        loss_op = binary_cross_entropy(labels=label_node[...,0], logits=net_output_ops['logits_ct']) + \
                  binary_cross_entropy(labels=label_node[...,1], logits=net_output_ops['logits_pt'])
    tf.summary.scalar('loss_ctpt2', loss_op)
    total_loss_op = loss_op + tf.multiply(reg_constant, loss_reg_op)
    tf.summary.scalar('total_loss', total_loss_op)
        
    if args.lr_policy == 'constant':
        lr_op = args.base_lr
    elif args.lr_policy == 'piecewise':
        #### piecewise learning rate   
        boundaries = [i*DECAY_STEPS for i in [1,2,3,4,5]]
        staged_lr = [args.base_lr * x for x in [1, 0.5, 0.25, 0.125, 0.0625, 0.03125]]
         
        lr_op = tf.train.piecewise_constant(global_step,
                                            boundaries, 
                                            staged_lr)
    elif args.lr_policy == 'expdecay':
        learning_rate_decay_factor=0.5
        decay_steps = 1000
        
        # Decay the learning rate exponentially based on the number of steps.
        lr_op = tf.train.exponential_decay(args.base_lr,
                                           global_step,
                                           decay_steps,
                                           learning_rate_decay_factor,
                                           staircase=True)
        
    tf.summary.scalar('learning_rate_ctpt2', lr_op)

    if args.opt_type == 'gd':
        optimiser = tf.train.GradientDescentOptimizer(learning_rate=lr_op)
    elif args.opt_type == 'momentum':
        optimiser = tf.train.MomentumOptimizer(learning_rate=lr_op, momentum=0.9)
    elif args.opt_type == 'adam':
        optimiser = tf.train.AdamOptimizer(learning_rate=lr_op)
    elif args.opt_type == 'adadelta':
        optimiser = tf.train.AdadeltaOptimizer(learning_rate=lr_op)
    elif args.opt_type == 'adagrad':
        optimiser = tf.train.AdagradOptimizer(learning_rate=lr_op)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimiser.minimize(total_loss_op, global_step=global_step)


    log_postfix = '{}_{}x{}x{}_aug{}_fctpt2_{}_opt{}_lr{}{}_b{}_loss{}_crf{}_reg{}_rs{}'.format(args.net_type,HEIGHT,WIDTH,DEPTH,
                                                                                                args.with_aug,
                                                               args.norm_type,
                                                 args.opt_type, 
                                                 args.lr_policy,
                                                 args.base_lr, 
                                                 args.batch_size,
                                                 args.loss_type,
                                                 args.use_crf,
                                                 args.reg_coef,
                                                 args.random_seed)
    log_dir = '{}_{}'.format(args.log_dir, log_postfix)
    
    
    if args.restore_ckpt == "" and args.action =='train':
        if tf.gfile.Exists(log_dir):
            tf.gfile.DeleteRecursively(log_dir)
        tf.gfile.MakeDirs(log_dir)
        
    # set up log file
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    log_filename = '{}/log_{}.txt'.format(log_dir, current_time)
    cmd_filename = '{}/log_curve_{}.sh'.format(log_dir, current_time)
    log_file_handle = open(log_filename, 'w')
    with open(cmd_filename, 'w') as cmd_file_handle:
        cmd_file_handle.write('python plot_learning_curves_dual.py {} {} {}\n'.format(log_dir,
                                                                                      log_postfix,
                                                                                      log_filename))

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True 
    sess = tf.Session(config=config)
    
    if args.action == 'train':
        summary_op = tf.summary.merge_all() 
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 0
    if args.step>0:
        step = args.step
    while not coord.should_stop():
        
        if step % STEPS_ONE_EPOCH == 0:
            # val one epoch
            # for subi, filenames in enumerate([train0_filenames, val_filenames, test_filenames]):
            for subi, filenames in enumerate([train0_filenames, 
                                              np.concatenate([val_filenames, test_filenames], axis=0)]):
                dice_val = {'ct':[], 'pt':[]}
                for f in filenames:
                    subject_id = f[0]
                    img_fn = f[1]
                    case_name = img_fn.split('/')[-1]
                
                    # Read the image nii with sitk and keep the pointer to the sitk.Image
                    # of an input
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

#                     if image_mean != None:
#                        image -= np.reshape(image_mean, [1,1,1,2])
#                     if image_std != None:
#                        image /= np.reshape(image_std, [1,1,1,2])
                         
                    pred_ct, pred_pt = sess.run([pred_ct_op, pred_pt_op], 
                                    feed_dict={image_node: image[np.newaxis,...],
                                               label_node: label[np.newaxis,...],
                                               phase_train: False})
                    dice_val_ct_ = computeDice(label[...,0], pred_ct[0])
                    dice_val_pt_ = computeDice(label[...,1], pred_pt[0])
                    dice_val['ct'].append(dice_val_ct_)
                    dice_val['pt'].append(dice_val_pt_)

                    log_file_handle.write('{} {} {}\n'.format(case_name, dice_val_ct_, dice_val_pt_))
                
                log_file_handle.write('{} {} Mean ctpt2 Dice(Before CRF): {} {}\n'.format(step, 
                                                                                    subsets[subi], 
                                                                                    np.mean(np.array(dice_val['ct'])),
                                                                                    np.mean(np.array(dice_val['pt']))))
            
        
        if step % STEPS_ONE_EPOCH == 0:
            checkpoint_path = os.path.join(log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)             
        
        # train one epoch
        start_time = time.time()
        image_batch, label_batch, case_name_batch = sess.run([train_images, train_labels, train_case_names])

        summary, _, loss_reg_value, \
        total_loss_value, loss_value \
        = sess.run([summary_op, train_op, loss_reg_op,
                    total_loss_op, loss_op], 
                   feed_dict={image_node: image_batch,
                              label_node: label_batch,
                              phase_train: True})
        
        duration = time.time() - start_time

        if step % STEPS_ONE_EPOCH == 0:
            summary_writer.add_summary(summary, global_step=step)

        step += 1
        if step==max_steps:
            break
        
    log_file_handle.close()
    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    main()