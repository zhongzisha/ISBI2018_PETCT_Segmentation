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
    parser.add_argument("--feat_index", type=int, default=0) # 0: ct, 1: pt
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--lr_policy', type=str, default='constant')
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument("--opt_type", type=str, default='gd')
    parser.add_argument('--reg_coef', type=float, default=1.0)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--restore_from_saved_model', type=str, default='')
    parser.add_argument('--save_prob_dir', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--test_filenames', type=str, default='testForTest0.csv')
    
    parser.add_argument('--sw_weight', type=float, default=1.0)
    parser.add_argument('--bw_weight', type=float, default=1.0)
    parser.add_argument('--cm_weight', type=float, default=1.0)
    parser.add_argument('--theta_alpha', type=float, default=1.0)
    parser.add_argument('--theta_beta', type=float, default=10.0)
    parser.add_argument('--theta_gamma', type=float, default=100.0)
    parser.add_argument('--num_iterations', type=int, default=5)
    parser.add_argument('--crf_lr_scale', type=float, default=100.0)
    
    parser.add_argument('--norm_type', type=str, default='nonorm')
    parser.add_argument('--loss_type', type=str, default='ce') # ce or dice
    
    
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
    STEPS_ONE_EPOCH = int(NUM_TRAIN_SAMPLES / BATCH_SIZE)
    max_steps = STEPS_ONE_EPOCH * 21
    DECAY_STEPS = STEPS_ONE_EPOCH*5
    print('steps one epoch: ', STEPS_ONE_EPOCH)
    print('decay steps: ', DECAY_STEPS)
    
    train_filenames_tfrecords = [f[1]+'/data2.tfrecords' for f in train_filenames]
    val_filenames_tfrecords = [f[1]+'/data2.tfrecords' for f in val_filenames]
    train_images, train_labels, train_case_names = dataset_input_from_tfrecords(train_filenames_tfrecords, 
                                                                                batch_size=BATCH_SIZE, 
                                                                                num_epochs=5000, 
                                                                                shuffle=True)
    val_images, val_labels, val_case_names = dataset_input_from_tfrecords(val_filenames_tfrecords, 
                                                                          batch_size=BATCH_SIZE, 
                                                                          num_epochs=5000, 
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
    elif args.net_type == 'myunet3d_crf':
        from myunet3d_basic import myunet3d_crf
        net_output_ops = myunet3d_crf(
            name='ct' if args.feat_index==0 else 'pt',
            inputs=image_node_new[...,args.feat_index][...,tf.newaxis],
            num_classes=NUM_CLASSES,
            phase_train=phase_train,
            use_bias=True,
            kernel_initializer=tf.glorot_uniform_initializer(),#tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(value=0.1),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            bias_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            use_crf=args.use_crf,
            args=args)
    elif args.net_type == 'myunet3d_isbi2018':
        from myunet3d_basic import myunet3d_isbi2018
        net_output_ops = myunet3d_isbi2018(
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
    elif args.net_type == 'myunet3d_isbi2018_2':
        from myunet3d_basic import myunet3d_isbi2018_2
        net_output_ops = myunet3d_isbi2018_2(
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
    elif args.net_type == 'myunet3d_improved':
        from myunet3d_improved import myunet3d_improved
        net_output_ops = myunet3d_improved(
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
    elif args.net_type == 'myunet3d_improved_usingresize':
        from myunet3d_improved_usingresize import myunet3d_improved_usingresize
        net_output_ops = myunet3d_improved_usingresize(
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
        
    modes=['ct','pt']
    image_node_shaped = tf.reshape(image_node[...,args.feat_index], 
                                   [-1, DEPTH, HEIGHT*WIDTH, 1])
    label_node_shaped = tf.cast(tf.reshape(label_node[...,args.feat_index],
                                           [-1, DEPTH, HEIGHT*WIDTH, 1]), tf.float32)
    for bi in range(BATCH_SIZE):
        tf.summary.image('input_{}_{}'.format(modes[args.feat_index],str(bi)), 
                         image_node_shaped[bi][tf.newaxis,...], 1)
        tf.summary.image('label_{}_{}'.format(modes[args.feat_index],str(bi)), 
                         label_node_shaped[bi][tf.newaxis,...], 1)

    prob_flat = tf.unstack(net_output_ops['y_prob'], num=2, axis=4) # [-1, 32, 32, 32]
    prob_flat = [tf.reshape(v,[-1,DEPTH, HEIGHT*WIDTH]) for v in prob_flat] #[-1,32,32*32] 
    prob_flat = tf.concat(prob_flat, axis=1) #[-1, 32*16, 32*32]
    prob_flat = tf.cast(tf.reshape(prob_flat, [-1, 2*DEPTH, HEIGHT*WIDTH, 1]), tf.float32)
    pred_flat = tf.cast(tf.reshape(net_output_ops['y_'],[-1,DEPTH, HEIGHT*WIDTH,1]), tf.float32)    
    
    for bi in range(BATCH_SIZE):
        tf.summary.image('prob_{}_{}'.format(modes[args.feat_index],str(bi)), 
                         prob_flat[bi][tf.newaxis,...], 1)
        tf.summary.image('pred_{}_{}'.format(modes[args.feat_index],str(bi)), 
                         pred_flat[bi][tf.newaxis,...], 1)
    
    pred_op = net_output_ops['y_']
    prob_op = net_output_ops['y_prob']
    print('pred_op shape: ', pred_op.shape)
    print('prob_op shape: ', prob_op.shape)
    dice_op = dice_tf(labels=label_node[...,args.feat_index], logits=prob_op[...,1])
    
    if args.use_crf:
        prob_crf_op = net_output_ops['y_prob_crf']
        prob_crf_flat = tf.unstack(prob_crf_op, num=2, axis=4) # [-1, 32, 32, 32]
        prob_crf_flat = [tf.reshape(v,[-1,DEPTH, HEIGHT*WIDTH]) for v in prob_crf_flat] #[-1,32,32*32] 
        prob_crf_flat = tf.concat(prob_crf_flat, axis=1) #[-1, 32*16, 32*32]
        prob_crf_flat = tf.cast(tf.reshape(prob_crf_flat, [-1, 2*DEPTH, HEIGHT*WIDTH, 1]), tf.float32)
        
        pred_crf_flat = tf.cast(tf.reshape(net_output_ops['y_crf'],[-1,DEPTH, HEIGHT*WIDTH,1]), tf.float32)
        
        for bi in range(BATCH_SIZE):
            tf.summary.image('prob_crf_{}_{}'.format(modes[args.feat_index],str(bi)), 
                             prob_crf_flat[bi][tf.newaxis,...], 1)
            tf.summary.image('pred_crf_{}_{}'.format(modes[args.feat_index],str(bi)), 
                             pred_crf_flat[bi][tf.newaxis,...], 1)
        
        pred_crf_op = net_output_ops['y_crf']
        dice_crf_op = dice_tf(labels=label_node[...,args.feat_index], logits=prob_crf_op[...,1])

    # 2. set up a loss function    # regularization loss
    reg_constant = tf.constant(args.reg_coef, dtype=tf.float32)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if len(reg_losses)!=0:
        loss_reg_op = tf.add_n(reg_losses)
    else:
        loss_reg_op = tf.constant(0, dtype=tf.float32)
    tf.summary.scalar('loss_reg_{}'.format(modes[args.feat_index]), loss_reg_op)
    # 2. set up a loss function
    if args.loss_type == 'ce':
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=net_output_ops['logits'],
            labels=label_node[...,args.feat_index])
        loss_op = tf.reduce_mean(ce)
    elif args.loss_type == 'dice':
        loss_op = dice_loss(labels=label_node[...,args.feat_index], logits=net_output_ops['logits'])
    elif args.loss_type == 'focal':
        loss_op = focal_loss(labels=label_node[...,args.feat_index], logits=net_output_ops['logits'])
    elif args.loss_type == 'bce':
        loss_op = binary_cross_entropy(labels=label_node[...,args.feat_index], logits=net_output_ops['logits'])
    tf.summary.scalar('loss_{}'.format(modes[args.feat_index]), loss_op)
    total_loss_op = loss_op + tf.multiply(reg_constant, loss_reg_op)
    tf.summary.scalar('total_loss_{}'.format(modes[args.feat_index]), total_loss_op)
    
    if args.use_crf:
        if args.loss_type == 'ce':
            ce_crf = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=net_output_ops['logits_crf'],
                labels=label_node[...,args.feat_index])
            loss_crf_op = tf.reduce_mean(ce_crf)
        elif args.loss_type == 'dice':
            loss_crf_op = dice_loss(labels=label_node[...,args.feat_index], logits=net_output_ops['logits_crf'])
        elif args.loss_type == 'focal':
            loss_crf_op = focal_loss(labels=label_node[...,args.feat_index], logits=net_output_ops['logits_crf'])
        elif args.loss_type == 'bce':
            loss_crf_op = binary_cross_entropy(labels=label_node[...,args.feat_index], logits=net_output_ops['logits_crf'])
        tf.summary.scalar('loss_crf_{}'.format(modes[args.feat_index]), loss_crf_op)
        total_loss_crf_op = loss_crf_op + tf.multiply(reg_constant, loss_reg_op)
        tf.summary.scalar('total_loss_{}_crf'.format(modes[args.feat_index]), total_loss_crf_op)
        
    if args.use_crf:
        crf_sw_op = tf.get_default_graph().get_tensor_by_name(modes[args.feat_index]+'/crf/spatial_weights:0')
        crf_bw_op = tf.get_default_graph().get_tensor_by_name(modes[args.feat_index]+'/crf/bilateral_weights:0')
        crf_cm_op = tf.get_default_graph().get_tensor_by_name(modes[args.feat_index]+'/crf/compatibility_matrix:0')
        
        noncrf_vars = []
        crf_vars = []
        for v in tf.trainable_variables():
            if 'crf' in v.name:
                crf_vars.append(v)
            else:
                noncrf_vars.append(v)
        
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
        
    tf.summary.scalar('learning_rate_{}'.format(modes[args.feat_index]), lr_op)

    if args.opt_type == 'gd':
        optimiser = tf.train.GradientDescentOptimizer(learning_rate=lr_op)
    elif args.opt_type == 'gd_diff_lr':
        optimiser = tf.train.GradientDescentOptimizer(learning_rate=lr_op)
        optimiser_crf = tf.train.GradientDescentOptimizer(learning_rate=lr_op*args.crf_lr_scale)
    elif args.opt_type == 'adam_diff_lr':
        optimiser = tf.train.AdamOptimizer(learning_rate=lr_op)
        optimiser_crf = tf.train.AdamOptimizer(learning_rate=lr_op*args.crf_lr_scale)
    elif args.opt_type == 'adam_gd_diff_lr':
        optimiser = tf.train.AdamOptimizer(learning_rate=lr_op)
        optimiser_crf = tf.train.GradientDescentOptimizer(learning_rate=lr_op*args.crf_lr_scale)
    elif args.opt_type == 'momentum':
        optimiser = tf.train.MomentumOptimizer(learning_rate=lr_op, momentum=0.9)
    elif args.opt_type == 'adam':
        optimiser = tf.train.AdamOptimizer(learning_rate=lr_op)
    elif args.opt_type == 'adadelta':
        optimiser = tf.train.AdadeltaOptimizer(learning_rate=lr_op)
    elif args.opt_type == 'adagrad':
        optimiser = tf.train.AdagradOptimizer(learning_rate=lr_op)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if 'diff_lr' in args.opt_type:
        grads = tf.gradients(total_loss_crf_op, noncrf_vars + crf_vars)
        grads1 = grads[:len(noncrf_vars)]
        grads2 = grads[len(noncrf_vars):]
        with tf.control_dependencies(update_ops):
            train_op1 = optimiser.apply_gradients(zip(grads1, noncrf_vars))
        train_op2 = optimiser_crf.apply_gradients(zip(grads2, crf_vars))
        train_op = tf.group(train_op1, train_op2)
    else:
        with tf.control_dependencies(update_ops):
            if args.use_crf:
                train_op = optimiser.minimize(total_loss_crf_op, global_step=global_step)
            else:
                train_op = optimiser.minimize(total_loss_op, global_step=global_step)


    log_postfix = '{}_{}x{}x{}_f{}_{}_opt{}_lr{}{}_b{}_loss{}_crf{}_reg{}_rs{}'.format(args.net_type,HEIGHT,WIDTH,DEPTH,
                                                               modes[args.feat_index],
                                                               args.norm_type,
                                                 args.opt_type, 
                                                 args.lr_policy,
                                                 args.base_lr, 
                                                 args.batch_size,
                                                 args.loss_type,
                                                 args.use_crf,
                                                 args.reg_coef,
                                                 args.random_seed)
    if args.use_crf:
        log_postfix = '{}/{}_{}_{}_{}_{}_{}_lr_scale_{}'.format(log_postfix,
                                                            args.sw_weight,
                                                            args.bw_weight,
                                                            args.cm_weight,
                                                            args.theta_alpha,
                                                            args.theta_beta,
                                                            args.theta_gamma, 
                                                            args.crf_lr_scale)
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
        cmd_file_handle.write('python plot_learning_curves_single.py {} {} {}\n'.format(log_dir,
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
    
    if args.restore_ckpt != "" and args.action == 'train':
        variables = tf.global_variables()
        restore_vars = []
        restore_var_names = [line.strip() for line in \
                             open('trainable_variables_{}_only.txt'.format(modes[args.feat_index]),'r').readlines()]
        for v in variables:
            found = False
            for name in restore_var_names:
                if name in v.name:
                    found = True
                    break
            if found:
                restore_vars.append(v)
        
        for v in restore_vars:
            print(v.name)
        
        restore_saver = tf.train.Saver(restore_vars)
        restore_saver.restore(sess, args.restore_ckpt)
        
    if args.restore_ckpt != "" and args.action == 'test':
        saver1 = tf.train.import_meta_graph('{}.meta'.format(args.restore_ckpt))
        saver1.restore(sess, '{}'.format(args.restore_ckpt))
    
    if args.restore_from_saved_model != "":
        
        pretrained_variables = {}
        with tf.Graph().as_default():
            # restore_from = '/media/ubuntu/working/petct_cnn/logs_dltk_ctpt_1e-4_v1(dont delete)/ct_only/1513006564'
            with tf.Session() as sess1:
                graph_old = loader.load(sess1, ['serve'], args.restore_from_saved_model)
                for v in tf.global_variables():
                    pretrained_variables[v.name] = v.eval()
                    
        count = 0
        for v in tf.global_variables():
            for name in pretrained_variables.keys():
                if modes[args.feat_index] + '/' + name == v.name:
                    print('{}: {} ==> {}'.format(count, name, v.name))
                    sess.run(v.assign(pretrained_variables[name]))
                    count += 1
                    break
        print(count, ' variables copied from previous model!')
        
    if args.action == 'test': 
        if args.save_prob_dir != '' and not os.path.exists(args.save_prob_dir):
            os.makedirs(args.save_prob_dir)
        if args.save_dir != '' and not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        
        for idx, filename in enumerate(args.test_filenames.split(',')):
            temp = pd.read_csv(
                DATA_ROOT + '/' + filename,
                dtype=object,
                keep_default_na=False,
                na_values=[]).as_matrix()
            if idx==0: 
                test_filenames = temp
            else:
                test_filenames = np.concatenate([test_filenames, temp], axis=0)
        
        dice_val = []
        if args.use_crf:
            dice_val_crf = []
        alldata = {}
        for f in test_filenames:
        # for f in np.concatenate([val_filenames, test_filenames], axis=0):
            subject_id = f[0]
            img_fn = f[1]
            case_name = img_fn.split('/')[-1]
            # if '2141' in case_name:
            #     continue
        
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
            
#             if image_mean != None:
#                image -= np.reshape(image_mean, [1,1,1,2])
#             if image_std != None:
#                image /= np.reshape(image_std, [1,1,1,2])
                 
            if args.use_crf:
                pred, pred_crf, prob, prob_crf = sess.run([pred_op, pred_crf_op,
                                                           prob_op, prob_crf_op], 
                                     feed_dict={image_node: image[np.newaxis,...],
                                                label_node: label[np.newaxis,...],
                                                phase_train: False})
            else:
                pred, prob = sess.run([pred_op, prob_op], 
                                feed_dict={image_node: image[np.newaxis,...],
                                           label_node: label[np.newaxis,...],
                                           phase_train: False})
            dice_val_ = computeDice(label[...,args.feat_index], pred[0])
            dice_val.append(dice_val_)
            if args.use_crf:
                dice_val_crf_ = computeDice(label[...,args.feat_index], pred_crf[0])
                dice_val_crf.append(dice_val_crf_)
            
            if args.save_prob_dir != '':
                alldata[case_name] = {}
                alldata[case_name]['image'] = image
                alldata[case_name]['label'] = label
                alldata[case_name]['pred'] = pred[0]
                alldata[case_name]['prob'] = prob[0]
                #alldata[case_name]['ct_sitk'] = ct_sitk
            log_file_handle.write('{} {} {}'.format(case_name, dice_val_,
                                    dice_val_crf_ if args.use_crf==1 else ''))
            
            if args.save_dir != '':
                case_save_dir = '{}/{}'.format(args.save_dir, case_name)
                if not os.path.exists(case_save_dir):
                    os.makedirs(case_save_dir)
                
                if args.use_crf==1:
                    new_sitk_ct = sitk.GetImageFromArray(pred[0].astype(np.int32))
                    new_sitk_ct.CopyInformation(ct_sitk)
                    sitk.WriteImage(new_sitk_ct, str('{}/crf1_pred_{}_before.nii.gz'.format(case_save_dir,
                                                                                       'ct' if args.feat_index==0 else 'pt')))
                    new_sitk_ct = sitk.GetImageFromArray(pred_crf[0].astype(np.int32))
                    new_sitk_ct.CopyInformation(ct_sitk)
                    sitk.WriteImage(new_sitk_ct, str('{}/crf1_pred_{}_after.nii.gz'.format(case_save_dir,
                                                                                          'ct' if args.feat_index==0 else 'pt')))
                    new_sitk_ct = sitk.GetImageFromArray(prob[0][...,1].astype(np.float32))
                    new_sitk_ct.CopyInformation(pt_sitk)
                    sitk.WriteImage(new_sitk_ct, str('{}/crf1_prob_{}_before.nii.gz'.format(case_save_dir,
                                                                                       'ct' if args.feat_index==0 else 'pt')))
                    new_sitk_ct = sitk.GetImageFromArray(prob_crf[0][...,1].astype(np.float32))
                    new_sitk_ct.CopyInformation(pt_sitk)
                    sitk.WriteImage(new_sitk_ct, str('{}/crf1_prob_{}_after.nii.gz'.format(case_save_dir,
                                                                                       'ct' if args.feat_index==0 else 'pt')))
                else:
                    new_sitk_ct = sitk.GetImageFromArray(pred[0].astype(np.int32))
                    new_sitk_ct.CopyInformation(ct_sitk)
                    sitk.WriteImage(new_sitk_ct, str('{}/crf0_pred_{}.nii.gz'.format(case_save_dir,
                                                                                       'ct' if args.feat_index==0 else 'pt')))
                    new_sitk_ct = sitk.GetImageFromArray(prob[0][...,1].astype(np.float32))
                    new_sitk_ct.CopyInformation(pt_sitk)
                    sitk.WriteImage(new_sitk_ct, str('{}/crf0_prob_{}.nii.gz'.format(case_save_dir,
                                                                                       'ct' if args.feat_index==0 else 'pt')))
        
        if args.save_prob_dir != '':
            alldata['val_filenames'] = val_filenames
            alldata['test_filenames'] = test_filenames
            pickle.dump( alldata, 
                         open( args.save_prob_dir + '/alldata.p', "wb" ) )
                
        log_file_handle.write('Mean {} Dice(Before CRF): {}'.format(modes[args.feat_index], np.mean(np.array(dice_val))))
        if args.use_crf:
            log_file_handle.write('Mean {} Dice(After  CRF): {}'.format(modes[args.feat_index], np.mean(np.array(dice_val_crf))))
        
        return
    
    
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
                dice_val = []
                if args.use_crf:
                    dice_val_crf = []
                for f in filenames:
                    subject_id = f[0]
                    img_fn = f[1]
                    case_name = img_fn.split('/')[-1]
                    # if '2141' in case_name:
                    #     continue
                
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
                         
                    if args.use_crf:
                        pred, pred_crf = sess.run([pred_op, pred_crf_op], 
                                             feed_dict={image_node: image[np.newaxis,...],
                                                        label_node: label[np.newaxis,...],
                                                        phase_train: False})
                    else:
                        pred = sess.run([pred_op], 
                                        feed_dict={image_node: image[np.newaxis,...],
                                                   label_node: label[np.newaxis,...],
                                                   phase_train: False})
                    dice_val_ = computeDice(label[...,args.feat_index], pred[0])
                    dice_val.append(dice_val_)
                    if args.use_crf:
                        dice_val_crf_ = computeDice(label[...,args.feat_index], pred_crf[0])
                        dice_val_crf.append(dice_val_crf_)
                    if args.use_crf:
                        log_file_handle.write('{} {} {}\n'.format(case_name, dice_val_, dice_val_crf_))
                    else:
                        log_file_handle.write('{} {}\n'.format(case_name, dice_val_))
                
                log_file_handle.write('{} {} Mean {} Dice(Before CRF): {}\n'.format(step, 
                                                                                    subsets[subi], 
                                                                                    modes[args.feat_index], 
                                                                                    np.mean(np.array(dice_val))))
                if args.use_crf:
                    log_file_handle.write('{} {} Mean {} Dice(After CRF): {}\n'.format(step, 
                                                                                       subsets[subi], 
                                                                                       modes[args.feat_index], 
                                                                                       np.mean(np.array(dice_val_crf))))
                 
            if args.use_crf:
                crf_sw_value, crf_bw_value, crf_cm_value \
                = sess.run([crf_sw_op, crf_bw_op, crf_cm_op])
                print('sw:', crf_sw_value)
                print('bw:', crf_bw_value)
                print('cm:', crf_cm_value)
            
        
        if step % STEPS_ONE_EPOCH == 0:
            if args.use_crf:
                checkpoint_path = os.path.join(log_dir, 'model_crf.ckpt')
            else:
                checkpoint_path = os.path.join(log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)             
        
        # train one epoch
        start_time = time.time()
        image_batch, label_batch, case_name_batch = sess.run([train_images, train_labels, train_case_names])
        if args.use_crf:
            summary, _, loss_reg_value, \
            total_loss_crf_value, loss_crf_value, \
            total_loss_value, loss_value, \
            pred, pred_crf, \
            dice, dice_crf \
            = sess.run([summary_op, train_op, loss_reg_op,
                        total_loss_crf_op, loss_crf_op,
                        total_loss_op, loss_op, 
                        pred_op, pred_crf_op,
                        dice_op, dice_crf_op], 
                     feed_dict={image_node: image_batch,
                                label_node: label_batch,
                                phase_train: False})
        else:
            summary, _, loss_reg_value, \
            total_loss_value, loss_value, \
            pred, dice = sess.run([summary_op, train_op, loss_reg_op,
                                     total_loss_op, loss_op, 
                                     pred_op, dice_op], 
                             feed_dict={image_node: image_batch,
                                        label_node: label_batch,
                                        phase_train: True})
        
        duration = time.time() - start_time
        
#         if args.use_crf:
#             log_file_handle.write('{}, step {:d}, {:.6f}, ' \
#                              '{:.6f}, {:.6f}, ' \
#                              '{:.6f}, {:.6f}, ' \
#                              '{:.3f}, {:.3f}, {}\n'.format(modes[args.feat_index],
#                                                          step, loss_reg_value, 
#                                                          total_loss_crf_value, loss_crf_value,
#                                                          total_loss_value, loss_value,
#                                                          dice, dice_crf, case_name_batch[0]))
#             log_file_handle.flush()
#         else:
#             log_file_handle.write('{}, step {:d}, {:.6f}, ' \
#                              '{:.6f}, {:.6f}, ' \
#                              '{:.3f}, {}\n'.format(modes[args.feat_index],
#                                                          step, loss_reg_value, 
#                                                          total_loss_value, loss_value,
#                                                          dice, case_name_batch[0]))
#             log_file_handle.flush()

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
