# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 07:22:32 2017

@author: ziszhong
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob

import tensorflow as tf
import numpy as np
import nibabel as nib
import pandas as pd
import SimpleITK as sitk

from myconfig import *

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_oneset(filenames):
    
    image_sum = np.zeros((DEPTH, HEIGHT, WIDTH, 2), dtype=np.float64)
    
    for f in filenames:
        img_fn = f[1]
        case_name = img_fn.split('/')[-1]
        filename = os.path.join(str(os.path.join(img_fn, 'data.tfrecords')))
        writer = tf.python_io.TFRecordWriter(filename)
        
        ct_sitk = sitk.ReadImage(str(os.path.join(img_fn, 'InputCT_ROI.nii.gz')))
        ct = sitk.GetArrayFromImage(ct_sitk).astype((np.float32))
        ptsuv_sitk = sitk.ReadImage(str(os.path.join(img_fn, 'InputPET_SUV_ROI.nii.gz')))
        ptsuv = sitk.GetArrayFromImage(ptsuv_sitk).astype((np.float32))
        lbl_ct = sitk.GetArrayFromImage(sitk.ReadImage(str(os.path.join(
            img_fn, 'GTV_Primary_ROI_CT.nii.gz')))).astype(np.uint8)
        lbl_pt = sitk.GetArrayFromImage(sitk.ReadImage(str(os.path.join(
            img_fn, 'GTV_Primary_ROI_PET.nii.gz')))).astype(np.uint8)
        
        ct[ct>200.] = 200.
        ct[ct<-500.] = -500.
        ct = 255*(ct+500)/(700.)
        ct = ct.astype(np.uint8)
        
        ptsuv[ptsuv<0.01]=0.01
        ptsuv[ptsuv>20.]=20.
        ptsuv = 255*(ptsuv-0.01)/(19.99)
        ptsuv = ptsuv.astype(np.uint8)

        image_raw = np.concatenate((ct[...,np.newaxis],ptsuv[...,np.newaxis]),axis=3) 
        label_raw = np.concatenate((lbl_ct[...,np.newaxis],lbl_pt[...,np.newaxis]),axis=3) 
        depth, height, width, channels = image_raw.shape
        example = tf.train.Example(features=tf.train.Features(feature={
            'case_name': _bytes_feature(case_name),
            'label_raw': _bytes_feature(label_raw.tostring()),
            'image_raw': _bytes_feature(image_raw.tostring())}))
        writer.write(example.SerializeToString()) 
        print(filename)
        writer.close()
        
        image_sum += image_raw.astype(np.float64)
        
    print('number of files: ', len(filenames))
    print(np.sum(image_sum, axis=(0,1,2))/(HEIGHT*WIDTH*DEPTH*len(filenames)))
        

def convert_oneset_for_str(filenames):
    
    image_sum = np.zeros((DEPTH, HEIGHT, WIDTH, 2), dtype=np.float64)
    
    for f in filenames:
        img_fn = f[1]
        case_name = img_fn.split('/')[-1]
        filename = os.path.join(str(os.path.join(img_fn, 'data2.tfrecords')))
        writer = tf.python_io.TFRecordWriter(filename)
        
        ct_sitk = sitk.ReadImage(str(os.path.join(img_fn, 'InputCT_ROI.nii.gz')))
        ct = sitk.GetArrayFromImage(ct_sitk).astype((np.float32))
        ptsuv_sitk = sitk.ReadImage(str(os.path.join(img_fn, 'InputPET_SUV_ROI.nii.gz')))
        ptsuv = sitk.GetArrayFromImage(ptsuv_sitk).astype((np.float32))
        lbl_ct = sitk.GetArrayFromImage(sitk.ReadImage(str(os.path.join(
            img_fn, 'GTV_Primary_ROI_CT{}.nii.gz'.format(GT_POSTFIX))))).astype(np.uint8)
        lbl_pt = sitk.GetArrayFromImage(sitk.ReadImage(str(os.path.join(
            img_fn, 'GTV_Primary_ROI_PET{}.nii.gz'.format(GT_POSTFIX))))).astype(np.uint8)
        
        ct[ct>200.] = 200.
        ct[ct<-500.] = -500.
        ct = 255*(ct+500)/(700.)
        ct = ct.astype(np.uint8)
        
        ptsuv[ptsuv<0.01]=0.01
        ptsuv[ptsuv>20.]=20.
        ptsuv = 255*(ptsuv-0.01)/(19.99)
        ptsuv = ptsuv.astype(np.uint8)

        image_raw = np.concatenate((ct[...,np.newaxis],ptsuv[...,np.newaxis]),axis=3) 
        label_raw = np.concatenate((lbl_ct[...,np.newaxis],lbl_pt[...,np.newaxis]),axis=3) 
        depth, height, width, channels = image_raw.shape
        example = tf.train.Example(features=tf.train.Features(feature={
            'case_name': _bytes_feature(tf.compat.as_bytes(case_name)),
            'label_raw': _bytes_feature(label_raw.tostring()),
            'image_raw': _bytes_feature(image_raw.tostring())}))
        writer.write(example.SerializeToString()) 
        # print(filename)
        writer.close()
        
        image_sum += image_raw.astype(np.float64)
        
    print('number of files: ', len(filenames))
    print(np.sum(image_sum, axis=(0,1,2))/(HEIGHT*WIDTH*DEPTH*len(filenames)))
        
def convert_oneset_2d(filenames):
    s = np.zeros((HEIGHT,WIDTH,2), dtype=np.float32)
    count = 0
    for f in filenames:
        img_fn = f[1]
        case_name = img_fn.split('/')[-1]
        filename = os.path.join(str(os.path.join(img_fn, 'data_2d.tfrecords')))
        
        writer = tf.python_io.TFRecordWriter(filename)
        
        ct_sitk = sitk.ReadImage(str(os.path.join(img_fn, 'InputCT_ROI.nii.gz')))
        ct = sitk.GetArrayFromImage(ct_sitk).astype((np.float32))
        ptsuv_sitk = sitk.ReadImage(str(os.path.join(img_fn, 'InputPET_SUV_ROI.nii.gz')))
        ptsuv = sitk.GetArrayFromImage(ptsuv_sitk).astype((np.float32))
        lbl_ct = sitk.GetArrayFromImage(sitk.ReadImage(str(os.path.join(
            img_fn, 'GTV_Primary_ROI_CT{}.nii.gz'.format(GT_POSTFIX))))).astype(np.uint8)
        lbl_pt = sitk.GetArrayFromImage(sitk.ReadImage(str(os.path.join(
            img_fn, 'GTV_Primary_ROI_PET{}.nii.gz'.format(GT_POSTFIX))))).astype(np.uint8)
        
        ct[ct>200.] = 200.
        ct[ct<-500.] = -500.
        ct = 255*(ct+500)/(700.)
        ct = ct.astype(np.uint8)
        
        ptsuv[ptsuv<0.01]=0.01
        ptsuv[ptsuv>20.]=20.
        ptsuv = 255*(ptsuv-0.01)/(19.99)
        ptsuv = ptsuv.astype(np.uint8)

        ctpt_and = np.logical_and(lbl_ct==1, lbl_pt==1).astype(np.uint8)
        for i in range(ctpt_and.shape[0]):
            if np.count_nonzero(ctpt_and[i,:,:])>20:
                image_raw = np.concatenate((ct[i,:,:,np.newaxis],ptsuv[i,:,:,np.newaxis]),axis=2)
                s += image_raw.astype(np.float32)
                label_raw = np.concatenate((lbl_ct[i,:,:,np.newaxis],lbl_pt[i,:,:,np.newaxis]),axis=2)
                example = tf.train.Example(features=tf.train.Features(feature={
                    'case_name': _bytes_feature(tf.compat.as_bytes('{}_{}'.format(case_name,i))),
                    'label_raw': _bytes_feature(label_raw.tostring()),
                    'image_raw': _bytes_feature(image_raw.tostring())}))
                writer.write(example.SerializeToString()) 
                count += 1
        # print(filename)
        writer.close()
    print(count, np.sum(s,axis=(0,1))/HEIGHT/WIDTH/count)
   
if __name__ == '__main__':
    
    train_filenames = pd.read_csv(
        TRAIN_FILENAME,
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
        
    # convert_oneset(train_filenames)
    # convert_oneset(val_filenames)
    # convert_oneset(test_filenames)

    convert_oneset_for_str(train_filenames)
    convert_oneset_for_str(val_filenames)
    convert_oneset_for_str(test_filenames)
    
    # convert_oneset_2d(train_filenames)
    # convert_oneset_2d(val_filenames)
    # convert_oneset_2d(test_filenames)










