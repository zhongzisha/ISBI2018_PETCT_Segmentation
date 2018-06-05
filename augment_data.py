import os
import numpy as np
import glob
import nibabel as nib

from myconfig import *

def rotate_1(img,lab):
    # img,lab must be [channels,height,width,depth]
    c1,h,w,d = img.shape
    c2 = lab.shape[0]
    img_rot = np.zeros((c1,h,w,d),dtype=np.float32)
    lab_rot = np.zeros((c2,h,w,d),dtype=np.uint8)
    for i in range(d):
        for j in range(c1):
            img_rot[j,:,:,i] = np.rot90(img[j,:,:,i],1)
        for j in range(c2):
            lab_rot[j,:,:,i] = np.rot90(lab[j,:,:,i],1)
    return img_rot, lab_rot
  
def rotate_2(img,lab):
    # img,lab must be [channels,height,width,depth]
    c1,h,w,d = img.shape
    c2 = lab.shape[0]
    img_rot = np.zeros((c1,h,w,d),dtype=np.float32)
    lab_rot = np.zeros((c2,h,w,d),dtype=np.uint8)
    for i in range(d):
        for j in range(c1):
            img_rot[j,:,:,i] = np.rot90(img[j,:,:,i],2)
        for j in range(c2):
            lab_rot[j,:,:,i] = np.rot90(lab[j,:,:,i],2)
    return img_rot, lab_rot      

def rotate_3(img,lab):
    # img,lab must be [channels,height,width,depth]
    c1,h,w,d = img.shape
    c2 = lab.shape[0]
    img_rot = np.zeros((c1,h,w,d),dtype=np.float32)
    lab_rot = np.zeros((c2,h,w,d),dtype=np.uint8)
    for i in range(d):
        for j in range(c1):
            img_rot[j,:,:,i] = np.rot90(img[j,:,:,i],3)
        for j in range(c2):
            lab_rot[j,:,:,i] = np.rot90(lab[j,:,:,i],3)
    return img_rot, lab_rot
    
def rotate_4(img,lab):
    # img,lab must be [channels,height,width,depth]
    c1,h,w,d = img.shape
    c2 = lab.shape[0]
    img_rot = np.zeros((c1,h,w,d),dtype=np.float32)
    lab_rot = np.zeros((c2,h,w,d),dtype=np.uint8)
    for i in range(d):
        for j in range(c1):
            img_rot[j,:,:,i] = np.fliplr(img[j,:,:,i])
        for j in range(c2):
            lab_rot[j,:,:,i] = np.fliplr(lab[j,:,:,i])
    return img_rot, lab_rot 
    
def rotate_5(img,lab):
    # img,lab must be [channels,height,width,depth]
    c1,h,w,d = img.shape
    c2 = lab.shape[0]
    img_rot = np.zeros((c1,h,w,d),dtype=np.float32)
    lab_rot = np.zeros((c2,h,w,d),dtype=np.uint8)
    for i in range(d):
        for j in range(c1):
            img_rot[j,:,:,i] = np.flipud(img[j,:,:,i])
        for j in range(c2):
            lab_rot[j,:,:,i] = np.flipud(lab[j,:,:,i])
    return img_rot, lab_rot 
    
def rotate_6(img,lab):
    # img,lab must be [channels,height,width,depth]
    c1,h,w,d = img.shape
    c2 = lab.shape[0]
    img_rot = np.zeros((c1,h,w,d),dtype=np.float32)
    lab_rot = np.zeros((c2,h,w,d),dtype=np.uint8)
    for i in range(d):
        for j in range(c1):
            img_rot[j,:,:,i] = np.flipud(np.rot90(img[j,:,:,i],1))
        for j in range(c2):
            lab_rot[j,:,:,i] = np.flipud(np.rot90(lab[j,:,:,i],1))
    return img_rot, lab_rot 

def rotate_7(img,lab):
    # img,lab must be [channels,height,width,depth]
    c1,h,w,d = img.shape
    c2 = lab.shape[0]
    img_rot = np.zeros((c1,h,w,d),dtype=np.float32)
    lab_rot = np.zeros((c2,h,w,d),dtype=np.uint8)
    for i in range(d):
        for j in range(c1):
            img_rot[j,:,:,i] = np.flipud(np.rot90(img[j,:,:,i],3))
        for j in range(c2):
            lab_rot[j,:,:,i] = np.flipud(np.rot90(lab[j,:,:,i],3))
    return img_rot, lab_rot 

def rotate_8(img,lab):
    # img,lab must be [channels,height,width,depth]
    c1,h,w,d = img.shape
    c2 = lab.shape[0]
    img_rot = np.zeros((c1,h,w,d),dtype=np.float32)
    lab_rot = np.zeros((c2,h,w,d),dtype=np.uint8)
    for i in range(d):
        for j in range(c1):
            img_rot[j,:,:,i] = np.flipud(np.rot90(img[j,:,:,i],2))
        for j in range(c2):
            lab_rot[j,:,:,i] = np.flipud(np.rot90(lab[j,:,:,i],2))
    return img_rot, lab_rot 

def rotate_9(img,lab):
    # img,lab must be [channels,height,width,depth]
    c1,h,w,d = img.shape
    c2 = lab.shape[0]
    img_rot = np.zeros((c1,h,w,d),dtype=np.float32)
    lab_rot = np.zeros((c2,h,w,d),dtype=np.uint8)
    for i in range(d):
        for j in range(c1):
            img_rot[j,:,:,i] = np.fliplr(np.rot90(img[j,:,:,i],1))
        for j in range(c2):
            lab_rot[j,:,:,i] = np.fliplr(np.rot90(lab[j,:,:,i],1))
    return img_rot, lab_rot 

def rotate_10(img,lab):
    # img,lab must be [channels,height,width,depth]
    c1,h,w,d = img.shape
    c2 = lab.shape[0]
    img_rot = np.zeros((c1,h,w,d),dtype=np.float32)
    lab_rot = np.zeros((c2,h,w,d),dtype=np.uint8)
    for i in range(d):
        for j in range(c1):
            img_rot[j,:,:,i] = np.fliplr(np.rot90(img[j,:,:,i],3))
        for j in range(c2):
            lab_rot[j,:,:,i] = np.fliplr(np.rot90(lab[j,:,:,i],3))
    return img_rot, lab_rot 

def rotate_11(img,lab):
    # img,lab must be [channels,height,width,depth]
    c1,h,w,d = img.shape
    c2 = lab.shape[0]
    img_rot = np.zeros((c1,h,w,d),dtype=np.float32)
    lab_rot = np.zeros((c2,h,w,d),dtype=np.uint8)
    for i in range(d):
        for j in range(c1):
            img_rot[j,:,:,i] = np.fliplr(np.rot90(img[j,:,:,i],2))
        for j in range(c2):
            lab_rot[j,:,:,i] = np.fliplr(np.rot90(lab[j,:,:,i],2))
    return img_rot, lab_rot 

HEIGHT = 96
WIDTH = 96
DEPTH = 48
data_root = DATA_ROOT
sub = 'train'
names = glob.glob(data_root + '/' + sub + '/A-*')
for name in names:
    # [channels, depth, height, width]
    
    image = np.zeros((3, HEIGHT, WIDTH, DEPTH),dtype=np.float32)
    label = np.zeros((2, HEIGHT, WIDTH, DEPTH),dtype=np.uint8)
    image_prefixes = ['InputCT_ROI', 'InputPET_ROI', 'InputPET_SUV_ROI']
    label_prefixes = ['GTV_Primary_ROI_CT{}'.format(GT_POSTFIX), 'GTV_Primary_ROI_PET{}'.format(GT_POSTFIX)]
    for i, prefix in enumerate(image_prefixes):
        f = nib.load('{}/{}.nii.gz'.format(name, prefix))
        image[i, ...] = f.get_data().astype(np.float32)
    for i, prefix in enumerate(label_prefixes):
        f = nib.load('{}/{}.nii.gz'.format(name, prefix))
        label[i, ...] = f.get_data().astype(np.uint8)
    
    c1 = 3
    c2 = 2
    num = 7
    image_raw_rot = np.zeros((num,c1,HEIGHT, WIDTH, DEPTH),dtype=np.float32)
    label_raw_rot = np.zeros((num,c2,HEIGHT, WIDTH, DEPTH),dtype=np.uint8)
    image_raw_rot[0,...], label_raw_rot[0,...] = rotate_1(image, label)
    image_raw_rot[1,...], label_raw_rot[1,...] = rotate_2(image, label)
    image_raw_rot[2,...], label_raw_rot[2,...] = rotate_3(image, label)
    image_raw_rot[3,...], label_raw_rot[3,...] = rotate_4(image, label)
    image_raw_rot[4,...], label_raw_rot[4,...] = rotate_5(image, label)
    image_raw_rot[5,...], label_raw_rot[5,...] = rotate_6(image, label)
    image_raw_rot[6,...], label_raw_rot[6,...] = rotate_7(image, label)
    # image_raw_rot[7,...], label_raw_rot[7,...] = rotate_8(image, label) # equal to rotate_4
    # image_raw_rot[8,...], label_raw_rot[8,...] = rotate_9(image, label) # equal to rotate_7
    # image_raw_rot[9,...], label_raw_rot[9,...] = rotate_10(image, label) # equal to rotate_6
    # image_raw_rot[10,...], label_raw_rot[10,...] = rotate_11(image, label) # equal to rotate_5
    
    affine = np.eye(4)
    for i in range(num):
        save_dir = name+'_rot'+str(i)
        os.makedirs(save_dir)
        for j, prefix in enumerate(image_prefixes):
            img = nib.Nifti1Image(image_raw_rot[i,j,:,:,:], affine)
            nib.save(img, '{}/{}.nii.gz'.format(save_dir, prefix))
        for j, prefix in enumerate(label_prefixes):
            img = nib.Nifti1Image(label_raw_rot[i,j,:,:,:], affine)
            nib.save(img, '{}/{}.nii.gz'.format(save_dir, prefix))
    
    print(name)
    # break








