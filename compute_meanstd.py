import numpy as np
import nibabel as nib
import glob
import pandas as pd

from myconfig import *

data_root = DATA_ROOT
train_filenames = pd.read_csv(
        TRAIN0_FILENAME,
        dtype=object,
        keep_default_na=False,
        na_values=[]).as_matrix()
means = list()
stds = list()

min_percent = 1
max_percent = 99
for case in train_filenames:
    name = case[1]
    ctf = nib.load(name + '/InputCT_ROI.nii.gz')
    ct = ctf.get_data().astype(np.float32)
    ct[ct>200.] = 200.
    ct[ct<-500.] = -500.
    ct = 255*(ct+500)/(700.)
#     ptf = nib.load(name + '/InputPET_ROI.nii.gz')
#     pt = ptf.get_data().astype(np.float32)
#     pt_min = np.percentile(pt, q=min_percent)
#     pt_max = np.percentile(pt, q=max_percent)
#     pt[pt<pt_min]=pt_min
#     pt[pt>pt_max]=pt_max
#     pt = 255*(pt-pt_min)/(pt_max-pt_min)
    ptsuvf = nib.load(name + '/InputPET_SUV_ROI.nii.gz')
    ptsuv = ptsuvf.get_data().astype(np.float32)
    ptsuv[ptsuv<0.01]=0.01
    ptsuv[ptsuv>20.]=20.
    ptsuv = 255*(ptsuv-0.01)/(19.99)
    image = np.stack([ct, ptsuv], axis=-1)
    means.append(image.mean(axis=(0,1,2)))
    stds.append(image.std(axis=(0,1,2)))
    
means = np.asarray(means)
print(means.shape)
image_mean = means.mean(axis=0)
stds = np.asarray(stds)
print(stds.shape)
image_std = stds.mean(axis=0)
print(image_mean)
print(image_std)
np.save(data_root + '/image_mean_ctpt.npy', 
        {'image_mean':image_mean.astype(np.float32),
         'image_std':image_std.astype(np.float32)})






