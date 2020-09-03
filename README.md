# ISBI2018_PETCT_Segmentation

This repository contains the code (in TensorFlow) for "[3D fully convolutional networks for co-segmentation of tumors on PET-CT images](https://ieeexplore.ieee.org/abstract/document/8363561/)" paper (ISBI 2018). Compared to the previous semi-automated methods, this method is highly automated without manually user-defined seeds. 

**UPDATED**

1. Uploaded the DFCN-CoSeg training and testing code for our extended work published in
https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.13331 (MP2018), which provided much details
compared to the ISBI2018 paper. 

2. Uploaded our previous trained models for `CT-Only`, `PET-Only` and `DFCN-CoSeg` networks studied in 
MP2018. The models can be downloaded in BaiduYun 
(https://pan.baidu.com/s/1tCsjfuckkU9IH8O4xewsRQ Password: tfkt).

3. As for now, I cannot install the outdated `tensorflow_gpu==1.4` in my working `Ubuntu 20.04`, 
so I uploaded two cases of PET-CT images and the testing code using `tensorflow_gpu==2.3`, 
interested readers can check the `test.sh` script.
**Please note that we just use the `tensorflow_gpu==2.3` in the testing code, not for training.**

## CT/PET Segmentation Results on One Patient

### 1. CT image

<img align="center" src="https://github.com/zhongzisha/ISBI2018_PETCT_Segmentation/raw/master/CT.PNG">

### 2. PET_SUV image

<img align="center" src="https://github.com/zhongzisha/ISBI2018_PETCT_Segmentation/raw/master/PET_SUV.PNG">

### 3. Ground Truth Segmentation on CT image

<img align="center" src="https://github.com/zhongzisha/ISBI2018_PETCT_Segmentation/blob/master/CT_Ground%20Truth.PNG">

### 4. Ground Truth Segmentation on PET_SUV image

<img align="center" src="https://github.com/zhongzisha/ISBI2018_PETCT_Segmentation/blob/master/PET_Ground%20Truth.PNG">

### 5. Prediction on CT image

<img align="center" src="https://github.com/zhongzisha/ISBI2018_PETCT_Segmentation/blob/master/Prediction_CT.PNG">

### 6. Prediction on PET_SUV image

<img align="center" src="https://github.com/zhongzisha/ISBI2018_PETCT_Segmentation/blob/master/Prediction_PET.PNG">

### 7. Wrong Predictions on CT image

<img align="center" src="https://github.com/zhongzisha/ISBI2018_PETCT_Segmentation/blob/master/Prediction_CT_Wrong.PNG">

### 8. Wrong Predictions on PET_SUV image

<img align="center" src="https://github.com/zhongzisha/ISBI2018_PETCT_Segmentation/blob/master/Prediction_PET_Wrong.PNG">

## Dependencies

- [Python2.7](https://www.python.org/downloads/)
- [TensorFlow(1.4.0+)](http://www.tensorflow.org)
- [DLTK](https://dltk.github.io/)
- other libraries

## Citation

If you find this useful, please cite our work as follows:

```
@INPROCEEDINGS{zszhong2018isbi_petct,
  author={Z. Zhong and Y. Kim and L. Zhou and K. Plichta and B. Allen and J. Buatti and X. Wu},
  booktitle={2018 IEEE 15th International Symposium on Biomedical Imaging (ISBI 2018)},
  title={3D fully convolutional networks for co-segmentation of tumors on PET-CT images},
  year={2018},
  volume={},
  number={},
  pages={228-231},
  keywords={Biomedical imaging;Computed tomography;Image segmentation;Lung;Three-dimensional displays;Tumors;co-segmentation;deep learning;fully convolutional networks;image segmentation;lung tumor segmentation},
  doi={10.1109/ISBI.2018.8363561},
  ISSN={},
  month={April},
}
```

## Contacts
zhongzisha@outlook.com

Any discussions or concerns are welcomed!
