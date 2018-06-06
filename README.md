# ISBI2018_PETCT_Segmentation

This repository contains the code (in TensorFlow) for "[3D fully convolutional networks for co-segmentation of tumors on PET-CT images](https://ieeexplore.ieee.org/abstract/document/8363561/)" paper (ISBI 2018). And one extended journal version with much more details is under revision.

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
