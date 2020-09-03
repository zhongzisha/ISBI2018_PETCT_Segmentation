import os

# DATA_ROOT = '/media/ubuntu/working/petct_mp_v2/v3/data/dataPETCT_CNN_48x48x24'
DATA_ROOT = '.'
TRAIN_FILENAME = DATA_ROOT + '/train0.csv'
TRAIN0_FILENAME = DATA_ROOT + '/trainForTest0.csv'
VAL_FILENAME = DATA_ROOT + '/valForTest0.csv'
TEST_FILENAME = DATA_ROOT + '/test0.csv'
HEIGHT = 96
WIDTH = 96
DEPTH = 48

IMAGE_MEAN = [91.439224, 30.57399 ]#[93.53382,36.65709]
IMAGE_STD  = [90.770485, 21.167143]#[92.32869,22.736958]


GT_POSTFIX = '_Staple'

if os.path.exists(TRAIN_FILENAME):
    NUM_TRAIN_SAMPLES = sum(1 for line in open(TRAIN_FILENAME)) - 1
if os.path.exists(VAL_FILENAME):
    NUM_VAL_SAMPLES = sum(1 for line in open(VAL_FILENAME)) - 1
if os.path.exists(TEST_FILENAME):
    NUM_TEST_SAMPLES = sum(1 for line in open(TEST_FILENAME)) - 1



NUM_CLASSES = 2