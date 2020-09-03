# For CT-Only
FEAT_INDEX=0
python test_single_3d.py \
                --use_crf 0 \
                --restore_ckpt_meta trained_models/ct-only/model.ckpt-7524.meta \
                --net_type myunet3d_bn_crf \
                --feat_index ${FEAT_INDEX} \
                --norm_type instancenorm_mean \
                --random_seed 42 \
                --test_filenames data/test.csv

# For PET-Only
FEAT_INDEX=1
python test_single_3d.py \
                --use_crf 0 \
                --restore_ckpt_meta trained_models/pet-only/model.ckpt-5016.meta \
                --net_type myunet3d_bn_crf \
                --feat_index ${FEAT_INDEX} \
                --norm_type instancenorm_mean \
                --random_seed 42 \
                --test_filenames data/test.csv


# for PET_CT_CoSeg
python test_ctpt_fusion2.py \
                --use_crf 0 \
                --restore_ckpt_meta trained_models/pet-ct-fusion/model.ckpt-40128.meta  \
                --net_type myfusionunet2_bn \
                --norm_type instancenorm_mean \
                --random_seed 42 \
                --test_filenames data/test.csv
