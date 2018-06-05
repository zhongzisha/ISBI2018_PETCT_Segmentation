FEAT_INDEX=1
GPU_ID=0
export CUDA_VISIBLE_DEVICES=$GPU_ID
python train_single_3d.py \
--use_crf 0 \
--step 0 \
--log_dir logs3d \
--net_type myunet3d_bn_crf \
--feat_index ${FEAT_INDEX} \
--batch_size 4 \
--base_lr 1e-4 \
--lr_policy piecewise \
--opt_type adam \
--norm_type instancenorm_mean \
--loss_type ce \
--reg_coef 0.1 \
--random_seed 42
