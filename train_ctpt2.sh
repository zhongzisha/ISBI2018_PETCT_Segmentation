GPU_ID=1
export CUDA_VISIBLE_DEVICES=$GPU_ID
python train_ctpt_fusion2.py \
--use_crf 0 \
--step 0 \
--log_dir logs3d \
--net_type myfusionunet2_bn \
--batch_size 2 \
--base_lr 1e-4 \
--lr_policy piecewise \
--opt_type adam \
--norm_type instancenorm_mean \
--loss_type dice \
--reg_coef 0.1 \
--random_seed 42 \
--with_aug 0 \
--num_epochs 21 \
--decay_epochs 5

# --restore_ckpt "logs_fct_optadam_lrconstant5e-05_b4_crf0_reg0.0_rs42/model.ckpt-2000" \
# --restore_ckpt "logs_f=ct_opt=adam_lr=constant0.0001_b2_crf=0_reg=0.0_rs=42/model.ckpt-19000" \
# --restore_from_saved_model "/media/ubuntu/working/petct_cnn/logs_dltk_ctpt/ct_only/1513006564/" \
# --restore_from_saved_model "/media/ubuntu/working/petct_cnn/logs_dltk_ctpt/pt_only/1512986497/" \
# --restore_from_saved_model "/media/ubuntu/working/petct_cnn/logs_dltk_ctpt/ctpt_fusion/1513123317/" \
# --restore_from_saved_model "/media/ubuntu/working/petct_cnn/logs_dltk_ctpt/ctpt_fusion2/1513149649/" \
# --restore_from_saved_model "/media/ubuntu/working/petct_cnn/logs_dltk_ctpt/ct_only/1512967074/" \ # this is the first epoch

