# SemiStain

The official implementation of "Semi-supervised Virtual IHC Staining via Pseudo-label Guided Contrastive Learning on Inconsistently Paired Training Data".

## training scripts:
```shell
sh run_train.sh
```

```shell
CUDA_VISIBLE_DEVICES=0 python train.py  --gpu_ids 0 --dataroot MIST/ER/TrainValAB/ \
  --name SemiStain_ER --checkpoints_dir checkpoints --model cptv8 --CUT_mode CUT --n_epochs 26 --n_epochs_decay 0 \
  --netD n_layers --ndf 32 --netG resnet_9blocks --n_layers_D 5 --normG instance --normD instance --weight_norm spectral \
  --lambda_GAN 1.0 --lambda_NCE 2.5 --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256 --lambda_gp 1.0 \
  --gp_weights [0.015625,0.03125,0.0625,0.125,0.25,1.0] --lambda_asp 5.0 --asp_loss_mode lambda_linear \
  --dataset_mode aligned --direction AtoB \
  --num_threads 1 --batch_size 1 --load_size 1024 --crop_size 512 --preprocess crop --flip_equivariance False \
  --display_winsize 512 --update_html_freq 100 --save_epoch_freq 1 --print_freq 1 \
  --display_id 0 \
  --gan_mode vanilla --use_semistain --psnr_txt data/ER_perpsnr.txt --ema 0.99 --psnr_threshold 80 \
  --continue_train --pretrained_name ER_stage1 --epoch 84 --lr 6e-5 \
  --lambda_consist 100.0 \
  --LSH_type weak_paired --LSH_ncluster 2 --lambda_LSH 2.0 --LSH_scalers 1,0,0,0,0 --LSH_patch_num 256 --nce_LSH 0.05 --fixed_cluster True --num_tables 5
```
