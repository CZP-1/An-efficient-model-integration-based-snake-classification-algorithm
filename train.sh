# #resnet50 baseline

# python train.py /home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size/1990 -c configs/base.yaml \
#         --model resnet50 \
#         --img-size 224 \
#         --batch-size 128 \
#         --output /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/resnet50_baseline_apex_aa_rand_m9_mstd0p5 \
#         --apex-amp \
#         --aa rand-m9-mstd0.5

# swin_large_patch4_window12_384
# python train_f1.py /home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size/1990 -c configs/SwinT.yaml \
#         --initial-checkpoint /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/swin_large_patch4_window12_384_rand_m9_mstd0.5/20220406-002636-swin_large_patch4_window12_384-384/Best_f1_score.pth.tar \
#         --cutmix 0.2 \
#         --lr 0.001 \
#         --aa rand-m9-mstd0.5 \
#         --output /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/swin_large_patch4_window12_384_ra_cutmix


# # tf_efficientnet_b6_ns
# python train_f1.py /home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size/1990 -c configs/tf_efficient_b6.yaml \
#         --freeze-layer 3 \
#         --lr 0.01 \
#         --output /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/

#crop swintranformer
# python train_detection.py /home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size/1990 -c configs/SwinT.yaml \
#         --cutmix 0.2 \
#         --lr 0.001 \
#         --aa rand-m9-mstd0.5 \
#         --output /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/swin_large_patch4_window12_384_ra_cutmix_crop

#微调加上--warm-epochs  0 同时decay-rate为0.1
# #ArcFace + swintranformer
# python train_arcface.py /home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size/1990 -c configs/SwinT.yaml \
#         --lr 0.001 \
#         --aa rand-m9-mstd0.5 \
#         --arcface \
#         -b 2 \
#         --smoothing 0 \
#         --output /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/swin_large_patch4_window12_384_ra_cutmix_arcface



# # convnext_large_384_in22ft1k
# python train_f1.py /home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size/1990 -c configs/volo_d5_512.yaml \
#         --lr 0.01 \
#         --output /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/volo_d5_512



# tf_efficientnet_b7_ns
# python train_f1.py /home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size/1990 -c configs/tf_efficient.yaml \
#         --resume /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/tf_efficient_imgsize600_freeze_2/20220411-202321-tf_efficientnet_b7_ns-600/last.pth.tar \
#         --freeze-layer 2 \
#         --decay-rate 0.1 \
#         --resume-loss 0.9572 \
#         --warmup-epochs 0 \
#         --output /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/tf_efficient_imgsize600_freeze_2_resume_real
        
        # --lr 0.001 \
        # --cutmix 0.2 \
        # --aa rand-m9-mstd0.5 \
        

# swin_large_patch4_window12_384
# python train_f1.py /home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size/1990 -c configs/SwinT.yaml \
#         --aa rand-m9-mstd0.5 \
#         --cutmix 0.2 \
#         --initial-checkpoint /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/swin_large_patch4_window12_384_all_data/20220417-170713-swin_large_patch4_window12_384-384/checkpoint-27.pth.tar \
#         --warmup-epochs 0 \
#         --lr 0.001 \
#         --output /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/swin_large_patch4_window12_384_all_data_ra_cutmix

# # tf_efficientnet_b7_ns
# python train_f1.py /home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size/1990 -c configs/tf_efficientnet_b7_ns_multistep.yaml \
#         --lr 0.01 \
#         --freeze-layer 2 \
#         --output /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/tf_efficientnet_b7_ns_data_crop_data

# tf_efficientnet_l2_ns
# python train_f1.py /home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size/1990 -c configs/tf_efficientnet_l2_ns.yaml \
#         --freeze-layer 3 \
#         --lr 0.01 \
#         --output /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/tf_efficientnet_l2_ns_freeze3
# # PIM
# python train_pim.py /home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size/1990 -c configs/pim.yaml \
#         --lr 0.001 \
#         -b 1 \
#         --output /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/swin_large_patch4_window12_384_pim
# tf_efficientnet_b7_ns
python train_f1.py /home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size/1990 -c configs/resnest269e.yaml \
        --lr 0.001 \
        --freeze-layer 5 \
        --initial-checkpoint /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/resnest269e_all_data/20220504-014731-resnest269e-416/checkpoint-26.pth.tar \
        --warmup-epochs 0 \
        --cutmix 0.2 \
        --output /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/resnest269e_all_data_cutmix