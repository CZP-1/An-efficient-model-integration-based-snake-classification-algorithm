# SAR mobilenetv3_large_100_miil + mean_std + cutmix + cosine
# python train_sampled.py /home/data3/changhao/Datasets/PBVS2022/train_images -c configs/base.yaml \
#         --model mobilenetv3_large_100_miil \
#         --img-size 224 \
#         --batch-size 64 \
#         --mean 0.485 0.456 0.406 \
#         --std 0.229 0.224 0.225 \
#         --cutmix 0.2 \
#         --sched  cosine \
#         --lr 0.01 \
#         --lr-cycle-limit 4 \
#         --lr-cycle-mul 2 \
#         --lr-cycle-decay 0.6 \
#         --epochs 5 \
#         --output /home/data3/changhao/WorkSpaceRecord/PBVS2022_sampled/sampled_mobilenetv3_large_100_miil_mean_std_cutmix_cosine_4multi

# SAR mobilenetv3_large_100_miil + mean_std + cutmix + cosine + seed
# python train_sampled.py /home/data3/changhao/Datasets/PBVS2022/train_images -c configs/base.yaml \
#         --model mobilenetv3_large_100_miil \
#         --img-size 224 \
#         --batch-size 64 \
#         --mean 0.485 0.456 0.406 \
#         --std 0.229 0.224 0.225 \
#         --cutmix 0.2 \
#         --seed 1 \
#         --drop-block 0.1 \
#         --output /home/data3/changhao/WorkSpaceRecord/PBVS2022_sampled/sampled_mobilenetv3_large_100_miil_mean_std_cutmix_seed1_model_dropblock

# SAR resnet50 + mean_std + cutmix + cosine
# python train_sampled.py /home/data3/changhao/Datasets/PBVS2022/train_images -c configs/base.yaml \
#         --model resnet50 \
#         --img-size 224 \
#         --batch-size 64 \
#         --mean 0.485 0.456 0.406 \
#         --std 0.229 0.224 0.225 \
#         --cutmix 0.2 \
#         --output /home/data3/changhao/WorkSpaceRecord/PBVS2022_sampled/sampled_resnet50_mean_std_cutmix

# SAR backbone + mean_std + cutmix + cosine
python train_sampled.py /home/lkd22/PBVS/all_data/SAR/train -c configs/base.yaml \
        --model vit_base_patch16_384 \
        --img-size 384 \
        --batch-size 64 \
        --mean 0.485 0.456 0.406 \
        --std 0.229 0.224 0.225 \
        --cutmix 0.2 \
        --seed 1 \
        --apex-amp \
        --output /home/data3/changhao/WorkSpaceRecord/PBVS2022_sampled/sampled_vit_base_patch16_384_mean_std_cutmix
