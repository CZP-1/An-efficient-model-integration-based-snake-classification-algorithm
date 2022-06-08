# SAR 1-9 finetune
# python train_2nd_1_9.py /home/data3/changhao/Datasets/PBVS2022/SAR_0_removed/train -c configs/base.yaml \
#         --num-classes 9 \
#         --model resnet50 \
#         --initial-checkpoint /home/data3/changhao/WorkSpaceRecord/PBVS2022_timm/fold0_resnet50/20220223-110944-resnet50-224/model_best.pth.tar \
#         --lr 0.001 \
#         --img-size 224 \
#         --batch-size 64 \
#         --mean 0.485 0.456 0.406 \
#         --std 0.229 0.224 0.225 \
#         --cutmix 0.2 \
#         --output /home/data3/changhao/WorkSpaceRecord/PBVS2022_sampled/sampled_resnet50_mean_std_cutmix_2nd_1_9

# # 0-9 finetune 1 stage
# python train_2nd_1_9.py /home/data3/changhao/Datasets/PBVS2022/SAR_0_removed/train -c configs/base.yaml \
#         --num-classes 10 \
#         --model resnet50 \
#         --initial-checkpoint /home/data3/changhao/WorkSpaceRecord/PBVS2022_sampled/sampled_resnet50_mean_std_cutmix_2nd_1_9/20220302-231606-resnet50-224/model_best.pth.tar \
#         --lr 0.001 \
#         --img-size 224 \
#         --batch-size 64 \
#         --mean 0.485 0.456 0.406 \
#         --std 0.229 0.224 0.225 \
#         --cutmix 0.2 \
#         --epochs 20 \
#         --output /home/data3/changhao/WorkSpaceRecord/PBVS2022_sampled/sampled_resnet50_mean_std_cutmix_2nd_10_1stage

# 0-9 finetune 2 stage
python train_2nd_1_9.py /home/data3/changhao/Datasets/PBVS2022/SAR_0_removed/train -c configs/base.yaml \
        --num-classes 10 \
        --model resnet50 \
        --initial-checkpoint /home/data3/changhao/WorkSpaceRecord/PBVS2022_sampled/sampled_resnet50_mean_std_cutmix_2nd_10_1stage/20220303-145657-resnet50-224/model_best.pth.tar \
        --lr 0.001 \
        --img-size 224 \
        --batch-size 64 \
        --mean 0.485 0.456 0.406 \
        --std 0.229 0.224 0.225 \
        --cutmix 0.2 \
        --epochs 10 \
        --freeze-layer 8 \
        --warmup-epochs 0 \
        --output /home/data3/changhao/WorkSpaceRecord/PBVS2022_sampled/sampled_resnet50_mean_std_cutmix_2nd_10_2stage
