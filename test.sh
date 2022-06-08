# SAR resnet50
# python test.py /home/data2/CZP/SnakeCLEF/test_images/SnakeCLEF2022-large_size \
#               --model resnet50 \
#               --img-size 224 \
#               --num-classes 1572 \
#               --crop-pct 1 \
#               --apex-amp \
#               --checkpoint /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/resnet50_baseline_apex/20220329-115540-resnet50-224/model_best.pth.tar

# python test.py /home/data2/CZP/SnakeCLEF/test_images/SnakeCLEF2022-large_size \
#               --model swin_large_patch4_window12_384 \
#               --img-size 384 \
#               --num-classes 1572 \
#               --crop-pct 1 \
#               --apex-amp \
#               --checkpoint /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/swin_large_patch4_window12_384_freeze_2_plateau/20220330-231754-swin_large_patch4_window12_384-384/Best_f1_score.pth.tar

# python test_top5.py /home/data2/CZP/SnakeCLEF/test_images/SnakeCLEF2022-large_size \
#               --model tf_efficientnet_b7_ns \
#               --img-size 600 \
#               --num-classes 1572 \
#               --crop-pct 1 \
#               --apex-amp \
#               --name efficientnet_b7_top5_val.csv \
#               -b 1 \
#               --checkpoint /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/tf_efficient_imgsize600_freeze_2_resume/20220413-174356-tf_efficientnet_b7_ns-600/Best_f1_score.pth.tar

# python test_fusion.py /home/data2/CZP/SnakeCLEF/test_images/SnakeCLEF2022-large_size \
#               --model swin_large_patch4_window12_384 \
#               --img-size 384 \
#               --num-classes 1572 \
#               --crop-pct 1 \
#               --apex-amp \
#               --name swint_epoch21_crop.csv \
#               --checkpoint /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/swin_large_patch4_window12_384_all_data_ra_cutmix/20220419-212108-swin_large_patch4_window12_384-384/last.pth.tar

# python test_fusion.py /home/data2/CZP/SnakeCLEF/test_images/SnakeCLEF2022-large_size \
#               --model tf_efficientnet_b7_ns \
#               --img-size 600 \
#               --num-classes 1572 \
#               --crop-pct 1 \
#               --apex-amp \
#               --name FocalLoss_crop.csv \
#               --checkpoint /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/tf_efficient_imgsize600_freeze_2_FocalLoss/20220419-015110-tf_efficientnet_b7_ns-600/Best_f1_score.pth.tar

# python test_fusion.py /home/data2/CZP/SnakeCLEF/test_images/SnakeCLEF2022-large_size \
#               --model tf_efficientnet_b7_ns \
#               --img-size 600 \
#               --num-classes 1572 \
#               --crop-pct 1 \
#               --apex-amp \
#               --name SeesawLoss_crop.csv \
#               --checkpoint /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/tf_efficient_imgsize600_freeze_2_SeesawLoss/20220419-020621-tf_efficientnet_b7_ns-600/Best_f1_score.pth.tar
# 模型融合
# python test_fusion.py /home/data2/CZP/SnakeCLEF/test_images/SnakeCLEF2022-large_size \
#               --model tf_efficientnet_b7_ns \
#               --img-size 600 \
#               --num-classes 1572 \
#               --crop-pct 1 \
#               --apex-amp \
#               --name ./output/efficientb7_alldata \
#               --checkpoint /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/tf_efficient_imgsize600_freeze_2_all_data/20220414-231415-tf_efficientnet_b7_ns-600/checkpoint-24.pth.tar

# python test_fusion.py /home/data2/CZP/SnakeCLEF/test_images/SnakeCLEF2022-large_size \
#               --model swin_large_patch4_window12_384 \
#               --img-size 384 \
#               --num-classes 1572 \
#               --crop-pct 1 \
#               --apex-amp \
            #   --name ./output/SwinT_alldata \
            #   --checkpoint /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/swin_large_patch4_window12_384_all_data_ra_cutmix/20220419-212108-swin_large_patch4_window12_384-384/last.pth.tar

# python test_fusion.py /home/data2/CZP/SnakeCLEF/test_images/SnakeCLEF2022-large_size \
#               --model swin_large_patch4_window12_384 \
#               --img-size 384 \
#               --num-classes 1572 \
#               --crop-pct 1 \
#               --apex-amp \
#               --name ./output/SwinT_test \
#               --checkpoint /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/swin_large_patch4_window12_384/20220331-222033-swin_large_patch4_window12_384-384/Best_f1_score.pth.tar

# python test_fusion.py /home/data2/CZP/SnakeCLEF/test_images/SnakeCLEF2022-large_size \
#               --model resnest269e \
#               --img-size 416 \
#               --num-classes 1572 \
#               --crop-pct 1 \
#               --apex-amp \
#               --name ./output/resnest269e_f5 \
#               --checkpoint /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/V100/resnest269eBest_f1_score.pth.tar

# python test_fusion.py /home/data2/CZP/SnakeCLEF/test_images/SnakeCLEF2022-large_size \
#               --model convnext_large_384_in22ft1k \
#               --img-size 384 \
#               --num-classes 1572 \
#               --crop-pct 1 \
#               --apex-amp \
#               --name ./output/convnext_large_f1 \
#               --checkpoint /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/V100/WorkSpackRecord/convnext_large_384_in22ft1k_freeze_1/20220419-005039-convnext_large_384_in22ft1k-384/model_best.pth.tar


python test_fusion.py /home/data2/CZP/SnakeCLEF/test_images/SnakeCLEF2022-large_size \
              --model resnest269e \
              --img-size 416 \
              --num-classes 1572 \
              --crop-pct 1 \
              --apex-amp \
              --name ./output/resnest269e_cutmix_epoch_7 \
              --checkpoint /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/resnest269e_all_data_cutmix/20220506-223408-resnest269e-416/checkpoint-7.pth.tar