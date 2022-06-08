python validate.py --data /home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size/1990 \
        --checkpoint /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/tf_efficient_imgsize600_freeze_2_resume/20220413-174356-tf_efficientnet_b7_ns-600/Best_f1_score.pth.tar \
        --model tf_efficientnet_b7_ns \
        --img-size 600 \
        --crop-pct 1.0 \
        --num-classes 1572 \
        --batch-size 4 \
        --apex-amp 
