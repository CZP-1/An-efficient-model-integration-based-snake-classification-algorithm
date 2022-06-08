# # SAR mobilenetv3_large_100_miil_chans1
python test_mobilenet.py /home/data3/changhao/Datasets/PBVS2022/images_SAR/NTIRE2021_Class_valid_images_SAR \
              --model mobilenetv3_large_100_miil \
              --img-size 224 \
              --num-classes 10 \
              --checkpoint /home/data3/changhao/WorkSpaceRecord/PBVS2022_timm/mobilenetv3_large_100_miil_chans1/20220228-145646-mobilenetv3_large_100_miil-224/model_best.pth.tar