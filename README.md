# An efficient model integration-based snake classification algorithm

## 1. Environment setting 

### 1.0. Package
* Several important packages
    - torch == 1.10.2+cu111
    - trochvision == 0.11.3+cu111
    
* Replace folder timm/ to our timm/ folder (We made some changes to the original Timm framework, such as adding TA, etc)  
    
    #### pytorch model implementation [timm](https://github.com/rwightman/pytorch-image-models)

### 1.1. Dataset
In this project, we use a large fungi's datasets from this challenge to evaluate performance:
* [Fungi2022](https://www.kaggle.com/competitions/fungiclef2022/data)

### 1.2. OS
- [x] Windows10
- [x] Ubuntu20.04
- [x] macOS (CPU only)

## 2. Train
- [x] Single GPU Training
- [x] DataParallel (single machine multi-gpus)
- [ ] DistributedDataParallel

(more information: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

### 2.1. data

Training sets and test sets are distributed with CSV labels corresponding to them.

### 2.2. configuration
you can directly modify yaml file (in ./configs/)

### 2.3. run.
take the SwinTransformer training process as an example.

1.  we train a basic model by dividing the training set and verification set 9:1
```
python train_f1.py ./matadata/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size/1990 -c configs/swin_large_384.yaml \
        --freeze-layer 2 \
        --batch-size 32 \
        --lr 0.01 \
        --decay-rate 0.9 \
        --output ./output/Swin-TF/freeze_layer_2
```

2. Add data augment and continue fine-tuning
```
python train_f1.py ./matadata/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size/1990 -c configs/swin_large_384.yaml \
        --output ./output/Swin-TF/DF20/All_aug/freeze_layer_2 \
        --initial-checkpoint ./output/Swin-TF/freeze_layer_2/20220430-123449-swin_large_patch4_window12_384-384/Best_Top1-ACC.pth.tar \
        --freeze-layer 2 \
        --lr 0.001 \
        --batch-size 32 \
        --warmup-epochs 0 \
        --cutmix 1 \
        --color-jitter 0.4 \
        --reprob 0.25 \
        --aa trivial \
        --decay-rate 0.9
```

3. Modify the loss function and continue fine-tuning
```
python train_f1.py ./matadata/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size/1990 -c configs/swin_large_384.yaml \
        --output ./output/Swin-TF/DF20/new_loss/freeze_layer_2 \
        --initial-checkpoint ./output/Swin-TF/DF20/All_aug/freeze_layer_2/20220430-123449-swin_large_patch4_window12_384-384/Best_Top1-ACC.pth.tar \
        --freeze-layer 2 \
        --lr 0.001 \
        --batch-size 32 \
        --warmup-epochs 0 \
        --cutmix 1 \
        --color-jitter 0.4 \
        --reprob 0.25 \
        --aa trivial \
        --decay-rate 0.9 \
        --Focalloss
```

4. Fine-tuning with full datasets
```
python train_f1.py ./matadata/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size/1990 -c configs/swin_large_384.yaml \
         --batch-size 32 \
         --img-size 384 \
         --output ./output/Swin-TF/DF20/All_data/swin_large_384 \
         --freeze-layer 2 \
         --initial-checkpoint ./output/Swin-TF/DF20/new_loss/freeze_layer_2/20220502-114033-swin_large_patch4_window12_384-384/Best_Top1-ACC.pth.tar \
         --lr 0.001 \
         --cutmix 1 \
         --color-jitter 0.4 \
         --reprob 0.25 \
         --aa trivial \
         --decay-rate 0.1 \
         --warmup-epochs 0 \
         --epochs 24 \
         --sched multistep \
         --checkpoint-hist 24 \
         --Focalloss
```


### 2.4. multi-gpus
```
sh dist_train.sh
```  

## 3. Evaluation
for details, see test.sh
```
sh test.sh
```

## 4. Model Ensemble
```
python tools/model_confusion.py
```
You need to download the logits from 5 and run the command python tools/model_confusion.py to get the result showing in the leaderboard.

## 5. model logits
It can be downloaded from Google Cloud Disk: https://drive.google.com/file/d/1vudXIAVEUJekXQfhgd_32HhTzxmDUxpr/view?usp=sharing

  
It can be directly used for model ensemble reasoning.

### Acknowledgment

* Thanks to [timm](https://github.com/rwightman/pytorch-image-models) for Pytorch implementation.
