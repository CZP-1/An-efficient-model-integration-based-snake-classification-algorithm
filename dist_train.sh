#!/bin/bash
# python3 -m torch.distributed.launch --nproc_per_node=2 train_f1.py \
#         /home3/CZP/Data/SnakeCLEF/SnakeCLEF2022-large_size/1990
#         -c configs/tf_efficientnet_l2_ns.yaml \
#         --freeze-layer 3 \
#         --lr 0.01 \
#         --sync-bn \
#         --output /home3/CZP/WorkSpackRecord/tf_efficientnet_l2_ns_freeze_3
# --rdzv_endpoint=$HOST_NODE_ADDR
# python3 -m torch.distributed.launch --nproc_per_node=2 train_f1.py /home/data2/CZP/SnakeCLEF/SnakeCLEF2022-large_size/1990 -c configs/tf_efficientnet_b7_ns_multistep.yaml \
#         --freeze-layer 2 \
#         --lr 0.01 \            
#         --sync-bn \
#         --output /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/tf_efficientnet_b7_ns_data_crop_data
torchrun --nproc_per_node=2 --rdzv_backend=c10d  train_dist.py /home/data2/CZP/SnakeCLEF/SnakeCLEF2022-large_size/1990 \
        -c configs/tf_efficientnet_b7_ns_multistep.yaml \
        --freeze-layer 2 \
        --lr 0.01 \
        -b 16 \
        --sync-bn \
        --output /home/data4/czp/FGVC2022_Snake/WorkspaceRecord/tf_efficientnet_b7_ns_data_crop_data