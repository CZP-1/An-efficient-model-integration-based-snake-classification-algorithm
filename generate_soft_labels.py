from __future__ import print_function
import os
# from dataset import Dataset
import torch
from torch.utils import data
import torch.nn.functional as F
# from models import *
import torchvision
import torch
import numpy as np
import random
import time

import torchvision.transforms as transforms
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from collections import defaultdict
import json
import argparse
from torchvision.datasets import ImageFolder

# from config.args import args
# from utils import load_model

import pandas as pd

from net import Network
from config import cfg, update_config
from dataset import *
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from scipy.special import softmax
from core.evaluate import FusionMatrix
from data_transform.randaugment import RandAugmentMC

def seed_reproducer(seed=6):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2019)
        Radnom seed.

    Example
    -------
    seed_reproducer(seed=2019).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = argparse.ArgumentParser(description='Train a network for ...')
    # ===============BBN
    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        # required=True,
        required=False,
        # default="configs/cifar10.yaml",
        default="configs/fgvc8_Semi.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # ==============================1/5 数据 | Datasets==============================
    parser.add_argument('--dataset', default='fgvc', type=str, help='dataset name')
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--image-size', default=600, type=int, metavar='N',
                        help='the train image size')
    parser.add_argument('--cropsize', default=448, type=int, metavar='N',
                        help='the train image size')
    parser.add_argument('--train-batch-size', default=64, type=int, metavar='N',
                        help='train batchsize (default: 256)')
    parser.add_argument('--test-batch-size', default=32, type=int, metavar='N',
                        help='test batchsize (default: 200)')
    parser.add_argument('--mixmethod', default='baseline', type=str, metavar='N',
                        choices=['baseline', 'cutmix', 'cutout', 'mixup', 'snapmix'],
                        help='mixmethod (default: "baseline")')
    parser.add_argument('--prob', type=float, default=1.0, help='')
    parser.add_argument('--beta', type=float, default=1.0, help='')
    parser.add_argument('--fold-num', default=5, type=int, metavar='N',
                        help='train batchsize (default: 256)')
    # ==============================2/5 模型 | model==============================
    parser.add_argument('--backbone', default='tf_efficientnet_b7_ns', type=str, metavar='PATH',
                        choices=['resnet50', 'se_resnet152', 'Senet154'],
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--num-classes', default=810, type=int, metavar='N',
                        help='number of classfication of image')
    parser.add_argument('--metric', default='linear', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--concatPooling', default=False, type=bool, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # ==============================3/5 损失函数 | Optimization options==============================
    parser.add_argument('--loss', default='CrossEntropyLoss',
                        choices=['CrossEntropyLoss', 'focal_loss'], metavar='N',
                        # help='loss (default='CrossEntropyLoss')')
                        )
    # ==============================4/5 优化器 | Optimization options==============================
    parser.add_argument('--optimizer', default='sgd',
                        choices=['sgd', 'rmsprop', 'adam', 'AdaBound', 'radam'], metavar='N',
                        help='optimizer (default=sgd)')
    parser.add_argument('--lr', '--learning-rate', default=0.025, type=float,
                        metavar='LR', help='initial learning rate，1e-2， 1e-4, 0.001')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--beta1', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--beta2', default=0.999, type=float, metavar='M',
                        help='momentum')
    # ==============================5/5 训练 | train==============================
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--parallel', default=False, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # Checkpoints
    parser.add_argument('-c', '--checkpoints_path', default='checkpoints', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--checkpoints_interval', default=10, type=int, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('-k', '--bestmodel_path', default='checkpoints', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default=False, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--out', default='/home/user/Disk1/WorkSpaceRecord/FGVC8/FGVC8_Semi/resnet50',
                        help='directory to output the result')
    # Miscs
    parser.add_argument('--Seed', default=1, type=int, help='manual seed')
    # Device options
    # parser.add_argument('--gpu-id', default='0, 1, 2, 3', type=str,
    # #                     help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--use_gpu', default=True, type=str, help='')

    args = parser.parse_args()
    update_config(cfg, args)

    # --------------
    seed_reproducer(seed=6)

    device = torch.device("cuda")

    # Load data
    total_l_data_json_path='/home-ustc/ch20/WorkSpace/FGVC8_Semi/BBN-master_cv/jsons/converted_fgvc8_Semi_l_total.json'
    with open(total_l_data_json_path, "r") as f:
        all_info = json.load(f)
    total_l_data = pd.DataFrame([i['fpath'] for i in all_info["annotations"]], columns=['fpath'])

    test_data_json_path = '/home-ustc/ch20/WorkSpace/FGVC8_Semi/BBN-master_cv/jsons/converted_fgvc8_Semi_test.json'
    with open(test_data_json_path, "r") as f:
        all_info = json.load(f)
    test_data = pd.DataFrame([i['fpath'] for i in all_info["annotations"]], columns=['fpath'])

    model = Network(cfg, mode="test", num_classes=810)

    if args.parallel:
        model = DataParallel(model)

    submission = []
    PATH = [
        "/home-ustc/ch20/WorkSpaceRecord/FGVC8_Semi/BBN-master_cv/fold0/BBN.fgvc8_Semi.effb7.100epoch/models/best_model.pth",
        "/home-ustc/ch20/WorkSpaceRecord/FGVC8_Semi/BBN-master_cv/fold1/BBN.fgvc8_Semi.effb7.100epoch/models/best_model.pth",
        "/home-ustc/ch20/WorkSpaceRecord/FGVC8_Semi/BBN-master_cv/fold2/BBN.fgvc8_Semi.effb7.100epoch/models/best_model.pth",
        "/home-ustc/ch20/WorkSpaceRecord/FGVC8_Semi/BBN-master_cv/fold3/BBN.fgvc8_Semi.effb7.100epoch/models/best_model.pth",
        "/home-ustc/ch20/WorkSpaceRecord/FGVC8_Semi/BBN-master_cv/fold4/BBN.fgvc8_Semi.effb7.100epoch/models/best_model.pth",
    ]

    folds = [0,1,2,3,4]

    train_data_cp = []
    for fold_i in folds:

        update_para = ['DATASET.TRAIN_JSON', './jsons/fold'+str(fold_i)+'_train.json',
                       'DATASET.VALID_JSON', './jsons/fold'+str(fold_i)+'_val.json',
                       'OUTPUT_DIR', '/home-ustc/ch20/WorkSpaceRecord/FGVC8_Semi/BBN-master_cv/fold'+str(fold_i)+'']
        cfg.merge_from_list(update_para)

        train_set = eval(cfg.DATASET.DATASET)("train", cfg)
        valid_set = eval(cfg.DATASET.DATASET)("valid", cfg)
        val_data_cp = pd.DataFrame([ i['fpath']  for i in valid_set.data ],columns=['fpath'])

        val_dataloader = DataLoader(
            valid_set,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TEST.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
        )

        submission = []
        model.load_model(PATH[fold_i])
        model.to(device)
        model.eval()

        for i in range(1):
            val_preds = []
            labels = []
            with torch.no_grad():
                for image, label, meta in tqdm(val_dataloader):
                    image = image.to(device)
                    output = model(image)
                    val_preds.append(output)  # 模型输出
                    labels.append(label)

                labels = torch.cat(labels)  # 将各batch的 labels列表拼成一个列表
                val_preds = torch.cat(val_preds)  # 将各batch的 val_preds列表拼成一个列表
                submission.append(val_preds.cpu().numpy())

        submission_ensembled = 0
        for sub in submission:
            submission_ensembled += softmax(sub, axis=1) / len(submission)
        # val_data_cp.iloc[:, 1:] = submission_ensembled  # 添加到验证集的class probablity
        val_data_cp=pd.concat([val_data_cp,pd.DataFrame(submission_ensembled)], axis = 1)
        train_data_cp.append(val_data_cp)  # 添加到总的训练集list中
    soft_labels = total_l_data[["fpath"]].merge(pd.concat(train_data_cp), how="left",on="fpath")  # 按data的image_id，进行merge
    soft_labels.to_csv("soft_labels.csv", index=False)  # soft_label是各个标签的概率，是一个软标签

    # ==============================================================================================================
    # Generate Submission file
    # ==============================================================================================================

    update_para = ['DATASET.TRAIN_JSON', './jsons/converted_fgvc8_Semi_l_total.json',
                   'DATASET.VALID_JSON', './jsons/converted_fgvc8_Semi_test.json',
                   'OUTPUT_DIR', '/home-ustc/ch20/WorkSpaceRecord/FGVC8_Semi/BBN-master_cv/submission']
    cfg.merge_from_list(update_para)

    test_data_transforms = transforms.Compose([transforms.Resize((600, 600)),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.RandomResizedCrop(size=(600, 600),scale=(0.2, 1.0),ratio=(0.75, 1.333333333)),
                                               RandAugmentMC(n=2, m=10),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                               ])
    test_dataset = ImageFolder('/home-ustc/ch20/Datasets/FGVC8_Semi/test', transform=test_data_transforms)
    test_dataloader = DataLoader(test_dataset,batch_size=64,shuffle=False,num_workers=cfg.TEST.NUM_WORKERS,pin_memory=cfg.PIN_MEMORY)
    submission = []
    for path in PATH:  # 5折的模型
        model.load_model(path)
        model.to(device)
        model.eval()

        for i in range(8):
            test_preds = []
            labels = []
            with torch.no_grad():
                for image, label in tqdm(test_dataloader):
                    image = image.to(device)
                    output = model(image)
                    test_preds.append(output)  # 模型输出
                    labels.append(label)

                labels = torch.cat(labels)
                test_preds = torch.cat(test_preds)
                submission.append(test_preds.cpu().numpy())  # submission列表中共有5*8个ndarray，即40组模型输出结果

    submission_ensembled = 0
    for sub in submission:
        submission_ensembled += softmax(sub, axis=1) / len(submission) # 每组的模型输出结果通过softmax得到概率，40组加起来除以40

    submission_ensembled = torch.from_numpy(submission_ensembled)
    topk = (1, 5)
    maxk = max(topk)
    _, pred = submission_ensembled.topk(maxk, 1, True, True)
    # pred = pred.t()  # .t() transpose
    pred = pred.data.cpu().numpy()
    result = list()
    for i in pred:
        result.append(str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + ' ' + str(i[3]) + ' ' + str(i[4]))
    result = pd.DataFrame(result)
    Id = [test_dataset.imgs[i][0].split('/')[-1] for i in range(len(test_dataset))]
    Id = pd.DataFrame(Id)
    submission = pd.concat([Id, result],axis=1)
    submission.columns = ['Id','Category']
    submission.to_csv('submission.csv',columns=['Id','Category'],index=False,header=True)
