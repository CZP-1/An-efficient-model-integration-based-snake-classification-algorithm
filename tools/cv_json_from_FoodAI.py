import json, os,sys
from re import I
import argparse
from tqdm import tqdm

from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Train FGVC Network")

    parser.add_argument(
        "--file",
        help="json file to be converted",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--root",
        help="root path to save image",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--sp",
        help="save path for converted file ",
        type=str,
        required=False,
        default="."
    )
    parser.add_argument(
        "--total-data",
        help="save path for converted file ",
        type=bool,
        required=False,
        default=True
    )


    args = parser.parse_args()
    return args

def convert(json_file, image_root,args):

    # normal_mean = (0.5, 0.5, 0.5)
    # normal_std = (0.5, 0.5, 0.5)
    # transform = transforms.Compose([
    #     transforms.Resize((448, 448)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=normal_mean, std=normal_std)
    # ])

    # if args.total_data:
    #     l_train_dataset = FGVCSSL_ImageFolder("/home/data1/changhao/Datasets/LargeFineFoodAI/Recognition/train", transform=transform)
    #     val_dataset = FGVCSSL_ImageFolder("/home/data1/changhao/Datasets/LargeFineFoodAI/Recognition/validation", transform=transform)
    #     l_train_dataset.samples.extend(val_dataset.samples)
    #     dataset = l_train_dataset
    # else:
    #     dataset = FGVCSSL_ImageFolder(image_root, transform=transform)


    path_train_csv = "/home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-TrainMetadata.csv"
    # path_train_csv = "crop_data.csv"
    path_test_csv = "/home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-TestMetadata.csv"
    path_img_root = "/home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size"
    data_train = pd.read_csv(path_train_csv,usecols=['class_id','file_path'])
    # classid = data_train['class_id'].unique().tolist()
    # count=0
    # for i in classid :
    #     if len(data_train[(data_train['class_id']==i)])<5:
    #         count+=1
    #         sample = data_train[(data_train['class_id']==i)]
    #         data_train = pd.concat([sample,data_train],axis=0)

    # data_train.reset_index(inplace=True)
    # print(count)
    # print(len(data_train))
    img_paths = np.array(data_train['file_path'])
    labels = np.array(data_train['class_id'])
    
    # 分层k折交叉验证分割
    kf = StratifiedKFold(n_splits=5, shuffle=False).split(img_paths, labels)
    fold_num = 'all'  # 可手动控制，训练第几折
    for fold, (trn_idx, val_idx) in enumerate(kf):
        if fold == fold_num or fold_num == 'all':  # 运行某一折

            train_img_paths = np.array(img_paths)[trn_idx]
            train_labels = np.array(labels)[trn_idx]
            val_img_paths = np.array(img_paths)[val_idx]
            val_labels = np.array(labels)[val_idx]

            train_annotation_file = pd.DataFrame(train_img_paths)
            train_annotation_file['class_id'] = train_labels
            save_path = os.path.join(args.sp, "data/newfold" + str(fold)+"_train.csv")
            print("Converted, Saveing converted file to {}".format(save_path))
            train_annotation_file.to_csv(save_path)

            val_annotation_file = pd.DataFrame(val_img_paths)
            val_annotation_file['class_id'] = val_labels
            save_path = os.path.join(args.sp, "data/newfold" + str(fold) + "_val.csv")
            val_annotation_file.to_csv(save_path)
            print(len(train_annotation_file['class_id'].unique().tolist()))

            print("Converted, Saveing converted file to {}".format(save_path))

class FGVCSSL_ImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(FGVCSSL_ImageFolder, self).__init__(root,
                                          transform=transform,
                                          target_transform=target_transform)
    # 继承之后，只需要重写某一个方法就可以了，会跳转到重写的方法！！！
    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]

        # 数字字符串排序
        # classes.sort() # eg. ['1', '11', '2', '20']
        classes = sorted(classes, key=lambda class_id: (int(class_id))) # eg. ['1', '2', '11', '20']

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

if __name__ == "__main__":
    args = parse_args()
    convert(args.file, args.root,args)
    # save_path = os.path.join(args.sp, "converted_" + os.path.split(args.file)[-1])
    # print("Converted, Saveing converted file to {}".format(save_path))
    # with open(save_path, "w") as f:
    #     json.dump(converted_annos, f)

    # 运行参数设置
    # --sp /home/changhao/WorkSpace/LargeFineFoodAI/pytorch-image-models-master/jsons