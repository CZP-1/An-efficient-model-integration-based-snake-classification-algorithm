import pandas as pd
import os
import glob
from math import ceil
def main():
    path_train_csv = "/home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-TrainMetadata.csv"
    path_test_csv = "/home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-TestMetadata.csv"
    path_img_root = "/home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size"
    data_train = pd.read_csv(path_train_csv,usecols=['class_id','file_path'])
    labels = data_train['class_id'].unique().tolist()
    val = pd.DataFrame()              # 划分出的test集合
    train = pd.DataFrame()
    for label in labels:
        data  = data_train[(data_train['class_id']==label)]
        sample = data.sample(ceil(0.1*len(data)))
        sample_index = sample.index

        # 剩余数据
        all_index = data.index
        residue_index = all_index.difference(sample_index) # 去除sample之后剩余的数据
        residue = data.loc[residue_index]  # 这里要使用.loc而非.iloc
        
        # 保存
        val = pd.concat([val, sample], ignore_index=True)
        train = pd.concat([train, residue], ignore_index=True)
        # print(1)

    val.to_csv('/home/data2/CZP/SnakeCLEF/matadata/val.csv')
    train.to_csv('/home/data2/CZP/SnakeCLEF/matadata/train.csv')
if __name__ == '__main__':
     main()