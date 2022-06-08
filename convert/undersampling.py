from random import sample
import cv2
import pandas as pd
import numpy
from PIL import Image
from sklearn.utils import shuffle

def main():
    
    data_train = pd.read_csv("/home/data4/czp/FGVC2022_Snake/pytorch-image-models/fold0_train.csv",usecols=['class_id','file_path'])
    classes = data_train['class_id'].unique().tolist()
    sampling_data = pd.DataFrame(columns=['class_id','file_path'])
    
    nums = 50
    for class_1 in classes :
        sample = data_train[(data_train['class_id']==class_1)]
        if len(sample)>=nums:
            sample_class = sample.sample(nums)
            sampling_data = pd.concat([sampling_data, sample_class], ignore_index=True)
        else:
            times = int(nums/len(sample))
            for i in range(times):
                sampling_data = pd.concat([sampling_data, sample], ignore_index=True)
            sampling_data = pd.concat([sampling_data, sample.head(nums-times*len(sample))], ignore_index=True)
    
    sampling_data = shuffle(sampling_data)
    sampling_data.reset_index(drop=True,inplace=True)
    sampling_data.to_csv('train_sample50.csv')
    
    print("yesyesyes")

if __name__ == '__main__':
    main()
    # test()
    # test1()
    # test2()
