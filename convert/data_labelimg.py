from random import sample
import cv2
import pandas as pd
import numpy
from PIL import Image

def main():
    
    data_train = pd.read_csv("/home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-TrainMetadata.csv",usecols=['class_id','file_path'])
    classes = data_train['class_id'].unique().tolist()
    count = 0
    for class_1 in classes :
        sample = data_train[(data_train['class_id']==class_1)]
        img_list = sample['file_path'].unique().tolist()
        if len(img_list)>=5:
            for i in range(5):
                path_image = "/home/data2/CZP/SnakeCLEF/SnakeCLEF2022-large_size/" + img_list[i]
                image = Image.open(path_image)
                if path_image.split('.')[-1=='png']:
                    image = image.convert('RGB') 
                path_save = '/home/data2/CZP/SnakeCLEF/data_detection/'+img_list[i].split('.')[0].split('/')[-1]+'.jpg'
                image.save(path_save)
                count = count + 1
        else :
            for i in range(len(img_list)):
                path_image = "/home/data2/CZP/SnakeCLEF/SnakeCLEF2022-large_size/" + img_list[i]
                image = Image.open(path_image)
                if path_image.split('.')[-1=='png']:
                    image = image.convert('RGB') 
                path_save = '/home/data2/CZP/SnakeCLEF/data_detection/'+img_list[i].split('.')[0].split('/')[-1]+'.jpg'
                image.save(path_save)
                count = count + 1
    print(count)

if __name__ == '__main__':
    main()
    # test()
    # test1()
    # test2()
