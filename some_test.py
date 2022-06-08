from random import sample
import cv2
import pandas as pd
import numpy

def main():
    data_test = pd.read_csv("/home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-TestMetadata.csv",header=0,usecols=['observation_id','file_path'])
    observation_ids = data_test['observation_id'].unique().tolist()
    test = pd.DataFrame()
    for observation_id in observation_ids:
        data  = data_test[(data_test['observation_id']==observation_id)]
        sample = data.sample(1)

        # 保存
        test = pd.concat([test, sample], ignore_index=True)
        # print(1)
    test.loc[:, ['file_path','observation_id']] = test.loc[:, ['observation_id','file_path']].to_numpy()
    test.columns = ['file_path','observation_id']
    print(observation_id)
def test():
    data_test = pd.read_csv("/home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-TestMetadata.csv",header=0,usecols=['observation_id','file_path'])
    # for i in range(len(data_test)):
    #     data_test.observation_id[i] = int(data_test.observation_id[i])
    data_test.sort_values(by='observation_id',axis=0,inplace=True)
    print("yes")
    
def test1():
    test = pd.read_csv("/home/data4/czp/FGVC2022_Snake/pytorch-image-models/results.csv")
    test.sort_values(by='ObservationId',axis=0,inplace=True)
    test.drop_duplicates('ObservationId',keep='last',inplace=True)
    test.to_csv("submition.csv",columns=['ObservationId', 'class_id'], index=False, header=True)
    print(test)

def test2():
    submition = pd.read_csv("/home/data4/czp/FGVC2022_Snake/pytorch-image-models/submition.csv")
    data_test = pd.read_csv("/home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-TestMetadata.csv")
    data_train = pd.read_csv("/home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-TrainMetadata.csv")
    countrys_trian = data_train['country'].unique().tolist()
    
    # index=0
    # country_class = pd.DataFrame(columns=['country','codes','endemic','classid'])
    # for country in countrys_trian:
    #     data_country = data_train[(data_train['country']==country)] 
    #     if len(data_country['class_id'].unique())==1:
    #         country_class.loc[index,'country'] = country
    #         country_class.loc[index,'codes'] = data_country['code'].unique().tolist()
    #         country_class.loc[index,'endemic'] = data_country['endemic'].unique().tolist()[0]
    #         country_class.loc[index,'classid'] = data_country['class_id'].unique().tolist()[0]
    #         index = index+1
    # country_class.to_csv('country.csv')
    country_class = pd.read_csv('country.csv')
    index = 0
    count = 0
    count_num = 0
    eda_country = pd.DataFrame(columns=['country','codes','test_code','endemic','test_endemic','classid','pred_class'])
    for i in range(len(country_class)):
        country = country_class.loc[i,'country']
        sample = data_test[(data_test['country']==country)]
        if len(sample)>0:
            count_num = count_num + len(sample)
        sample.reset_index(drop=True,inplace=True)
        for j in range(len(sample)):
            eda_country.loc[index,'country'] = country
            eda_country.loc[index,'codes'] = country_class.loc[i,'codes']
            eda_country.loc[index,'endemic'] = country_class.loc[i,'endemic']
            eda_country.loc[index,'classid'] = country_class.loc[i,'classid']
            eda_country.loc[index,'test_code'] = sample['code'].unique().tolist()
            eda_country.loc[index,'test_endemic'] = sample['endemic'].unique().tolist()[0]
            ob_id = sample.loc[j,'observation_id']
            file_path = sample.loc[j,'file_path']
            pred_sample = submition[(submition['ObservationId']==ob_id)]
            eda_country.loc[index,'pred_class'] = pred_sample['class_id'].unique().tolist()[0]
            if (eda_country.loc[index,'pred_class']==eda_country.loc[index,'classid']):
                count=count+1
            index = index+1

    eda_country.to_csv('country_eda.csv')
    print(count)
    print(count_num)

def test3():
    
    data_train =pd.read_csv('fold0_train.csv')
    data_val =pd.read_csv('fold0_val.csv')
    classes_train  = data_train['class_id'].unique().tolist()
    classes_val  = data_val['class_id'].unique().tolist()
    for class_1 in classes_train:
        if class_1 not in classes_val:
            sample_train = data_train[(data_train['class_id']==class_1)]
            sample_val = sample_train.sample(1)
            data_val = pd.concat([data_val,sample_val], ignore_index=True)
            data_train = data_train.append(sample_val).drop_duplicates(keep=False)
    data_val.to_csv("new_val.csv")
    data_train.to_csv("new_train.csv")

def test4():
    data_train = pd.read_csv("fold0_train.csv",header=0)
    data_crop = pd.read_csv("data_crop.csv",header=0)
    file_paths = data_crop['file_path'].unique().tolist()
    for i in range(len(data_train)):
        if data_train.loc[i,'file_path'] not in file_paths:
            data_train = data_train.append(data_train.loc[[i]]).drop_duplicates(keep=False)
    data_train.to_csv('crop_train.csv')
    data_val = pd.read_csv("fold0_val.csv",header=0)
    for i in range(len(data_val)):
        if data_val.loc[i,'file_path'] not in file_paths:
            data_val = data_val.append(data_val.loc[[i]]).drop_duplicates(keep=False)
    data_val.to_csv('crop_val.csv')

def test5():
    data_train = pd.read_csv("data/newfold2_train.csv",header=0)
    # data_train = pd.read_csv("/home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-TrainMetadata.csv",header=0)
    # data_train = shuffle(data_train,random_state=0)
    # num_classes = 1572
    # data_train = pd.read_csv("train_sample50.csv",header=0)
    samples = [("/home/data2/CZP/SnakeCLEF/SnakeCLEF2022-large_size" + '/'+ data_train.file_path[index], data_train.class_id[index]) for index in range(len(data_train))]
    
    # val
    # data_val = pd.read_csv("/home/data2/CZP/SnakeCLEF/matadata/val.csv",header=0)
    data_val = pd.read_csv("data/newfold2_val.csv",header=0)
    samples_1 = [("/home/data2/CZP/SnakeCLEF/SnakeCLEF2022-large_size" + '/'+ data_val.file_path[index], data_val.class_id[index]) for index in range(len(data_val))]
    samples.extend(samples_1)
    from sklearn.utils import shuffle
    samples=shuffle(samples,random_state=5)
    print(len(samples))
if __name__ == '__main__':
    # main()
    # test()
    # test1()
    # test2()
    # test3()
    # test4()
    test5()

