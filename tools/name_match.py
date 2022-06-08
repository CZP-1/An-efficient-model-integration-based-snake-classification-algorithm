from unittest import result
import pandas as pd 
def main():
    data_train = pd.read_csv("/home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-TrainMetadata.csv")
    data_train.rename(columns={"observation_id":"ObservationId"},inplace=True)
    
    result = pd.read_csv('output/tf_efficientnet_l2_epoch9_old.csv')
    data_train=pd.concat([data_train,result],axis=0)
    data_train.sort_values(by='ObservationId',inplace=True)
    data_train.reset_index(inplace=True)
    count = 0
    count_change=0
    for i in range(1,len(data_train)-1):
        if data_train['endemic'][i]!=True and data_train['endemic'][i]!=False:
            if data_train['class_id'][i-1]==data_train['class_id'][i+1]:
                if data_train['endemic'][i-1]==True or data_train['endemic'][i-1]==False:
                    if data_train['endemic'][i+1]==True or data_train['endemic'][i+1]==False:
                        print(data_train['endemic'][i-1])
                        print(data_train['endemic'][i+1])
                        if data_train['class_id'][i+1]!=data_train['class_id'][i]:
                            count_change+=1
                            # print(data_train['ObservationId'][i])
                        index = result[(result['ObservationId']==data_train['ObservationId'][i])].index
                        result.loc[index,'class_id']=data_train['class_id'][i+1]
                        count+=1
    # for j in range(len(result)):
    #     obid=result['ObservationId'][j]
    #     print(obid)
    #     for i in range(len(data_train)-1):
    #         if data_train['observation_id'][i]<=obid and data_train['observation_id'][i+1]>=obid and data_train['class_id'][i]==data_train['class_id'][i+1]:
    #             print(data_train['observation_id'][i],data_train['observation_id'][i+1])
    #             if result['class_id'][j]!=data_train['class_id'][i]:
                    
    #                 count_change+=1
    #                 print(j)
    #             result['class_id'][j]=data_train['class_id'][i]
    #             count+=1
    print(count)
    print(count_change)
    result.to_csv('./output/name_match.csv')

if __name__=='__main__':
    main()