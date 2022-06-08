import pandas as pd
def main1():
    x1 = 1
    x2 = 1
    x3 = 1
    x4 = 1
    x5 = 1
    # x2 = 1
    result_path = './output/fold5_efficientnetb7.csv'

    result_1 = pd.read_csv('output/efficientb7_f2_fold0_soft.csv')
    result_2 = pd.read_csv('output/efficientb7_f2_fold1_soft.csv')
    result_3 = pd.read_csv('output/efficientb7_f2_fold2_soft.csv')
    result_4 = pd.read_csv('output/efficientb7_f2_fold3_soft.csv')
    result_5 = pd.read_csv('output/efficientb7_f2_fold4_soft.csv')
    # result_2 = pd.read_csv('output/SwinT_alldata_soft.csv')
    result = pd.DataFrame(result_2.ObservationId)

    result['class_id']=0
    for i in range(len(result_1)):
        max_class = 1e-10
        for j in range(1572):
            value = x1*result_1.loc[i,str(j)] + x2*result_2.loc[i,str(j)] + x3*result_3.loc[i,str(j)] + x4*result_4.loc[i,str(j)] +x5*result_5.loc[i,str(j)]
            if max_class < value:
                max_class = value
                class_id = j
        result.loc[i,'class_id'] = class_id
    result.to_csv(result_path, columns=['ObservationId', 'class_id'], index=False, header=True)

def main2():
    x1 = 0.15
    x2 = 6
    x3 = 4 
    x4 = 6
    x5 = 0.01
    # x2 = 1
    # result_path = './output/l2_beit_6p53p5.csv'
    result_path = './output1/swint_b7_beit_l2_0p15646.csv'
#swint b7 beit l2
    result_1 = pd.read_csv('output/SwinT_alldata_soft.csv')
    result_2 = pd.read_csv('output/efficientb7_alldata_soft.csv')
    result_3 = pd.read_csv('output/beit_large_patch16_512_epoch37_soft.csv')
    result_4 = pd.read_csv('output/tf_efficientnet_l2_epoch12_new_soft.csv')
    # result_4 = pd.read_csv('output/efficientb7_f2_fold3_soft.csv')
    # result_5 = pd.read_csv('output/resnest269e_epoch_30_soft.csv')
    # result_2 = pd.read_csv('output/SwinT_alldata_soft.csv')
    result = pd.DataFrame(result_2.ObservationId)
    # result['class_id']=0
    
    result_soft = x1*result_1 + x2*result_2 + x3*result_3 + x4*result_4 
    result_soft.drop(columns='ObservationId',inplace=True)
    result_soft['class_id'] = result_soft.idxmax(axis=1)
    result_soft = pd.concat([result,result_soft],axis=1)
    result_soft.to_csv(result_path, columns=['ObservationId', 'class_id'], index=False, header=True)
if __name__ =='__main__':
    # main1()
    main2()


