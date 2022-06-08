# from turtle import position
import pandas as pd
import numpy as np
import re
def main():
    top5 = pd.read_csv('efficientnet_b7_top5_val.csv')
    top1 = pd.read_csv('new_val.csv',usecols=['class_id'])
    top5 = pd.concat([top5,top1],axis=1,ignore_index=True)
    top5.columns=['path_img','class_id5','class_id']
    top5['position'] = -1
    for i in range(len(top5)):
        
        list_class = top5.loc[i,'class_id5']
        list_class = list_class[1:-1]
        list_class = re.split('\s+',list_class)
        while '' in list_class:
            list_class.remove('')
        # list_class = list_class.replace(' ',',')
        # list_class = ast.literal_eval(list_class)
        if top5.loc[i,'path_img']=='1996/Lycodonomorphus_rufulus/119484254.jpeg':
            print(list_class)

        top1 = str(top5.loc[i,'class_id'])
        if top1 in list_class:
            top5.loc[i,'position'] = list_class.index(top1) + 1
    top5.to_csv('./output/efficientnet_b7_top5_1.csv')

def main2():
    data = pd.read_csv('./output/efficientnet_b7_top5_1.csv')
    top_ana = pd.DataFrame(columns=['class_id','nums','position']) 
    idx=0
    for i in range(5):
        topk = data[(data['position']==i+1)]
        classid = topk['class_id'].unique().tolist()
        for cls in classid:
            len_cls = len(topk[(topk['class_id'])==cls])
            top_ana.loc[idx,'class_id'] = cls
            top_ana.loc[idx,'nums'] = len_cls
            top_ana.loc[idx,'position'] = i+1
            idx+=1
    top_ana.to_csv('./output/top_ana.csv')

def main3():
    data = pd.read_csv('./output/efficientnet_b7_top5_1.csv')
    class_ids = data['class_id'].unique().tolist()
    class_top = pd.DataFrame(columns=['class_id','num_top1','num_top2','num_top3','num_top4','num_top5','num_error'])
    idx=0
    for class_id in class_ids:
        class_data = data[(data['class_id']==class_id)]
        class_top.loc[idx,'class_id'] = class_id
        class_top.loc[idx,'num_top1'] = len(class_data[(class_data['position']==1)])
        class_top.loc[idx,'num_top2'] = len(class_data[(class_data['position']==2)])
        class_top.loc[idx,'num_top3'] = len(class_data[(class_data['position']==3)])
        class_top.loc[idx,'num_top4'] = len(class_data[(class_data['position']==4)])
        class_top.loc[idx,'num_top5'] = len(class_data[(class_data['position']==5)])
        class_top.loc[idx,'num_error'] = len(class_data[(class_data['position']==-1)])
        idx+=1
    class_top.to_csv('./output/class_top.csv')

def main4():
    class_top = pd.read_csv('./output/class_top.csv')
    # over0_5 = []
    over_1 = []
    for i in range(1572):
        if class_top.loc[i,'num_top2']>1*class_top.loc[i,'num_top1']:
            over_1.append(class_top.loc[i,'class_id'])
    print(over_1)

    data_soft = pd.read_csv('output/efficientb7_alldata_soft.csv')
    # data_soft['class_id'] = data_soft.T.apply(lambda x: x.nlargest(2).idxmin())
    data_soft['class_id'] = data_soft.columns[data_soft.values.argsort(1)[:, -2]]
    data_result = pd.read_csv('output/efficientb7_alldata.csv')
    out = data_soft.apply(lambda x: x.sort_values().unique()[-2], axis=1)
    # 构建dataframe新的列
    data_soft['value_top2'] = out
    count = 0
    for i in range(len(data_result)):
        if data_soft.loc[i,'class_id'] in over_1:
            data_result.loc[i,'class_id'] = data_soft.loc[i,'class_id']
            count+=1
    print(count)
    data_result.to_csv('./output/efficientb7_alldata_top2.csv')

if __name__ == '__main__':
    # main()
    # main2()
    # main3()
    main4()