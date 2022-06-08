import pandas as pd

def count_label():
    path_train = "/home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-TrainMetadata.csv"
    path_test = "/home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-TestMetadata.csv"
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)
    Observationids = train['observation_id'].unique().tolist()
    classids = train['class_id'].unique().tolist()
    Observationids_test = test['observation_id'].unique().tolist()
    binomial_names = train['binomial_name'].unique().tolist()
    countrys = train['country'].unique().tolist()
    codes = train['code'].unique().tolist()
    endemics = train['endemic'].unique().tolist()

    countrys_test = test['country'].unique().tolist()
    codes_test = test['code'].unique().tolist()

    # class_img = pd.DataFrame(columns=['class_ids','observation_ids'])
    # for i,classes in enumerate(classids) :
    #     sample = train[(train['class_id']==classes)]
    #     class_img.loc[i,0] = len(sample)
    #     class_img.loc[i,1] = len(sample['observation_id'].unique())
    # class_img.to_csv("class_img.csv")

    # obid_img = pd.DataFrame(columns=['observation_ids'])
    # for i,Observationid in enumerate(Observationids) :
    #     sample = train[(train['observation_id']==Observationid)]
    #     obid_img.loc[i,0] = len(sample)
    # obid_img.to_csv("obid_img.csv")     

    # country_ana = pd.DataFrame(columns=['codes','classes','observation_ids'])
    # for i,country in enumerate(countrys) :
    #     sample = train[(train['country']==country)]
    #     sample_code = sample['code'].unique()
    #     # code_list=[]
    #     # for t in range(len(sample_code)):
    #     #     # print(country)
    #     #     code_list.append()
    #     #     print(f"{country} matches codes:")
    #     sample_class = sample['class_id'].unique()
    #     sample_obids = sample['observation_id'].unique()
    #     country_ana.loc[i,'codes'] = sample_code
    #     country_ana.loc[i,'classes'] = len(sample_class)
    #     country_ana.loc[i,'observation_ids'] = len(sample_obids)
    # country_ana.to_csv("country_ana.csv")

    # code_ana = pd.DataFrame(columns=['countrys','classes','observation_ids'])
    # for i,code in enumerate(codes):
    #     sample = train[(train['code']==code)]
    #     sample_country = sample['country'].unique()
    #     sample_class = sample['class_id'].unique()
    #     sample_obids = sample['observation_id'].unique()
    #     code_ana.loc[i,'countrys'] = len(sample_country)
    #     code_ana.loc[i,'classes'] = len(sample_class)
    #     code_ana.loc[i,'observation_ids'] = len(sample_obids)
    # code_ana.to_csv("code_ana.csv")






if __name__ == '__main__':
    
    count_label()