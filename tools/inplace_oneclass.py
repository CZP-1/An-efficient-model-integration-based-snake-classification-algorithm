import pandas as pd
def main():
    result = pd.read_csv('epoch24_alldata_crop.csv')
    country_data = pd.read_csv('country.csv')
    countrys = country_data['country'].unique().tolist()
    test = pd.read_csv('/home/data2/CZP/SnakeCLEF/matadata/SnakeCLEF2022-TestMetadata.csv')
    count = 0
    for i in range(len(test)):
        country = test.loc[i,'country']
        if country in countrys:
            obid = test.loc[i,'observation_id']
            index_result = result[(result['ObservationId']==obid)].index
            index_country = country_data[(country_data['country']==country)].index
            if result.loc[index_result[0],'class_id'] != country_data.loc[index_country[0],'classid']:
                count = count + 1
            result.loc[index_result[0],'class_id'] = country_data.loc[index_country[0],'classid']
    result.drop(result.columns[0],axis= 1 ,inplace=True)
    print(count)
    result.to_csv('inplace.csv')
if __name__ == '__main__':
    main()