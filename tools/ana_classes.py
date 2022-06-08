from stat import SF_APPEND
import pandas as pd
def main():
    confusion_matric = pd.read_csv('confusion_matrix_T.csv',header=0)
    # confusion_matric = confusion_matric.T
    # confusion_matric.to_csv('confusion_matrix_T.csv',header=0,index=False)


    nums = 5
    acc = 0.5
    with open ('ana_classes_T.txt','w') as file :
        for i in range(len(confusion_matric)):
            sum_column = confusion_matric[str(i)].sum()
            # confusion_matric.loc[1572,str(i)] = sum_column
            if (confusion_matric.loc[i,str(i)]) / sum_column < acc and sum_column > nums:
                file.write(f'类别{i}总数超过{nums}张且精度小于{acc*100}%\n')
                sample = confusion_matric[(confusion_matric[str(i)] > 0)]
                sample.sort_values(by=str(i),inplace=True,axis=0,ascending=False)
                for j in range(len(sample)):
                    index = sample.index[j]
                    num = sample.loc[index,str(i)]
                    file.write(f'\t其中{num}张被分为了{index}\n')
                
    file.close()
if __name__ == '__main__':
    main()