import csv
import os
from pathlib import Path


filename_train1='./csv2.0/train_data_map.csv'
filename_train2='./csv2.0/train_data.csv'
filename_val1='./csv2.0/val_data_map.csv'
filename_val2='./csv2.0/val_data.csv'
filename_test1='./csv2.0/test_data_map.csv'
filename_test2='./csv2.0/test_data.csv'


def generate_label(filename1,filename2):
    with open(filename2,'w',encoding='utf-8',newline='') as wt:
        writer=csv.writer(wt)
        with open(filename1,'r',encoding='utf-8')as fp:
            reader = csv.reader(fp)
            cnt=0
            tmp=''
            for data in reader:
                if(data[0]!=tmp):
                    cnt+=1
                    tmp=data[0]
                data.append(cnt)
                writer.writerow(data)
            print(cnt)

            

generate_label(filename_train1,filename_train2)
generate_label(filename_val1,filename_val2)
generate_label(filename_test1,filename_test2)