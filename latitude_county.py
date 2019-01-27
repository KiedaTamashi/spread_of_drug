import os
import pandas as pd

def read_file(file_path):
    with open(file_path,"r") as f:
        list_origin = f.readlines()
        list_origin = list_origin[87:]
        new_list = list()
        for i in range(0, len(list_origin)):
            #KY =18 , OH = 36, PA =39 , VA =48 , WV =50
            list_origin[i] = list_origin[i].rstrip('\n')
            pair = list_origin[i].split(",")
            list_origin[i] = pair[1],pair[3].replace("\'",""),pair[4],pair[5].rstrip(')')
            num = pair[1]

            if (num == '18') or (num == '36')or(num =='39')or(num =='48')or(num == '50'):
                # print(num)
                new_list.append(list_origin[i])

        print(new_list[0])
        return new_list

def read_target(file_path,list):
    df = pd.read_excel(file_path)
    for indexs in df.index:
        state = df.iloc[indexs]['State']
        if state == 'KY':
            sno = '18'
        elif state == 'OH':
            sno = '36'
        elif state == 'PA':
            sno = '39'
        elif state == 'VA':
            sno = '48'
        elif state == 'WV':
            sno = '50'
        county = df.iloc[indexs]['COUNTY']
        LATITUDE = 0
        LONGITUDE = 0
        count = 0
        for item in list:
            if item[0]==sno and item[1].lower()==county.lower():
                LATITUDE += float(item[2])
                LONGITUDE += float(item[3])
                count+=1
        if count !=0:
            LATITUDE /= count
            LONGITUDE /= count
        df.iloc[indexs,5] = LATITUDE
        df.iloc[indexs,6] = LONGITUDE
    df.to_csv('test1.csv')

def read_target2(file_path,list):
    df = pd.read_excel(file_path)
    for indexs in df.index:
        state = df.iloc[indexs]['State']
        if state == 'KY':
            sno = '18'
        elif state == 'OH':
            sno = '36'
        elif state == 'PA':
            sno = '39'
        elif state == 'VA':
            sno = '48'
        elif state == 'WV':
            sno = '50'
        county = df.iloc[indexs]['COUNTY']
        LATITUDE = 0
        LONGITUDE = 0
        count = 0
        for item in list:
            if item[0]==sno and item[1].lower()==county.lower():
                LATITUDE += float(item[2])
                LONGITUDE += float(item[3])
                count+=1
        if count !=0:
            LATITUDE /= count
            LONGITUDE /= count
        df.iloc[indexs,5] = LATITUDE
        df.iloc[indexs,6] = LONGITUDE
    df.to_csv('test2.csv')




list_where2find = read_file("./us_cities.sql")
read_target('./county.xlsx',list_where2find)
