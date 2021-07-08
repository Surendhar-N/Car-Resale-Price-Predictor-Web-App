"""
Created on Mon Jul  5 19:04:31 2021

@author: Surendhar
"""
import joblib
import pandas as pd
import numpy as np
import re
import sys
import math
"""
#*****************************************************************************
Load Model
#*****************************************************************************
"""
filename = 'D:\ML Projects\Car Price Predictor\predictor.pkl'
loaded_model = joblib.load(filename)

n = len(sys.argv)
details = ""
for i in range(1,n):
    details = details + sys.argv[i] + " "
    
details = details[1:-2]
detail_lst = details.split(",")
"""
#*****************************************************************************
Load Value need to predict
#*****************************************************************************
"""
col_name = ['name','year','km_driven','mileage','engine','max_power','torque','seats','fuel_CNG','fuel_Diesel','fuel_LPG','fuel_Petrol','seller_type_Dealer','seller_type_Individual','seller_type_Trustmark Dealer','transmission_Automatic','transmission_Manual','owner_First Owner','owner_Fourth & Above Owner','owner_Second Owner','owner_Test Drive Car','owner_Third Owner']
cont_col = ['name','year','km_driven','mileage','engine','max_power','torque','seats']
cat_col = ['fuel','seller_type','transmission','owner']
sent_data = pd.DataFrame(columns = col_name)

for i in detail_lst:
    data = i.split(":")
    if len(data)>1:
        if data[0] in cont_col:
            if data[1] == '"':
                sent_data[data[0]] = np.nan
            else:
                sent_data[data[0]] = [data[1]]
        elif data[0] in cat_col:
            temp_col = data[0]+'_'+data[1]
            sent_data[temp_col] = [1]
    elif len(data) == 1:
        sent_data[data[0]] = np.nan
sent_data = sent_data.fillna(0)

sent_data.to_csv('D:\ML Projects\Car Price Predictor\data_rf.csv',mode = 'w', index=False)


"""
#*****************************************************************************
#Replace NaN Values
#*****************************************************************************
"""
def replace_nan(data_set,name_c,val):
    if name_c in data_set.values:
        dataframe = data_set.loc[data_set['name']==name_c]
        for ind in dataframe.index:
            ans = dataframe[val].iloc[0]
            if pd.isnull(ans):
                continue
            else:
                return ans
    else:
        ans = np.nan
    return ans


"""
#*****************************************************************************
#Renaming Suffix Values
#*****************************************************************************
"""
def suffix_rem(data_set,cols,dels):
    for ind in data_set.index:
        for i in range (0,len(cols)):
            col = cols[i]
            deli = dels[i]
            string = data_set[col][ind]
            if pd.isnull(string) or string == 0:
                car_name = data_set.name.iloc[ind]
                string = replace_nan(data_set,car_name,col)
                seat_no = replace_nan(data_set,car_name,'seats')
                if pd.isnull(string) or string == 0:
                    string = str(data_set[col].mode()[0])
                if pd.isnull(seat_no):
                    seat_no = data_set['seats'].mode()[0]
                data_set['seats'][ind] = seat_no
            else:
                string = str(data_set[col][ind])
            str_lst = re.split(deli,string,1)
            if col == 'torque' and float(str_lst[0]) < 30:
                data_set[col][ind] = float(str_lst[0]) * 9.80665
            else:
                data_set[col][ind] = str_lst[0]
            '''if str(data_set['seats'][ind]) == 'NaN':
                seats = 'seats'
                data_set[seats][ind] = replace_nan(data_set,car_name,seats)'''
    data_set.drop(labels=['name'], axis=1, inplace = True)
    return data_set

"""
#*****************************************************************************
#Process Data
#*****************************************************************************
"""
def process_data(data_set):
    cols = ['mileage','engine','max_power','torque']
    dels = ['[K k]','[c C]','[b B]','[K k N n @ (]']
    processed_data = suffix_rem(data_set,cols,dels)
    return processed_data
        
def data_type_set(data_set):
    for col in data_set.columns:
        if (data_set[col].dtype != np.number):
            data_set = data_set.astype({col:'float'})
    return data_set

"""
#*****************************************************************************
#Prediction
#*****************************************************************************
"""
processed_data = process_data(sent_data) 
final = data_type_set(processed_data)
sent_data.to_csv('D:\ML Projects\Car Price Predictor\data_rrf.csv',mode = 'w', index=False)

predict = loaded_model.predict(sent_data)


predict = int(predict)
def round_up(x):
    div_100 = int(math.ceil(x / 100))
    final_val = div_100 * 100
    return final_val

predict = round_up(predict)

print(predict)
