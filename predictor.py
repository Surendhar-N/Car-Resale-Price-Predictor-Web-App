# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 16:13:35 2021

@author: Surendhar
"""


"""
*****************************************************************************
Import necessary libraries and packages 
*****************************************************************************
"""
import pandas as pd
import numpy as np
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib


"""
*****************************************************************************
Load Data
*****************************************************************************
"""
train_data = pd.read_csv('D:\ML Projects\Car Price Predictor\\Car details v3.csv')
pd.set_option('display.max_columns',None)

sent_file = open('D:\ML Projects\Car Price Predictor\\output.json')
sent_json = json.load(sent_file)
col_name = ['name','year','km_driven','mileage','engine','max_power','torque','seats','fuel_CNG','fuel_Diesel','fuel_LPG','fuel_Petrol','seller_type_Dealer','seller_type_Individual','seller_type_Trustmark Dealer','transmission_Automatic','transmission_Manual','owner_First Owner','owner_Fourth & Above Owner','owner_Second Owner','owner_Test Drive Car','owner_Third Owner']
cont_col = ['name','year','km_driven','mileage','engine','max_power','torque','seats']
cat_col = ['fuel','seller','transmission','owner']
sent_data = pd.DataFrame(columns = col_name)
for key,value in sent_json.items():
    if key in cont_col:
        sent_data[key] = [value]
    elif key in cat_col:
        temp_col = key+'_'+value
        sent_data[temp_col] = 1
sent_data = sent_data.fillna(0)

    
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
            if pd.isnull(string):
                car_name = data_set.name.iloc[ind]
                string = replace_nan(train_data,car_name,col)
                seat_no = replace_nan(train_data,car_name,'seats')
                if pd.isnull(string):
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
    #car_name = data_set.name
    #data_set[['Maker','Model']] = car_name.str.split(' ',1,expand=True)
    data_set.drop(labels=['name'], axis=1, inplace = True)
    return data_set

"""
#*****************************************************************************
#Encoding
#*****************************************************************************
"""

def cat_encoder(data_set,cols):
    get_dum = pd.get_dummies(data_set, columns=cols)
    return get_dum

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

def encoder(data_set):
    #Encoding Data
    cols = ['fuel','seller_type','transmission','owner']
    encoded_data = cat_encoder(data_set,cols)
    return encoded_data
        
def data_type_set(data_set):
    for col in data_set.columns:
        if (data_set[col].dtype != np.number):
            data_set = data_set.astype({col:'float'})
    return data_set

"""
#*****************************************************************************
#Data Split
#*****************************************************************************
"""
processed_data = process_data(train_data) 
encoded_data = encoder(processed_data)
final = data_type_set(encoded_data)
X = final.loc[:, final.columns != 'selling_price']
y = final.loc[:,'selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

"""
#*****************************************************************************
#Model Preparation
#*****************************************************************************
"""


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)
model = rf_random.best_estimator_


"""
#*****************************************************************************
Save Model
#*****************************************************************************
"""

joblib.dump(model, 'predictor.pkl')
"""
#*****************************************************************************
New Model
#*****************************************************************************
"""

processed_data = process_data(sent_data) 
final = data_type_set(processed_data)
predict = model.predict(final)

sent_data['prediction'] = predict
sent_data.to_csv('result_rf.csv',mode = 'w', index=False)


