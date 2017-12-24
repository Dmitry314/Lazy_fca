#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 18:02:26 2017

@author: dmitriy
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def load_data(n = 1):

    name_train = 'train' + str(n) + '.csv'
    name_test = 'test' + str(n) + '.csv'
    
    train  = pd.read_csv(name_train)
    test = pd.read_csv(name_test)
    
    return train, test

train, test = load_data()


def transform_features_to_num(train, test):
    
    if(train.shape[1] != test.shape[1]):
        print("Fatal error")
        return 0
    
    for i in range(0, train.shape[1]):
        le = preprocessing.LabelEncoder()
        le.fit(train.iloc[:, i])
        
        train.iloc[:, i] = le.transform(train.iloc[:, i])
        test.iloc[:, i] = le.transform(test.iloc[:, i])
       
    return train, test
   


def benchmark_test(train, test, clf):
    
    train_X = train.drop('V10', axis = 1).as_matrix()
    train_Y = train['V10'].as_matrix()
    
    test_X = test.drop('V10', axis = 1).as_matrix()
    test_Y = test['V10'].as_matrix()
    
    
    clf.fit(train_X, train_Y)
    res = clf.predict(test_X)
    
    pct = 0.0001
    for i in range(0, len(res)):
        
        if(res[i] == test_Y[i]):
            pct += 1
    
    pct = pct / len(res)
    
    return pct
    

def single_back_test():
    
    clf = RandomForestClassifier(max_depth=8, random_state=0)
    res = benchmark_test(train, test, clf)
    print(res)


def back_test_all_data():
    print("Using Random forest")
    for i in range(1, 11):
        print(" accuracy on dataset: ", i)
        train, test = load_data(i)
        transform_features_to_num(train, test)
        
        clf = RandomForestClassifier(max_depth=8, random_state=0)
        res = benchmark_test(train, test, clf)
        print(res)
            




"""
######################################################################
"""

def intersect(pandas_df , obj):
    tmp = pd.DataFrame(index = pandas_df.index, columns = pandas_df.columns)
    tmp_ar = tmp.as_matrix()
    obj_ar = obj.as_matrix()
    pandas_df_ar = pandas_df.as_matrix()
    for i in range(0, pandas_df.shape[0]):
        for j  in range(0, pandas_df.shape[1]):
            if(pandas_df_ar[i][j] == obj_ar[j]):
                tmp_ar[i][j] = obj[j]
                
    tmp = pd.DataFrame(data = tmp_ar, index = tmp.index, columns = tmp.columns)
                
    
    
    return tmp
        

def inseted(pandas_df, obj, number_of_contradictions = 1):
    n_nan = 0
    for i in range(0, len(obj)):
        if(np.isnan(obj[i])):
            n_nan += 1
    
    num_of_inset = 0
    
    pandas_df_ar = pandas_df.as_matrix()
    obj_ar = obj.as_matrix()
    for i in range(0, pandas_df.shape[0]):
        count = 0
        for j  in range(0, pandas_df.shape[1]):
            
            if(pandas_df_ar[i][j] == obj_ar[j]):
                count += 1
            
            if(count == len(obj_ar) - n_nan):
                num_of_inset += 1
    
        if(num_of_inset == number_of_contradictions):
            break
    
    return num_of_inset




def full_back_test_intersect():
    
    train, test = load_data(7)
    transform_features_to_num(train, test)
    
    train_X_plus = train[train['V10']  == 1]
    train_X_plus = train_X_plus.drop('V10', axis = 1)
    
    train_X_minus = train[train['V10'] == 0]
    train_X_minus = train_X_minus.drop('V10', axis = 1)
    
    
    test_X = test.drop('V10', axis = 1)
    test_Y = test['V10']
    
    
   
    for_plus = []
    for_minus = []
    
    for i in range(0, test_X.shape[0]):
        print(i)
       
        
        tmp_1 = intersect(train_X_plus, test_X.iloc[i])
        tmp_2 = intersect(train_X_minus, test_X.iloc[i])
        
        current_for_plus = 0
        current_for_minus = 0
        
        
        for j in range(0, tmp_1.shape[0]):
            a = inseted(train_X_minus, tmp_1.iloc[j])
            if(a == 0):
                current_for_plus += 1
                
                
        for j in range(0, tmp_2.shape[0]):
            a = inseted(train_X_plus, tmp_2.iloc[j])
            if(a == 0):
                current_for_minus += 1
        
        for_plus.append(current_for_plus)
        for_minus.append(current_for_minus)
       
    answer = []
    for i in range(0, test_X.shape[0]):
        if(for_plus[i] > for_minus[i]):
            a = 1
        else:
            a = 0
        
        answer.append(a)
        
    pct = 0.00001
    for i in range(0, len(answer)):
        if(answer[i] == test_Y[i]):
            pct +=1
    
    pct = pct /len(answer)
    print(pct)
        
            
def partial_back_test_intersect():
    """
    Here we cut off train dataset in order to reduce time
    """
    print(" Here we cut off train dataset in order to reduce time")
    train, test = load_data(7)
    transform_features_to_num(train, test)
    
    train = train[:int(0.2*train.shape[0])]
    
    train_X_plus = train[train['V10']  == 1]
    train_X_plus = train_X_plus.drop('V10', axis = 1)
    
    train_X_minus = train[train['V10'] == 0]
    train_X_minus = train_X_minus.drop('V10', axis = 1)
    
    
    test_X = test.drop('V10', axis = 1)
    test_Y = test['V10']
    
    
   
    for_plus = []
    for_minus = []
    
    for i in range(0, test_X.shape[0]):
        print(i)
       
        
        tmp_1 = intersect(train_X_plus, test_X.iloc[i])
        tmp_2 = intersect(train_X_minus, test_X.iloc[i])
        
        current_for_plus = 0
        current_for_minus = 0
        
        
        for j in range(0, tmp_1.shape[0]):
            a = inseted(train_X_minus, tmp_1.iloc[j])
            if(a == 0):
                current_for_plus += 1
                
                
        for j in range(0, tmp_2.shape[0]):
            a = inseted(train_X_plus, tmp_2.iloc[j])
            if(a == 0):
                current_for_minus += 1
        
        for_plus.append(current_for_plus)
        for_minus.append(current_for_minus)
       
    answer = []
    for i in range(0, test_X.shape[0]):
        if(for_plus[i] > for_minus[i]):
            a = 1
        else:
            a = 0
        
        answer.append(a)
        
    pct = 0.00001
    for i in range(0, len(answer)):
        if(answer[i] == test_Y[i]):
            pct +=1
    
    pct = pct /len(answer)
    print(pct)
        
    
        
    
    
back_test_all_data()
partial_back_test_intersect()
"""
It will take a plenty of time
"""
full_back_test_intersect()













    
    
    
    
    
    

