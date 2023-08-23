#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import pickle


# In[13]:


scaler_filename = 'scaler.pkl'


# In[3]:


def dummy_encode(cat_feat):
    keys = {'SeniorCitizen_1', 'Partner_Yes', 
           'Dependents_Yes', 'OnlineSecurity_Yes', 'TechSupport_Yes', 
           'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 
           'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check'}

    res = {i: None for i in keys}
    
    res['SeniorCitizen_1'] = 1 if cat_feat['SeniorCitizen'] == 'Yes' else 0 
    res['Partner_Yes'] = 1 if cat_feat['Partner'] == 'Yes' else 0
    res['Dependents_Yes'] = 1 if cat_feat['Dependents'] == 'Yes' else 0
    res['OnlineSecurity_Yes'] = 1 if cat_feat['OnlineSecurity'] == 'Yes' else 0
    res['TechSupport_Yes'] = 1 if cat_feat['TechSupport'] == 'Yes' else 0
    res['PaperlessBilling_Yes'] = 1 if cat_feat['PaperlessBilling'] == 'Yes' else 0
    
    if cat_feat['Contract'] == 'One year':
        res['Contract_One year'], res['Contract_Two year'] = 1, 0
    elif cat_feat['Contract'] == 'Two year':
        res['Contract_One year'], res['Contract_Two year'] = 0, 1
    else:
        res['Contract_One year'], res['Contract_Two year'] = 0, 0
        
    if cat_feat['PaymentMethod'] == 'Credit card':
        res['PaymentMethod_Credit card (automatic)'], res['PaymentMethod_Electronic check'] = 1, 0
    elif cat_feat['PaymentMethod'] == 'Electronic check':
        res['PaymentMethod_Credit card (automatic)'], res['PaymentMethod_Electronic check'] = 0, 1
    else:
        res['PaymentMethod_Credit card (automatic)'], res['PaymentMethod_Electronic check'] = 0, 0 
        
    return res


# In[27]:


def scale_features(num_feat):
    keys = {'tenure', 'MonthlyCharges'}
    res = {i: None for i in keys}
    df = pd.DataFrame(num_feat, index=[0])
    
    load_scaler = pickle.load(open(scaler_filename, 'rb'))
    [[res['tenure'], res['MonthlyCharges']]] = load_scaler.transform(df).tolist()
    
    return res


# In[28]:


def transform_features(features):
    cf = ['SeniorCitizen', 'Partner', 'Dependents', 'OnlineSecurity',
         'TechSupport', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    nf = ['tenure', 'MonthlyCharges']
    ordered_keys = ['tenure', 'MonthlyCharges', 'SeniorCitizen_1', 'Partner_Yes',
           'Dependents_Yes', 'OnlineSecurity_Yes', 'TechSupport_Yes',
           'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
           'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check']
    
    cat_features = {i: features[i] for i in cf}
    num_features = {i: features[i] for i in nf}
    
    res = dummy_encode(cat_features)
    temp = scale_features(num_features)
    res.update(temp)
    
    res = {key: res[key] for key in ordered_keys}
    
    return res

