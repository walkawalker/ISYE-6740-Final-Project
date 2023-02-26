# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 18:59:09 2022

@author: whill
"""

import pandas as pd
import numpy as np
import matplotlib
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess(df):
    df.drop(columns=['Split Coef'], inplace=True)
    df.set_index('Dates', inplace=True)
    X = df.loc[:, df.columns != 'Close']
    y = df['Close']
    features = np.array(X.columns)
    return X, y, features

def preprocess_partial(df, split='mom'):
    df.drop(columns=['Split Coef'], inplace=True)
    df.set_index('Dates', inplace=True)
    B = df.loc[:, ['Open', 'High', 'Low']]
    y = df['Close']
    if split == 'mom':
        mom_cols = [col for col in df.columns if 'momentum' in col]
        A = df.loc[:, mom_cols]
        X= A.join(B,how='left')
        features = np.array(X.columns)
    elif split == 'trend':
        mom_cols = [col for col in df.columns if 'trend' in col]
        A = df.loc[:, mom_cols]
        X= A.join(B,how='left')
        features = np.array(X.columns)
    elif split == 'vol':
        mom_cols = [col for col in df.columns if 'volatility' in col]
        A = df.loc[:, mom_cols]
        X= A.join(B,how='left')
        features = np.array(X.columns)
    return X, y, features


def callLasso(X,y,features):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=642)
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', Lasso())])
    search = GridSearchCV(pipeline, {'model__alpha': np.arange(0.1,10,0.1)},cv=5, scoring='neg_mean_squared_error', verbose=3)    
    search.fit(X_train,y_train)
    alph = search.best_params_
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)
    imp_feats = features[importance>0]
    return alph, coefficients, imp_feats
def corrplots(newcorr):
    fig, ax = plt.subplots(figsize= (9,5))
    sns.heatmap(newcorr, annot=True, fmt='.3f', 
                cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax, linewidths=0.2)
    ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
    plt.savefig(f'data_project/corr_plts/{key}_corrplot.png', pad_inches=0.1)

if __name__ == "__main__":
    flag = input('Please enter full or parse: ')
    split = input('Please enter one of [mom, trend, vol]:  ')
    if flag == 'full':
        obj = pd.read_pickle(r'data_project\stocksandindicators.pickle')
        ct = 0 
        idx_list = []
        df_all = pd.DataFrame()
        for key in obj:
            idx_list.append(key)
            X, y, features = preprocess(obj[key])
            alph, coefficients, imp_feats = callLasso(X, y, features)
            df_temp = pd.DataFrame(columns=X.columns)
            df_temp.loc[0] = coefficients
            df_all =pd.concat([df_all,df_temp],)
            print(alph, imp_feats)
            corrplots(X[imp_feats].corr())
        df_all.set_index(idx_list, inplace=True)
        df_all.to_csv('data_project/LassoCoefficientsAllModels.csv')
    elif flag == 'parse':
        obj = pd.read_pickle(r'data_project\stocksandindicators.pickle')
        ct = 0 
        idx_list = []
        df_all = pd.DataFrame()
        for key in obj:
            idx_list.append(key)
            X, y, features = preprocess_partial(obj[key], split=split)
            alph, coefficients, imp_feats = callLasso(X, y, features)
            df_temp = pd.DataFrame(columns=X.columns)
            df_temp.loc[0] = coefficients
            df_all =pd.concat([df_all,df_temp],)
            print(alph, imp_feats)
            #corrplots(X[imp_feats].corr())
        df_all.index=idx_list
        df_all.to_csv(f'data_project/LassoCoefficientsAllModels_partial_{split}.csv')
    else: 
        print("restart")
    
        
