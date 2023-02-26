# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:14:17 2022

@author: whill
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics
from keras import callbacks 


def lstm_mod(X,y,lenf, dates):
    train_size = int(len(X)*0.8)
    test_size= len(X)- train_size
    x_train, x_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]
    datestrain, datestest = dates[0:train_size], dates[train_size:len(dates)]

    scalerX = StandardScaler().fit(x_train)
    scalerY = StandardScaler().fit(y_train)
    
    x_train = scalerX.transform(x_train)
    y_train = scalerY.transform(y_train)
    x_test = scalerX.transform(x_test)
    y_test = scalerY.transform(y_test)
      
    x_train = x_train.reshape((x_train.shape[0],lenf,1))
    x_test = x_test.reshape((x_test.shape[0],lenf,1))
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  
                                        mode ="min", patience = 5,  
                                        restore_best_weights = True) 
    model = Sequential()
    model.add(LSTM(256, input_shape=(lenf, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    model.fit(x=x_train,y=y_train,epochs=25,validation_data=(x_test,y_test),shuffle=False,callbacks =[earlystopping])
    trainPredict = model.predict(x_train)
    testPredict = model.predict(x_test)

    # invert predictions
    trainPredict_y = scalerY.inverse_transform(trainPredict)
    testPredict_y = scalerY.inverse_transform(testPredict)
    y_true_train = y[0:train_size]
    y_true_test = y[-test_size:]

    df_ytrue_train = pd.DataFrame({'Dates' : datestrain.values,'Close': y_true_train.ravel(), 'Close Predicted' : trainPredict_y.flatten()})
    df_ytrue_test = pd.DataFrame({'Dates' : datestest.values, 'Close': y_true_test.ravel(), 'Close Predicted' : testPredict_y.flatten()})
    #print('Model accuracy (%)')
    Y_t=scalerY.inverse_transform(y_train)
    y_test_T = scalerY.inverse_transform(y_test)
    Accuracy = ((1-(metrics.mean_absolute_error(Y_t, trainPredict_y))))#/Y_t.mean()))*100)
    #print('')
    #print('Prediction performance')
    MAE =((metrics.mean_absolute_error(y_test_T, testPredict)))#/y_test_T.mean())*100)
    MSE=metrics.mean_squared_error(y_test_T,  testPredict)
    RMSE=np.sqrt(MSE)
    R2= metrics.r2_score(y_test_T,  testPredict)
    return df_ytrue_train, df_ytrue_test, Accuracy, MAE, MSE, RMSE, R2, datestrain, datestest
def preprocess(df):
    df.drop(columns=['Split Coef'], inplace=True)
    df.set_index('Dates', inplace=True)
    X = df.loc[:, collist]
    y = df[['Close']].values
    features = np.array(X.columns)
    return X, y, features    
def maketrainplots(df_ytrue_train, indic_type, key,df_type):
    fig, ax = plt.subplots(figsize= (9,5))
    plt.plot(df_ytrue_train['Dates'], df_ytrue_train['Close'],
                                label='Closing Price')
    plt.plot(df_ytrue_train['Dates'], df_ytrue_train['Close Predicted'],'r--',
                                label='Predicted Closing Price')
    plt.legend()
    plt.savefig(f'data_project/Price Plots/{indic_type}/train_{indic_type}_{df_type}_{key}.png')
    plt.close()
def maketestplots(df_ytrue_test, indic_type, key, df_type):
    fig, ax = plt.subplots(figsize= (9,5))
    plt.plot(df_ytrue_test['Dates'], df_ytrue_test['Close'],
                                label='Closing Price')
    plt.plot(df_ytrue_test['Dates'], df_ytrue_test['Close Predicted'],'r--',
                                label='Predicted Closing Price')
    plt.legend()
    plt.savefig(f'data_project/Price Plots/{indic_type}/test_{indic_type}_{df_type}_{key}.png')
    plt.close()
if __name__ == "__main__":   
    obj = pd.read_pickle(r'data_project/stocksandindicators.pickle')
    sp500_df = pd.read_csv('data_project/Stocks in the SP 500 Index.csv')
    indic_type = 'Mom'
    df_indicators = pd.read_csv('data_project/LassoCoefficientsAllModels_partial_mom.csv')
    df_indicators.rename(columns={'Unnamed: 0' : 'Stock'}, inplace=True)
    df_indicators.set_index('Stock', inplace=True)
    #sp500_df['Dividend yield'] = sp500_df['Dividend yield'].map(lambda x: x.rstrip('%'))
    #sp500_df['Dividend yield'] = sp500_df['Dividend yield'].apply(pd.to_numeric)
    #df_dividend = sp500_df.loc[sp500_df['Dividend yield']>0]
    #df_not_dividend = sp500_df.loc[sp500_df['Dividend yield']==0]
    
    df_sector_IT = sp500_df.loc[sp500_df['GICS Sector']=='Information Technology']
    df_sector_CS = sp500_df.loc[sp500_df['GICS Sector']=='Communication Services']
    df_sector_CD = sp500_df.loc[sp500_df['GICS Sector']=='Consumer Discretionary']
    df_sector_Fin = sp500_df.loc[sp500_df['GICS Sector']=='Financials']
    df_sector_HC = sp500_df.loc[sp500_df['GICS Sector']=='Health Care']
    df_sector_Energy = sp500_df.loc[sp500_df['GICS Sector']=='Energy']
    df_sector_Staples = sp500_df.loc[sp500_df['GICS Sector']=='Consumer Staples']
    df_sector_Utils = sp500_df.loc[sp500_df['GICS Sector']=='Utilities']
    df_sector_Materials = sp500_df.loc[sp500_df['GICS Sector']=='Materials']
    df_sector_Industrials = sp500_df.loc[sp500_df['GICS Sector']=='Industrials']
    df_sector_RE = sp500_df.loc[sp500_df['GICS Sector']=='Real Estate']
    
    #df_dividend = df_dividend['Symbol'].tolist()
    #df_not_dividend = df_not_dividend['Symbol'].tolist()
    
    df_sector_IT = df_sector_IT['Symbol'].tolist()
    df_sector_CS = df_sector_CS['Symbol'].tolist()
    df_sector_CD = df_sector_CD['Symbol'].tolist()
    df_sector_Fin = df_sector_Fin['Symbol'].tolist()
    df_sector_HC = df_sector_HC['Symbol'].tolist()
    df_sector_Energy = df_sector_Energy['Symbol'].tolist()
    df_sector_Staples = df_sector_Staples['Symbol'].tolist()
    df_sector_Utils = df_sector_Utils['Symbol'].tolist()
    df_sector_Materials = df_sector_Materials['Symbol'].tolist()
    df_sector_Industrials = df_sector_Industrials['Symbol'].tolist()
    df_sector_RE = df_sector_RE['Symbol'].tolist()
    
    #fulllist = [df_dividend,df_not_dividend,df_sector_IT,df_sector_CS,df_sector_CD,df_sector_Fin,df_sector_HC,
    #            df_sector_Energy,df_sector_Staples,df_sector_Utils,df_sector_Materials,df_sector_Industrials,df_sector_RE]
    fulllist = [df_sector_Staples,df_sector_Utils,df_sector_Materials,df_sector_Industrials,df_sector_RE]#
    #fulllist = [df_sector_Energy,df_sector_IT,df_sector_CS,df_sector_CD,df_sector_Fin,df_sector_HC]
               # df_sector_Staples,df_sector_Utils,df_sector_Materials,df_sector_Industrials,df_sector_RE]
    #splitnames = ['dividend', 'no dividend','Information Technology','Communication Services','Consumer Discretionary','Financials','Health Care',
    #              'Energy','Consumer Staples','Utilities','Materials','Industrials','Real Estate']
    splitnames = ['Consumer Staples','Utilities','Materials','Industrials','Real Estate']#
    #splitnames = ['Energy','Information Technology','Communication Services','Consumer Discretionary','Financials','Health Care']
                 #'Consumer Staples','Utilities','Materials','Industrials','Real Estate']
    ct = 0
    
    for dfl in fulllist:
        df_temp = df_indicators[df_indicators.index.isin(dfl)]
        lendft = len(df_temp)
        params_list=[]
        for col in df_temp:
            count0 = df_temp[col].isin([0]).sum()
            if (lendft - count0)/lendft < 0.7:
                df_temp.drop(col, axis=1, inplace=True)
            else:
                params_list.append(f'{col}: {count0}')
        
        collist = df_temp.columns
        df_errors = pd.DataFrame()
        df_type = splitnames[ct]
        with open(f'data_project/Price Plots/{indic_type}/parameters_{indic_type}_{df_type}.txt', 'w') as fp:
            for item in params_list:
                # write each item on a new line
                fp.write("%s\n" % item)

        for key in obj:
            
            if key in df_temp.index:
                #idx_list.append(key)
                X, y, features = preprocess(obj[key])
                df_ytrue_train, df_ytrue_test, Accuracy, MAE, MSE, RMSE, R2, datestrain, datestest = lstm_mod(X,y,len(features), obj[key].index)
                maketrainplots(df_ytrue_train, indic_type, key, df_type)
                maketestplots(df_ytrue_test, indic_type, key, df_type)
                #print(key, X, features)
                df_temp_errors = pd.DataFrame(columns=['Accuracy','MAE','MSE','RMSE','R2', 'Start Date Train', 'Start Date Test', 'End Date Test'])
                df_temp_errors.loc[0] = [Accuracy, MAE, MSE,RMSE,R2, datestrain[0], datestest[0], datestest[-1]]
                df_errors =pd.concat([df_errors,df_temp_errors])
                print(df_errors)
        df_errors.index=dfl
        df_errors.to_csv(f'data_project/Price Plots/{indic_type}/errors_{indic_type}_{df_type}.csv')
        print(splitnames[ct])
        ct+=1