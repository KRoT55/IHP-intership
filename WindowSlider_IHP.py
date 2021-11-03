# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 14:52:20 2021

@author: mentor
@author: honcharenko
"""


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.pipeline import make_pipeline
import io, os, sys
import joblib
from contextlib import redirect_stdout
from datetime import datetime
import shutil


import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import classification_report

from pckg import load_SWaT_Dataset_Normal_v0, print_pipeline_params
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.pipeline import make_pipeline
import io, os, sys
import joblib
from contextlib import redirect_stdout
from datetime import datetime
import shutil
from sklearn.model_selection import GridSearchCV




class WindowSlider( BaseEstimator, TransformerMixin ):
    #Class Constructor
    def __init__( self, size = 2, stride = 1 ):
        self.size = size
        self.stride = stride
    #Return self nothing else to do here
    def fit( self, X, y = None ):
        return self
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        X_windowed = []
        n_windows = X.shape[0]-self.size+1
        X_pd = pd.DataFrame(X)
        for i in range(0,n_windows,self.stride):
            row = pd.Series(X_pd[i:i+self.size].values.flatten())
            X_windowed.append(row)
        X_windowed = pd.concat(X_windowed,axis=1).T
        return X_windowed
      
    
    def transform_labels( self, y   ):
        j=0
        k=0
        l=0
        y_windowed = []
        n_windows = y.shape[0]-self.size+1
        y_pd = pd.DataFrame(y)
        for i in range(0,n_windows,self.stride):
            row = pd.Series(y_pd[i:i+self.size].values.flatten())
            if (pd.Series(row[:]).any())==1:
                    j=j+1
                    #print(" 1")
                    print("Sum of 1",j)
                    y_windowed.append(pd.Series([1]))
            else:
                    k=k+1
                    #print( '0')
                    print("Sum of 0",k)
                    y_windowed.append(pd.Series([0]))
                #print(row)
            l=i
            print("Sum of Itter", l)
        y_windowed = pd.concat(y_windowed,axis=1).T
        return y_windowed 
      
      
            
if (__name__ == '__main__'):

    d = { 'feature3': [0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1]}
    
    X = pd.DataFrame(data=d)
    y = pd.DataFrame(data=d['feature3'])
    y_test=0
    slider = WindowSlider(size = 4, stride =1)
    #slider_y = WindowSlider(size = 4, stride =1)
    X_windowed = slider.fit_transform(X)
    y_windowed = slider.transform_labels(y)
    #y_test=y.any(1)
   # if (y==1).any():
    #    y_test[:]=1
    #if ( y==1).any() :
    #    
    #else :
    #    y_test[:]=0
    
"""
    print ("LOAD TRAINING DATA")
    print("------------")
    #X_train, y_train = load_SWaT_Dataset_Normal_v0(path = data_path, nrows=nrows)   
    raw_data = pd.read_csv('C:/Users/honcharenko/Desktop/SWat_20k_rows_50-50.csv') 
    raw_data.rename(columns={'Normal1' : 'label'}, inplace = True)
    raw_data.loc[raw_data['label'] != 'Normal', 'label'] = 1
    raw_data.loc[raw_data['label'] == 'Normal', 'label'] = 0
    X_train = raw_data.drop(columns=['label', 'Time'])
    y_train = raw_data.drop(columns=['Time'])
    y_train = raw_data['label']
    print("loaded","samples with", "features and",y_train.nunique(),"unique labels from")
    print("") 
    X_windowed = slider.fit_transform(X_train)
    y_windowed = slider.transform_labels(y_train)
    
  
        
    """
