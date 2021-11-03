# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:56:20 2021

@mentor: aftowicz
@author: honcharenko
"""
from pckg import load_SWaT_Dataset_Attack_v0, print_pipeline_params
import joblib
from WindowSlider import WindowSlider
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix   
import io, os, sys
from contextlib import redirect_stdout
from datetime import datetime
import shutil   
        
if (__name__ == '__main__'):
    #%% files
    now = datetime.now()
    experiment_dir = "20210716_110403"
    #data_path = "SWaT_Dataset_Attack_v0.csv"
    data_path = "SWaT_Dataset_Attack_v0.csv"
    joblib_path = "%s//pipeline.joblib" % experiment_dir
    stdout_path = "%s/%s_test_out.txt" % (experiment_dir, now.strftime("%Y%m%d_%H%M%S"))
    #%% BUFFER STDOUT
    f_buffer = io.StringIO()
    with redirect_stdout(f_buffer):
        #%% LOAD DATA
        print ("LOAD TEST DATA")
        print("------------")
        X_test, y_test = load_SWaT_Dataset_Attack_v0(path = data_path, nrows=None)
        X_test, y_test = load_SWaT_Dataset_Attack_v0(path = data_path, nrows=None)
        print("loaded",X_test.shape[0],"samples with",X_test.shape[1], "features and",y_test.nunique(),"unique labels from",os.path.relpath(data_path))
        print("") 
        #%% LOAD PIPELINE
        print("LOAD PIPELINE")
        print("------------")
        ml_pipeline = joblib.load(joblib_path)
        print("loaded pipeline from", os.path.relpath(joblib_path))
        print_pipeline_params(ml_pipeline)
        print("") 
        #%% TEST
        print("TEST")
        print("------------") 
        y_pred = ml_pipeline.predict(X_test)
        y_pred = (-y_pred + 1)/2
        y_test=ml_pipeline.named_steps['windowslider'].transform_labels(y_test)
        labels = ['Normal', 'Attack']
        print(classification_report(y_test,y_pred,target_names=labels))
        print("confusion matrix")
        print(confusion_matrix(y_test, y_pred))
        #disp = ConfusionMatrixDisplay(cm,display_labels=labels)
        #disp.plot(cmap = 'Blues') 
        #plt.savefig('cm.png')
    #%% SAVE STDOUT TO FILE
    os.makedirs(os.path.dirname(os.path.abspath(stdout_path)), exist_ok=True)
    with open(stdout_path, mode='w') as std_f:
        std_f.write(f_buffer.getvalue())
    #%% PRINT STDOUT
    print("Redirecting stdout to",os.path.relpath(stdout_path))
    print("") 
    print(f_buffer.getvalue())

    
    
    
