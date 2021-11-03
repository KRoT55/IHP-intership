# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:06:00 2021

@author: mentor
@author: honcharenko
"""
from pckg import load_SWaT_Dataset_Normal_v0, print_pipeline_params
from WindowSlider import WindowSlider
from WindowSlider_y import WindowSlider_y
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


from pckg import load_SWaT_Dataset_Attack_v0, print_pipeline_params
import joblib
from WindowSlider import WindowSlider
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix   
import io, os, sys
from contextlib import redirect_stdout
from datetime import datetime
import shutil   
#%%
 
    
if (__name__ == '__main__'):
    #%% files
    now = datetime.now()
    experiment_dir = now.strftime("%Y%m%d_%H%M%S")
    data_path = "SWaT_Dataset_Normal_v0.csv"
    joblib_path = "%s/pipeline.joblib" % experiment_dir
    stdout_path = "%s/train_out.txt" % experiment_dir
    #%% variables
    nrows = 1000 # None for all samples
    pca_n_components = 6
    ocsvm_nu = 1e-3
    ocsvm_gamma = 1e-3
    ocsv_kernel ='rbf'
    #%% BUFFER STDOUT
    f_buffer = io.StringIO()
    with redirect_stdout(f_buffer):
        #%% LOAD
        print ("LOAD TRAINING DATA")
        print("------------")
        #X_train, y_train = load_SWaT_Dataset_Normal_v0(path = data_path, nrows=nrows)   
        raw_data = pd.read_csv('SWaT_Dataset_Normal_v0.csv') 
        #raw_data = pd.read_csv('WADI_14days.csv')  
        raw_data.rename(columns={'Normal/Attack' : 'label'}, inplace = True)
        raw_data.loc[raw_data['label'] != 'Normal', 'label'] = 1
        raw_data.loc[raw_data['label'] == 'Normal', 'label'] = 0
        X_train = raw_data.drop(columns=['label', ' Timestamp'])
        y_train = raw_data['label']
        print("loaded",X_train.shape[0],"samples with",X_train.shape[1], "features and",y_train.nunique(),"unique labels from",os.path.relpath(data_path))
        print("") 
        #%% PREPARE PIPELINE
        print("PIPELINE")
        print("------------")
        ml_pipeline = make_pipeline(StandardScaler(),
                                    PCA(pca_n_components),
                                    WindowSlider(4),
                                    OneClassSVM(nu=ocsvm_nu, 
                                                kernel=ocsv_kernel,
                                                gamma=ocsvm_gamma)
                                    )
        print_pipeline_params(ml_pipeline)
        print("") 
        #%% TRAIN
        print("TRAIN")
        print("------------") 
        ml_pipeline.fit(X_train)
        ocsvm = ml_pipeline.steps[-1][1]
        print("fitted",ocsvm.n_support_,"support_vectors with", ocsvm._intercept_,"intercept")
        print("") 
            
        #%% LOAD TEST DATA
        print ("LOAD TEST DATA")
        print("------------")
        X_test, y_test = load_SWaT_Dataset_Attack_v0(path = data_path, nrows=None)
        X_test, y_test = load_SWaT_Dataset_Attack_v0(path = data_path, nrows=None)
        print("loaded",X_test.shape[0],"samples with",X_test.shape[1], "features and",y_test.nunique(),"unique labels from",os.path.relpath(data_path))
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
        
        
        
                
        # Set the parameters by cross-validation
        tuned_parameters = [{'estimator__pca__n_components': [2,3,4,5,6,7,8],
                              'estimator__windowslider__size': [2,3,4,5,6,7,8],
                              'estimator__oneclasssvm__kernel': ['rbf','linear'],
                              'estimator__oneclasssvm__gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                              'estimator__oneclasssvm__nu': [1e-2, 1e-3, 1e-4, 1e-6]}]
        scores = [ 'recall']
        
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()
        
            clf = GridSearchCV(
                ml_pipeline, tuned_parameters, scoring='%s_macro' % score
            )
            clf.fit(X_train, y_train)
        
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()
        
            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()
                
                
                
                
        
        #%% SAVE
        print ("SAVE MODEL")
        print("------------") 
        os.makedirs(os.path.dirname(os.path.abspath(joblib_path)), exist_ok=True)
        joblib.dump(ml_pipeline,joblib_path)
        print("saved pipeline to", os.path.relpath(joblib_path)) 
        print("") 
    #%% SAVE STDOUT TO FILE
    os.makedirs(os.path.dirname(os.path.abspath(stdout_path)), exist_ok=True)
    with open(stdout_path, mode='w') as std_f:
        std_f.write(f_buffer.getvalue())
    #%% PRINT STDOUT
    print("Redirecting stdout to",os.path.relpath(stdout_path))
    print("") 
    print(f_buffer.getvalue())
    
