# -*- coding: utf-8 -*-
#
# Written by Zaixu Cui: zaixucui@gmail.com;
#                       Zaixu.Cui@pennmedicine.upenn.edu
#
# If you use this code, please cite: 
#                       Cui et al., 2018, Cerebral Cortex; 
#                       Cui and Gong et al., 2018, NeuroImage; 
#                       Cui et al., 2016, Human Brain Mapping.
# (google scholar: https://scholar.google.com.hk/citations?user=j7amdXoAAAAJ&hl=zh-TW&oi=ao)
#
import os
import scipy.io as sio
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
  
def LinearRegression_KFold_RandomCV(Subjects_Data, Subjects_Score, Fold_Quantity, ResultantFolder):
    #
    # linear regression with K-fold cross-validation
    # K-fold cross-validation is random, as the split of all subjects into K groups is random
    # Therefore, here, we repeated multiple times and averaged all accuracies
    #
    # Subjects_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Subjects_Score:
    #     n*1 vector, n is subjects quantity
    # Fold_Quantity:
    #     Fold quantity for the cross-validation
    #     5 or 10 is recommended generally, the small the better accepted by community, but the results may be worse as traning samples are fewer
    # ResultantFolder:
    #     Path of the folder storing the results
    #

    if not os.path.exists(ResultantFolder):
        os.mkdir(ResultantFolder)
    
    Subjects_Quantity = len(Subjects_Score)
    EachFold_Size = np.int(np.fix(np.divide(Subjects_Quantity, Fold_Quantity)))
    Remain = np.mod(Subjects_Quantity, Fold_Quantity)
    RandIndex = np.arange(Subjects_Quantity)
    np.random.shuffle(RandIndex)
    
    Fold_Corr = [];
    Fold_MAE = [];

    for j in np.arange(Fold_Quantity):

        Fold_J_Index = RandIndex[EachFold_Size * j + np.arange(EachFold_Size)]
        if Remain > j:
            Fold_J_Index = np.insert(Fold_J_Index, len(Fold_J_Index), RandIndex[EachFold_Size * Fold_Quantity + j])

        Subjects_Data_test = Subjects_Data[Fold_J_Index, :]
        Subjects_Score_test = Subjects_Score[Fold_J_Index]
        Subjects_Data_train = np.delete(Subjects_Data, Fold_J_Index, axis=0)
        Subjects_Score_train = np.delete(Subjects_Score, Fold_J_Index) 
 
        normalize = preprocessing.MinMaxScaler()
        Subjects_Data_train = normalize.fit_transform(Subjects_Data_train)
        Subjects_Data_test = normalize.transform(Subjects_Data_test)

        clf = linear_model.LinearRegression()
        clf.fit(Subjects_Data_train, Subjects_Score_train)
        Fold_J_Score = clf.predict(Subjects_Data_test)

        Fold_J_Corr = np.corrcoef(Fold_J_Score, Subjects_Score_test)
        Fold_J_Corr = Fold_J_Corr[0,1]
        Fold_Corr.append(Fold_J_Corr)
        Fold_J_MAE = np.mean(np.abs(np.subtract(Fold_J_Score,Subjects_Score_test)))
        Fold_MAE.append(Fold_J_MAE)
    
        Fold_J_result = {'Index':Fold_J_Index, 'Test_Score':Subjects_Score_test, 'Predict_Score':Fold_J_Score, 'Corr':Fold_J_Corr, 'MAE':Fold_J_MAE}
        Fold_J_FileName = 'Fold_' + str(j) + '_Score.mat'
        ResultantFile = os.path.join(ResultantFolder, Fold_J_FileName)
        sio.savemat(ResultantFile, Fold_J_result)

    Fold_Corr = [0 if np.isnan(x) else x for x in Fold_Corr]
    Mean_Corr = np.mean(Fold_Corr)
    Mean_MAE = np.mean(Fold_MAE)
    Res_NFold = {'Mean_Corr':Mean_Corr, 'Mean_MAE':Mean_MAE};
    ResultantFile = os.path.join(ResultantFolder, 'Res_NFold.mat')
    sio.savemat(ResultantFile, Res_NFold)
    return (Mean_Corr, Mean_MAE)  
        
def LinearRegression_Weight(Subjects_Data, Subjects_Score, ResultantFolder):
    #
    # Function to generate the contribution weight of all features
    # We generally use all samples to construct a new model to extract the weight of all features
    #
    # Subjects_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Subjects_Score:
    #     n*1 vector, n is subjects quantity
    # ResultantFolder:
    #     Path of the folder storing the results
    #
   
    if not os.path.exists(ResultantFolder):
        os.mkdir(ResultantFolder)

    Scale = preprocessing.MinMaxScaler()
    Subjects_Data = Scale.fit_transform(Subjects_Data)
    clf = linear_model.LinearRegression()
    clf.fit(Subjects_Data, Subjects_Score)
    Weight = clf.coef_ / np.sqrt(np.sum(clf.coef_ **2))
    Weight_result = {'w_Brain':Weight}
    sio.savemat(ResultantFolder + '/w_Brain.mat', Weight_result)
    return;
