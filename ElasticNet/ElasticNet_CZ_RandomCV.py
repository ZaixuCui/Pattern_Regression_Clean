# -*- coding: utf-8 -*-
#
# Written by Zaixu Cui: zaixucui@gmail.com;
#                       Zaixu.Cui@pennmedicine.upenn.edu
#
# If you use this code, please cite: 
#                       Cui et al., 2018, Cerebral Cortex; 
#                       Cui and Gong, 2018, NeuroImage; 
#                       Cui et al., 2016, Human Brain Mapping.
# (google scholar: https://scholar.google.com.hk/citations?user=j7amdXoAAAAJ&hl=zh-TW&oi=ao)
#
import os
import scipy.io as sio
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from joblib import Parallel, delayed

def ElasticNet_KFold_RandomCV(Subjects_Data, Subjects_Score, Fold_Quantity, Alpha_Range, L1_ratio_Range, CVRepeatTimes_ForInner, ResultantFolder, Parallel_Quantity):
    #
    # Elastic-Net regression with random K-fold cross-validation 
    # Because the k-fold separation is random, this prediction generally needed to be repeated several times.
    # It is easy to write a for loop to repeat this function, each time should generate a slightly different result. 
    #
    # Subjects_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Subjects_Score:
    #     n*1 vector, n is subjects quantity
    # Fold_Quantity:
    #     Fold quantity for the cross-validation
    #     5 or 10 is recommended generally, the small the better accepted by community, but the results may be worse as traning samples are fewer
    # Alpha_Range:
    #     Range of alpha, the regularization parameter balancing the training error and regularization penalty
    #     Our previous paper used (2^(-10), 2^(-9), ..., 2^4, 2^5), see Cui and Gong (2018), NeuroImage
    # L1_ratio_Range:
    #     Range of l1 ratio, the parameter balancing l1 and l2 penalty
    #     Our previous paper 10 values in the range [0.2,1], see Cui et al., (2018), Cerebral Cortex
    # CVRepeatTimes_ForInner:
    #     The repeatition time for the inner random CV, which was used to select the optimal alpha parameter
    #     i.e., 20
    # ResultantFolder:
    #     Path of the folder storing the results
    # Parallel_Quantity:
    #     Parallel multi-cores on one single computer, at least 1
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

        Optimal_Alpha, Optimal_L1_ratio = ElasticNet_OptimalAlpha_KFold(Subjects_Data_train, Subjects_Score_train, Fold_Quantity, Alpha_Range, L1_ratio_Range, CVRepeatTimes_ForInner, ResultantFolder, Parallel_Quantity)

        normalize = preprocessing.MinMaxScaler()
        Subjects_Data_train = normalize.fit_transform(Subjects_Data_train)
        Subjects_Data_test = normalize.transform(Subjects_Data_test)

        clf = linear_model.ElasticNet(alpha=Optimal_Alpha, l1_ratio=Optimal_L1_ratio)
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

def ElasticNet_OptimalAlpha_KFold(Training_Data, Training_Score, Fold_Quantity, Alpha_Range, L1_ratio_Range, CVRepeatTimes, ResultantFolder, Parallel_Quantity):
    #
    # Select optimal regularization parameter using nested random k-fold cross-validation
    #
    # Training_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Training_Score:
    #     n*1 vector, n is subjects quantity
    # Fold_Quantity:
    #     Fold quantity for the cross-validation
    #     5 or 10 is recommended generally, the small the better accepted by community, but the results may be worse as traning samples are fewer
    # Alpha_Range:
    #     Range of alpha, the regularization parameter balancing the training error and regularization penalty
    #     Our previous paper used (2^(-10), 2^(-9), ..., 2^4, 2^5), see Cui and Gong (2018), NeuroImage
    # L1_ratio_Range:
    #     Range of l1 ratio, the parameter balancing l1 and l2 penalty
    #     Our previous paper 10 values in the range [0.2,1], see Cui et al., (2018), Cerebral Cortex
    # CVRepeatTimes:
    #     The repeatition time, i.e., 20
    #     As k-fold split is random, we need to repeat
    # ResultantFolder:
    #     Path of the folder storing the results
    # Parallel_Quantity:
    #     Parallel multi-cores on one single computer, at least 1
    #
    
    if not os.path.exists(ResultantFolder):
        os.mkdir(ResultantFolder);

    Subjects_Quantity = len(Training_Score)
    Inner_EachFold_Size = np.int(np.fix(np.divide(Subjects_Quantity, Fold_Quantity)))
    Remain = np.mod(Subjects_Quantity, Fold_Quantity)
    
    Parameter_Combination_Quantity = len(Alpha_Range) * len(L1_ratio_Range)
    Inner_Corr_Mean = np.zeros((CVRepeatTimes, Parameter_Combination_Quantity))
    Inner_MAE_inv_Mean = np.zeros((CVRepeatTimes, Parameter_Combination_Quantity))
    for i in np.arange(CVRepeatTimes):

        RandIndex = np.arange(Subjects_Quantity)
        np.random.shuffle(RandIndex)

        Inner_Corr = np.zeros((Fold_Quantity, Parameter_Combination_Quantity))
        Inner_MAE_inv = np.zeros((Fold_Quantity, Parameter_Combination_Quantity))

        for k in np.arange(Fold_Quantity):
          
            print(k)
            Inner_Fold_K_Index = RandIndex[Inner_EachFold_Size * k + np.arange(Inner_EachFold_Size)]
            if Remain > k:
                Inner_Fold_K_Index = np.insert(Inner_Fold_K_Index, len(Inner_Fold_K_Index), RandIndex[Inner_EachFold_Size * Fold_Quantity + k])

            Inner_Fold_K_Data_test = Training_Data[Inner_Fold_K_Index, :]
            Inner_Fold_K_Score_test = Training_Score[Inner_Fold_K_Index]
            Inner_Fold_K_Data_train = np.delete(Training_Data, Inner_Fold_K_Index, axis=0)
            Inner_Fold_K_Score_train = np.delete(Training_Score, Inner_Fold_K_Index)
        
            Scale = preprocessing.MinMaxScaler()
            Inner_Fold_K_Data_train = Scale.fit_transform(Inner_Fold_K_Data_train)
            Inner_Fold_K_Data_test = Scale.transform(Inner_Fold_K_Data_test)    
        
            Parallel(n_jobs=Parallel_Quantity,backend="threading")(delayed(ElasticNet_SubAlpha)(Inner_Fold_K_Data_train, Inner_Fold_K_Score_train, Inner_Fold_K_Data_test, Inner_Fold_K_Score_test, Alpha_Range, L1_ratio_Range, l, ResultantFolder) for l in np.arange(Parameter_Combination_Quantity))        
            for l in np.arange(Parameter_Combination_Quantity):
                Fold_l_Mat_Path = ResultantFolder + '/Alpha_' + str(l) + '.mat';
                Fold_l_Mat = sio.loadmat(Fold_l_Mat_Path)
                Inner_Corr[k, l] = Fold_l_Mat['Corr'][0][0]
                Inner_MAE_inv[k, l] = Fold_l_Mat['MAE_inv']
                os.remove(Fold_l_Mat_Path)
            
            Inner_Corr = np.nan_to_num(Inner_Corr)
        Inner_Corr_Mean[i, :] = np.mean(Inner_Corr, axis=0)
        Inner_MAE_inv_Mean[i, :] = np.mean(Inner_MAE_inv, axis=0)

    Inner_Corr_CVMean = np.mean(Inner_Corr_Mean, axis=0)
    Inner_MAE_inv_CVMean = np.mean(Inner_MAE_inv_Mean, axis=0)
    Inner_Corr_CVMean = (Inner_Corr_CVMean - np.mean(Inner_Corr_CVMean)) / np.std(Inner_Corr_CVMean)
    Inner_MAE_inv_CVMean = (Inner_MAE_inv_CVMean - np.mean(Inner_MAE_inv_CVMean)) / np.std(Inner_MAE_inv_CVMean)
    Inner_Evaluation = Inner_Corr_CVMean + Inner_MAE_inv_CVMean
    
    Inner_Evaluation_Mat = {'Inner_Corr':Inner_Corr, 'Inner_MAE_inv':Inner_MAE_inv, 'Inner_Corr_CVMean':Inner_Corr_CVMean, 'Inner_MAE_inv_CVMean':Inner_MAE_inv_CVMean, 'Inner_Evaluation':Inner_Evaluation}
    sio.savemat(ResultantFolder + '/Inner_Evaluation.mat', Inner_Evaluation_Mat)
    
    Optimal_Combination_Index = np.argmax(Inner_Evaluation) 
    
    Optimal_Alpha_Index = np.int64(np.ceil((Optimal_Combination_Index + 1) / len(L1_ratio_Range))) - 1
    Optimal_Alpha = Alpha_Range[Optimal_Alpha_Index]
    Optimal_L1_ratio_Index = np.mod(Optimal_Combination_Index, len(L1_ratio_Range))
    Optimal_L1_ratio = L1_ratio_Range[Optimal_L1_ratio_Index]
    return (Optimal_Alpha, Optimal_L1_ratio)

def ElasticNet_SubAlpha(Training_Data, Training_Score, Testing_Data, Testing_Score, Alpha_Range, L1_ratio_Range, Parameter_Combination_Index, ResultantFolder):
    #
    # Sub-function for optimal regularization parameter selection
    # The range of Parameter_Combination_Index is: 0----(len(Alpha_Range)*len(L1_ratio_Range)-1))
    # Calculating the alpha index and l1_ratio index from the parameter_combination_index
    #
    # Training_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Training_Score:
    #     n*1 vector, n is subjects quantity
    # Testing_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Testing_Score:
    #     n*1 vector, n is subjects quantity
    # Alpha_Range:
    #     See help in function 'ElasticNet_OptimalAlpha_KFold'
    # L1_ratio_Range:
    #     See help in function 'ElasticNet_OptimalAlpha_KFold'
    # Parameter_Combination_Index:
    #     The indice of the (alpha, L1_ratio) combination we tested in the whole range
    # ResultantFolder:
    #     Folder to storing the results
    #
   
    Alpha_Index = np.int64(np.ceil((Parameter_Combination_Index + 1) / len(L1_ratio_Range))) - 1
    L1_ratio_Index = np.mod(Parameter_Combination_Index, len(L1_ratio_Range))
    clf = linear_model.ElasticNet(l1_ratio=L1_ratio_Range[L1_ratio_Index], alpha=Alpha_Range[Alpha_Index])
    clf.fit(Training_Data, Training_Score)
    Predict_Score = clf.predict(Testing_Data)
    Fold_Corr = np.corrcoef(Predict_Score, Testing_Score)
    Fold_Corr = Fold_Corr[0,1]
    Fold_MAE_inv = np.divide(1, np.mean(np.abs(Predict_Score - Testing_Score)))
    Fold_result = {'Corr': Fold_Corr, 'MAE_inv':Fold_MAE_inv}
    ResultantFile = ResultantFolder + '/Alpha_' + str(Parameter_Combination_Index) + '.mat'
    sio.savemat(ResultantFile, Fold_result)
    
def ElasticNet_Weight(Subjects_Data, Subjects_Score, Alpha_Range, L1_ratio_Range, Nested_Fold_Quantity, ResultantFolder, Parallel_Quantity):
    #
    # Function to generate the contribution weight of all features
    # We generally use all samples to construct a new model to extract the weight of all features
    #
    # Subjects_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Subjects_Score:
    #     n*1 vector, n is subjects quantity
    # Alpha_Range:
    #     Range of alpha, the regularization parameter balancing the training error and L2 penalty
    #     Our previous paper used (2^(-10), 2^(-9), ..., 2^4, 2^5), see Cui and Gong (2018), NeuroImage
    # L1_ratio_Range:
    #     Range of l1 ratio, the parameter balancing l1 and l2 penalty
    #     Our previous paper 10 values in the range [0.2,1], see Cui et al., (2018), Cerebral Cortex
    # Nested_Fold_Quantity:
    #     Fold quantity for the nested cross-validation, which was used to select the optimal parameter
    #     5 or 10 is recommended generally, the small the better accepted by community, but the results may be worse as traning samples are fewer
    # ResultantFolder:
    #     Path of the folder storing the results
    # Parallel_Quantity:
    #     Parallel multi-cores on one single computer, at least 1
    #

    if not os.path.exists(ResultantFolder):
        os.mkdir(ResultantFolder)

    # Select optimal alpha using inner fold cross validation
    Optimal_Alpha, Optimal_L1_ratio = ElasticNet_OptimalAlpha_KFold(Subjects_Data, Subjects_Score, Nested_Fold_Quantity, Alpha_Range, L1_ratio_Range, ResultantFolder, Parallel_Quantity)
    
    Scale = preprocessing.MinMaxScaler()
    Subjects_Data = Scale.fit_transform(Subjects_Data)
    clf = linear_model.ElasticNet(alpha=Optimal_Alpha, l1_ratio=Optimal_L1_ratio)
    clf.fit(Subjects_Data, Subjects_Score)
    Weight = clf.coef_ / np.sqrt(np.sum(clf.coef_ **2))
    Weight_result = {'w_Brain':Weight, 'alpha':Optimal_Alpha}
    sio.savemat(ResultantFolder + '/w_Brain.mat', Weight_result)
    return;
