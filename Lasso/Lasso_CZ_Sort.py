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
import time
from sklearn import linear_model
from sklearn import preprocessing
from joblib import Parallel, delayed
  
def Lasso_KFold_Sort_Permutation(Subjects_Data, Subjects_Score, Times_IDRange, Fold_Quantity, Alpha_Range, ResultantFolder, Parallel_Quantity, Max_Queued, QueueOptions):
     
    #
    # Lasso regression with K-fold cross-validation
    #
    # Subjects_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Subjects_Score:
    #     n*1 vector, n is subjects quantity
    # Times_IDRange:
    #     The index of permutation test, for example np.arange(1000)
    # Fold_Quantity:
    #     Fold quantity for the cross-validation
    #     5 or 10 is recommended generally, the small the better accepted by community, but the results may be worse as traning samples are fewer
    # Alpha_Range:
    #     Range of alpha, the regularization parameter balancing the training error and L1 penalty
    #     Our previous paper used (2^(-10), 2^(-9), ..., 2^4, 2^5), see Cui and Gong (2018), NeuroImage
    # ResultantFolder:
    #     Path of the folder storing the results
    # Parallel_Quantity:
    #     Parallel multi-cores on one single computer, at least 1
    # Max_Queued:
    #     The maximum jobs to be submitted to SGE cluster at the same time 
    # QueueOptions:
    #     Generally is '-q all.q' for SGE cluster 
    #
 
    if not os.path.exists(ResultantFolder):
        os.mkdir(ResultantFolder)
    Subjects_Data_Mat = {'Subjects_Data': Subjects_Data}
    Subjects_Data_Mat_Path = ResultantFolder + '/Subjects_Data.mat'
    sio.savemat(Subjects_Data_Mat_Path, Subjects_Data_Mat)
    Finish_File = []
    Times_IDRange_Todo = np.int64(np.array([]))
    for i in np.arange(len(Times_IDRange)):
        ResultantFolder_I = ResultantFolder + '/Time_' + str(Times_IDRange[i])
        if not os.path.exists(ResultantFolder_I):
            os.mkdir(ResultantFolder_I)
        if not os.path.exists(ResultantFolder_I + '/Res_NFold.mat'):
            Times_IDRange_Todo = np.insert(Times_IDRange_Todo, len(Times_IDRange_Todo), Times_IDRange[i])
            Configuration_Mat = {'Subjects_Data_Mat_Path': Subjects_Data_Mat_Path, 'Subjects_Score': Subjects_Score, 'Fold_Quantity': Fold_Quantity, \
                'Alpha_Range': Alpha_Range, 'ResultantFolder_I': ResultantFolder_I, 'Parallel_Quantity': Parallel_Quantity};
            sio.savemat(ResultantFolder_I + '/Configuration.mat', Configuration_Mat)
            system_cmd = 'python3 -c ' + '\'import sys;\
                sys.path.append("' + os.getcwd() + '");\
                from Lasso_CZ_Sort import Lasso_KFold_Sort_Permutation_Sub;\
                import os;\
                import scipy.io as sio;\
                configuration = sio.loadmat("' + ResultantFolder_I + '/Configuration.mat");\
                Subjects_Data_Mat_Path = configuration["Subjects_Data_Mat_Path"];\
                Subjects_Score = configuration["Subjects_Score"];\
                Fold_Quantity = configuration["Fold_Quantity"];\
                Alpha_Range = configuration["Alpha_Range"];\
                ResultantFolder_I = configuration["ResultantFolder_I"];\
                Parallel_Quantity = configuration["Parallel_Quantity"];\
                Lasso_KFold_Sort_Permutation_Sub(Subjects_Data_Mat_Path[0], Subjects_Score[0], Fold_Quantity[0][0], Alpha_Range[0], ResultantFolder_I[0], Parallel_Quantity[0][0])\' ';
            system_cmd = system_cmd + ' > "' + ResultantFolder_I + '/Lasso.log" 2>&1\n'
            Finish_File.append(ResultantFolder_I + '/Res_NFold.mat')
            script = open(ResultantFolder_I + '/script.sh', 'w') 
            script.write(system_cmd)
            script.close()

    Jobs_Quantity = len(Finish_File)
    if len(Times_IDRange_Todo) > Max_Queued:
        Submit_Quantity = Max_Queued
    else:
        Submit_Quantity = len(Times_IDRange_Todo)
    for i in np.arange(Submit_Quantity):
        ResultantFolder_I = ResultantFolder + '/Time_' + str(Times_IDRange_Todo[i])
        #Option = ' -V -o "' + ResultantFolder_I + '/perm_' + str(Times_IDRange_Todo[i]) + '.o" -e "' + ResultantFolder_I + '/perm_' + str(Times_IDRange_Todo[i]) + '.e"';
        #cmd = 'qsub ' + ResultantFolder_I + '/script.sh ' + QueueOptions + ' -N perm_' + str(Times_IDRange_Todo[i]) + Option;
        #print(cmd);
        #os.system(cmd)
        os.system('at -f "' + ResultantFolder_I + '/script.sh" now')
    Finished_Quantity = 0;
    while 1:        
        for i in np.arange(len(Finish_File)):
             if os.path.exists(Finish_File[i]):
                 Finished_Quantity = Finished_Quantity + 1
                 print(Finish_File[i])
                 del(Finish_File[i])
                 print(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
                 print('Finish quantity = ' + str(Finished_Quantity))
                 if Submit_Quantity < len(Times_IDRange_Todo):
                     ResultantFolder_I = ResultantFolder + '/Time_' + str(Times_IDRange_Todo[Submit_Quantity]);
                     #Option = ' -V -o "' + ResultantFolder_I + '/perm_' + str(Times_IDRange_Todo[Submit_Quantity]) + '.o" -e "' + ResultantFolder_I + '/perm_' + str(Times_IDRange_Todo[Submit_Quantity]) + '.e"';     
                     #cmd = 'qsub ' + ResultantFolder_I + '/script.sh ' + QueueOptions + ' -N perm_' + str(Times_IDRange_Todo[Submit_Quantity]) + Option
                     #print(cmd);
                     #os.system(cmd);
                     os.system('at -f "' + ResultantFolder_I + '/script.sh" now')
                     Submit_Quantity = Submit_Quantity + 1
                 break;
        if Finished_Quantity >= Jobs_Quantity:
            break;    

def Lasso_KFold_Sort_Permutation_Sub(Subjects_Data_Mat_Path, Subjects_Score, Fold_Quantity, Alpha_Range, ResultantFolder, Parallel_Quantity):
    #
    # For permutation test, This function will call 'Lasso_KFold_Sort' function
    #
    # Subjects_Data_Mat_Path:
    #     The path of .mat file that contain a variable named 'Subjects_Data'
    #     Variable 'Subjects_Data' is a n*m matrix, n is subjects quantity, m is features quantity
    # Other variables are the same with function 'Lasso_KFold_Sort'
    #

    data = sio.loadmat(Subjects_Data_Mat_Path)
    Subjects_Data = data['Subjects_Data']
    Lasso_KFold_Sort(Subjects_Data, Subjects_Score, Fold_Quantity, Alpha_Range, ResultantFolder, Parallel_Quantity, 1);

def Lasso_KFold_Sort(Subjects_Data, Subjects_Score, Fold_Quantity, Alpha_Range, ResultantFolder, Parallel_Quantity, Permutation_Flag):
    #
    # Lasso regression with K-fold cross-validation
    # K-fold cross-validation is random, as the split of all subjects into K groups is random
    # Here we first sorted subjects according to their scores, and then split the subjects into K groups according to the order
    # For example, 1st, (k+1)th, ... are the first group; 2nd, (k+2)th, ... are the second group, ...
    # With this method, the split into K fold can be fixed. 
    # However, for prediction, we can not make sure that any new individuals will with the same distribution with our training samples.
    # Therefore, we generally use sorted K-folder (just this method) as main result and the repeated random K-fold as validation
    #
    # Subjects_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Subjects_Score:
    #     n*1 vector, n is subjects quantity
    # Fold_Quantity:
    #     Fold quantity for the cross-validation
    #     5 or 10 is recommended generally, the small the better accepted by community, but the results may be worse as traning samples are fewer
    # Alpha_Range:
    #     Range of alpha, the regularization parameter balancing the training error and L1 penalty
    #     Our previous paper used (2^(-10), 2^(-9), ..., 2^4, 2^5), see Cui and Gong (2018), NeuroImage
    # ResultantFolder:
    #     Path of the folder storing the results
    # Parallel_Quantity:
    #     Parallel multi-cores on one single computer, at least 1
    # Permutation_Flag:
    #     1: this is for permutation, then the socres will be permuted
    #     0: this is not for permutation
    #

    if not os.path.exists(ResultantFolder):
            os.mkdir(ResultantFolder)
    Subjects_Quantity = len(Subjects_Score)
    # Sort the subjects score
    Sorted_Index = np.argsort(Subjects_Score)
    Subjects_Data = Subjects_Data[Sorted_Index, :]
    Subjects_Score = Subjects_Score[Sorted_Index]

    EachFold_Size = np.int(np.fix(np.divide(Subjects_Quantity, Fold_Quantity)))
    MaxSize = EachFold_Size * Fold_Quantity
    EachFold_Max = np.ones(Fold_Quantity, np.int) * MaxSize
    tmp = np.arange(Fold_Quantity - 1, -1, -1)
    EachFold_Max = EachFold_Max - tmp;
    Remain = np.mod(Subjects_Quantity, Fold_Quantity)
    for j in np.arange(Remain):
        EachFold_Max[j] = EachFold_Max[j] + Fold_Quantity
    
    Fold_Corr = [];
    Fold_MAE = [];

    for j in np.arange(Fold_Quantity):

        Fold_J_Index = np.arange(j, EachFold_Max[j], Fold_Quantity)
        Subjects_Data_test = Subjects_Data[Fold_J_Index, :]
        Subjects_Score_test = Subjects_Score[Fold_J_Index]
        Subjects_Data_train = np.delete(Subjects_Data, Fold_J_Index, axis=0)
        Subjects_Score_train = np.delete(Subjects_Score, Fold_J_Index) 

        if Permutation_Flag:
            # If doing permutation, the training scores should be permuted, while the testing scores remain
            Subjects_Index_Random = np.arange(len(Subjects_Score_train));
            np.random.shuffle(Subjects_Index_Random);
            Subjects_Score_train = Subjects_Score_train[Subjects_Index_Random]
            if j == 0:
                RandIndex = {'Fold_0': Subjects_Index_Random}
            else:
                RandIndex['Fold_' + str(j)] = Subjects_Index_Random

        Optimal_Alpha, Inner_Corr, Inner_MAE_inv = Lasso_OptimalAlpha_KFold(Subjects_Data_train, Subjects_Score_train, Fold_Quantity, Alpha_Range, ResultantFolder, Parallel_Quantity)

        normalize = preprocessing.MinMaxScaler()
        Subjects_Data_train = normalize.fit_transform(Subjects_Data_train)
        Subjects_Data_test = normalize.transform(Subjects_Data_test)

        clf = linear_model.Lasso(alpha = Optimal_Alpha)
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

def Lasso_APredictB_Permutation(Training_Data, Training_Score, Testing_Data, Testing_Score, Times_IDRange, Alpha_Range, Nested_Fold_Quantity, ResultantFolder, Parallel_Quantity):
    #
    # Permutation test for 'Lasso_APredictB'
    #
    
    if not os.path.exists(ResultantFolder):
        os.mkdir(ResultantFolder)
    for i in np.arange(len(Times_IDRange)):
        ResultantFolder_I = ResultantFolder + '/Time_' + str(Times_IDRange[i])
        if not os.path.exists(ResultantFolder_I):
            os.mkdir(ResultantFolder_I)
        if not os.path.exists(ResultantFolder_I + '/APredictB.mat'):
            Lasso_APredictB(Training_Data, Training_Score, Testing_Data, Testing_Score, Alpha_Range, Nested_Fold_Quantity, ResultantFolder_I, Parallel_Quantity, 1)

def Lasso_APredictB(Training_Data, Training_Score, Testing_Data, Testing_Score, Alpha_Range, Nested_Fold_Quantity, ResultantFolder, Parallel_Quantity, Permutation_Flag):
    #
    # Lasso regression with training data to predict testing data
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
    #     Range of alpha, the regularization parameter balancing the training error and L1 penalty
    #     Our previous paper used (2^(-10), 2^(-9), ..., 2^4, 2^5), see Cui and Gong (2018), NeuroImage
    # Nested_Fold_Quantity:
    #     Fold quantity for the nested cross-validation, which was used to select the optimal parameter
    #     5 or 10 is recommended generally, the small the better accepted by community, but the results may be worse as traning samples are fewer
    # ResultantFolder:
    #     Path of the folder storing the results
    # Parallel_Quantity:
    #     Parallel multi-cores on one single computer, at least 1
    # Permutation_Flag:
    #     1: this is for permutation, then the socres will be permuted
    #     0: this is not for permutation
    #
    
    if not os.path.exists(ResultantFolder):
        os.mkdir(ResultantFolder)

    if Permutation_Flag:
        # If do permutation, the training scores should be permuted, while the testing scores remain
        # Fold12
        Training_Index_Random = np.arange(len(Training_Score))
        np.random.shuffle(Training_Index_Random)
        Training_Score = Training_Score[Training_Index_Random]
        Random_Index = {'Training_Index_Random': Training_Index_Random}
        sio.savemat(ResultantFolder + '/Random_Index.mat', Random_Index);

    # Select optimal alpha & L1_ratio using inner fold cross validation
    Optimal_Alpha, Inner_Corr, Inner_MAE_inv = Lasso_OptimalAlpha_KFold(Training_Data, Training_Score, Nested_Fold_Quantity, Alpha_Range, ResultantFolder, Parallel_Quantity)

    Scale = preprocessing.MinMaxScaler()
    Training_Data = Scale.fit_transform(Training_Data)
    Testing_Data = Scale.transform(Testing_Data)  
    
    clf = linear_model.Lasso(alpha=Optimal_Alpha)
    clf.fit(Training_Data, Training_Score)
    Predict_Score = clf.predict(Testing_Data)

    Predict_Corr = np.corrcoef(Predict_Score, Testing_Score)
    Predict_Corr = Predict_Corr[0,1]
    Predict_MAE = np.mean(np.abs(np.subtract(Predict_Score, Testing_Score)))
    Predict_result = {'Test_Score':Testing_Score, 'Predict_Score':Predict_Score, 'Weight':clf.coef_, 'Predict_Corr':Predict_Corr, 'Predict_MAE':Predict_MAE, 'alpha':Optimal_Alpha}
    sio.savemat(ResultantFolder+'/APredictB.mat', Predict_result)
    return (Predict_Corr, Predict_MAE)

def Lasso_OptimalAlpha_KFold(Training_Data, Training_Score, Fold_Quantity, Alpha_Range, ResultantFolder, Parallel_Quantity):
    #
    # Select optimal regularization parameter using nested cross-validation
    #
    # Training_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Training_Score:
    #     n*1 vector, n is subjects quantity
    # Fold_Quantity:
    #     Fold quantity for the cross-validation
    #     5 or 10 is recommended generally, the small the better accepted by community, but the results may be worse as traning samples are fewer
    # Alpha_Range:
    #     Range of alpha, the regularization parameter balancing the training error and L1 penalty
    #     Our previous paper used (2^(-10), 2^(-9), ..., 2^4, 2^5), see Cui and Gong (2018), NeuroImage
    # ResultantFolder:
    #     Path of the folder storing the results
    # Parallel_Quantity:
    #     Parallel multi-cores on one single computer, at least 1
    #
    
    Subjects_Quantity = len(Training_Score)
    Sorted_Index = np.argsort(Training_Score)
    Training_Data = Training_Data[Sorted_Index, :]
    Training_Score = Training_Score[Sorted_Index]
    
    Inner_EachFold_Size = np.int(np.fix(np.divide(Subjects_Quantity, Fold_Quantity)))
    MaxSize = Inner_EachFold_Size * Fold_Quantity
    EachFold_Max = np.ones(Fold_Quantity, np.int) * MaxSize
    tmp = np.arange(Fold_Quantity - 1, -1, -1)
    EachFold_Max = EachFold_Max - tmp
    Remain = np.mod(Subjects_Quantity, Fold_Quantity)
    for j in np.arange(Remain):
    	EachFold_Max[j] = EachFold_Max[j] + Fold_Quantity
    
    Inner_Corr = np.zeros((Fold_Quantity, len(Alpha_Range)))
    Inner_MAE_inv = np.zeros((Fold_Quantity, len(Alpha_Range)))
    Alpha_Quantity = len(Alpha_Range)
    for k in np.arange(Fold_Quantity):
        
        Inner_Fold_K_Index = np.arange(k, EachFold_Max[k], Fold_Quantity)
        Inner_Fold_K_Data_test = Training_Data[Inner_Fold_K_Index, :]
        Inner_Fold_K_Score_test = Training_Score[Inner_Fold_K_Index]
        Inner_Fold_K_Data_train = np.delete(Training_Data, Inner_Fold_K_Index, axis=0)
        Inner_Fold_K_Score_train = np.delete(Training_Score, Inner_Fold_K_Index)
        
        Scale = preprocessing.MinMaxScaler()
        Inner_Fold_K_Data_train = Scale.fit_transform(Inner_Fold_K_Data_train)
        Inner_Fold_K_Data_test = Scale.transform(Inner_Fold_K_Data_test)    
        
        Parallel(n_jobs=Parallel_Quantity,backend="threading")(delayed(Lasso_SubAlpha)(Inner_Fold_K_Data_train, Inner_Fold_K_Score_train, Inner_Fold_K_Data_test, Inner_Fold_K_Score_test, Alpha_Range[l], l, ResultantFolder) for l in np.arange(len(Alpha_Range)))        
        
        for l in np.arange(Alpha_Quantity):
            print(l)
            Fold_l_Mat_Path = ResultantFolder + '/Alpha_' + str(l) + '.mat';
            Fold_l_Mat = sio.loadmat(Fold_l_Mat_Path)
            Inner_Corr[k, l] = Fold_l_Mat['Corr'][0][0]
            Inner_MAE_inv[k, l] = Fold_l_Mat['MAE_inv']
            os.remove(Fold_l_Mat_Path)
            
        Inner_Corr = np.nan_to_num(Inner_Corr)
    Inner_Corr_Mean = np.mean(Inner_Corr, axis=0)
    Inner_Corr_Mean = (Inner_Corr_Mean - np.mean(Inner_Corr_Mean)) / np.std(Inner_Corr_Mean)
    Inner_MAE_inv_Mean = np.mean(Inner_MAE_inv, axis=0)
    Inner_MAE_inv_Mean = (Inner_MAE_inv_Mean - np.mean(Inner_MAE_inv_Mean)) / np.std(Inner_MAE_inv_Mean)
    Inner_Evaluation = Inner_Corr_Mean + Inner_MAE_inv_Mean
    
    Inner_Evaluation_Mat = {'Inner_Corr':Inner_Corr, 'Inner_MAE_inv':Inner_MAE_inv, 'Inner_Evaluation':Inner_Evaluation}
    sio.savemat(ResultantFolder + '/Inner_Evaluation.mat', Inner_Evaluation_Mat)
    
    Optimal_Alpha_Index = np.argmax(Inner_Evaluation) 
    Optimal_Alpha = Alpha_Range[Optimal_Alpha_Index]
    return (Optimal_Alpha, Inner_Corr, Inner_MAE_inv)

def Lasso_SubAlpha(Training_Data, Training_Score, Testing_Data, Testing_Score, Alpha, Alpha_ID, ResultantFolder):
    #
    # Sub-function for optimal regularization parameter selection
    #
    # Training_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Training_Score:
    #     n*1 vector, n is subjects quantity
    # Testing_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Testing_Score:
    #     n*1 vector, n is subjects quantity
    # Alpha:
    #     Value of alpha to test
    # Alpha_ID:
    #     The indice of the alpha we tested in the alpha range
    # ResultantFolder:
    #     Folder to storing the results
    #

    clf = linear_model.Lasso(alpha=Alpha)
    clf.fit(Training_Data, Training_Score)
    Predict_Score = clf.predict(Testing_Data)
    Fold_Corr = np.corrcoef(Predict_Score, Testing_Score)
    Fold_Corr = Fold_Corr[0,1]
    Fold_MAE_inv = np.divide(1, np.mean(np.abs(Predict_Score - Testing_Score)))
    Fold_result = {'Corr': Fold_Corr, 'MAE_inv':Fold_MAE_inv}
    ResultantFile = ResultantFolder + '/Alpha_' + str(Alpha_ID) + '.mat'
    sio.savemat(ResultantFile, Fold_result)
    
def Lasso_Weight(Subjects_Data, Subjects_Score, Alpha_Range, Nested_Fold_Quantity, ResultantFolder, Parallel_Quantity):
    #
    # Function to generate the contribution weight of all features
    # We generally use all samples to construct a new model to extract the weight of all features
    #
    # Subjects_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Subjects_Score:
    #     n*1 vector, n is subjects quantity
    # Alpha_Range:
    #     Range of alpha, the regularization parameter balancing the training error and L1 penalty
    #     Our previous paper used (2^(-10), 2^(-9), ..., 2^4, 2^5), see Cui and Gong (2018), NeuroImage
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
    Optimal_Alpha, Inner_Corr, Inner_MAE_inv = Lasso_OptimalAlpha_KFold(Subjects_Data, Subjects_Score, Nested_Fold_Quantity, Alpha_Range, ResultantFolder, Parallel_Quantity)

    Scale = preprocessing.MinMaxScaler()
    Subjects_Data = Scale.fit_transform(Subjects_Data)
    clf = linear_model.Lasso(alpha=Optimal_Alpha)
    clf.fit(Subjects_Data, Subjects_Score)
    Weight = clf.coef_ / np.sqrt(np.sum(clf.coef_ **2))
    Weight_result = {'w_Brain':Weight, 'alpha':Optimal_Alpha}
    sio.savemat(ResultantFolder + '/w_Brain.mat', Weight_result)
    return;
