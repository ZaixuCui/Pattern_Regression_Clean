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
import time
from sklearn import linear_model
from sklearn import preprocessing
from joblib import Parallel, delayed

def ElasticNet_LOOCV_Permutation(Subjects_Data, Subjects_Score, Times_IDRange, Alpha_Range, L1_ratio_Range, ResultantFolder, Parallel_Quantity, Max_Queued, QueueOptions):
     
    #
    # Elastic-net regression with K-fold cross-validation
    #
    # Subjects_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Subjects_Score:
    #     n*1 vector, n is subjects quantity
    # Times_IDRange:
    #     The index of permutation test, for example np.arange(1000)
    # Alpha_Range:
    #     Range of alpha, the regularization parameter balancing the training error and regularization penalty
    #     Our previous paper used (2^(-10), 2^(-9), ..., 2^4, 2^5), see Cui and Gong (2018), NeuroImage
    # L1_ratio_Range:
    #     Range of l1 ratio, the parameter balancing l1 and l2 penalty
    #     Our previous paper 10 values in the range [0.2,1], see Cui et al., (2018), Cerebral Cortex
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
            Configuration_Mat = {'Subjects_Data_Mat_Path': Subjects_Data_Mat_Path, 'Subjects_Score': Subjects_Score, \
                'Alpha_Range': Alpha_Range, 'L1_ratio_Range': L1_ratio_Range, 'ResultantFolder_I': ResultantFolder_I, 'Parallel_Quantity': Parallel_Quantity};
            sio.savemat(ResultantFolder_I + '/Configuration.mat', Configuration_Mat)
            system_cmd = 'python3 -c ' + '\'import sys;\
                sys.path.append("' + os.getcwd() + '");\
                from ElasticNet_CZ_Sort import ElasticNet_KFold_Sort_Permutation_Sub;\
                import os;\
                import scipy.io as sio;\
                configuration = sio.loadmat("' + ResultantFolder_I + '/Configuration.mat");\
                Subjects_Data_Mat_Path = configuration["Subjects_Data_Mat_Path"];\
                Subjects_Score = configuration["Subjects_Score"];\
                Alpha_Range = configuration["Alpha_Range"];\
                L1_ratio_Range = configuration["L1_ratio_Range"];\
                ResultantFolder_I = configuration["ResultantFolder_I"];\
                Parallel_Quantity = configuration["Parallel_Quantity"];\
                ElasticNet_KFold_Sort_Permutation_Sub(Subjects_Data_Mat_Path[0], Subjects_Score[0], Alpha_Range[0], L1_ratio_Range[0], ResultantFolder_I[0], Parallel_Quantity[0][0])\' ';
            system_cmd = system_cmd + ' > "' + ResultantFolder_I + '/ElasticNet.log" 2>&1\n'
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

def ElasticNet_LOOCV_Permutation_Sub(Subjects_Data_Mat_Path, Subjects_Score, Alpha_Range, L1_ratio_Range, ResultantFolder, Parallel_Quantity):
    #
    # For permutation test, This function will call 'ElasticNet_KFold_Sort' function
    #
    # Subjects_Data_Mat_Path:
    #     The path of .mat file that contain a variable named 'Subjects_Data'
    #     Variable 'Subjects_Data' is a n*m matrix, n is subjects quantity, m is features quantity
    # Other variables are the same with function 'ElasticNet_KFold_Sort'
    #
    
    data = sio.loadmat(Subjects_Data_Mat_Path)
    Subjects_Data = data['Subjects_Data']
    ElasticNet_LOOCV(Subjects_Data, Subjects_Score, Alpha_Range, L1_ratio_Range, ResultantFolder, Parallel_Quantity, 1); 

def ElasticNet_LOOCV(Subjects_Data, Subjects_Score, Alpha_Range, L1_ratio_Range, ResultantFolder, Parallel_Quantity, Permutation_Flag):
    #
    # Elastic-Net regression with leave-one-out cross-validation
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
    
    Predicted_Score = np.zeros((1, Subjects_Quantity))
    Predicted_Score = Predicted_Score[0]
    for j in np.arange(Subjects_Quantity):

        Subjects_Data_test = Subjects_Data[j, :]
        Subjects_Data_test = Subjects_Data_test.reshape(1,-1)
        Subjects_Score_test = Subjects_Score[j]
        Subjects_Data_train = np.delete(Subjects_Data, j, axis=0)
        Subjects_Score_train = np.delete(Subjects_Score, j) 

        if Permutation_Flag:
            # If doing permutation, the training scores should be permuted, while the testing scores remain
            Subjects_Index_Random = np.arange(len(Subjects_Score_train));
            np.random.shuffle(Subjects_Index_Random);
            Subjects_Score_train = Subjects_Score_train[Subjects_Index_Random]
            if j == 0:
                RandIndex = {'Fold_0': Subjects_Index_Random}
            else:
                RandIndex['Fold_' + str(j)] = Subjects_Index_Random  

        Optimal_Alpha, Optimal_L1_ratio = ElasticNet_OptimalAlpha_LOOCV(Subjects_Data_train, Subjects_Score_train, Alpha_Range, L1_ratio_Range, ResultantFolder, Parallel_Quantity)

        normalize = preprocessing.MinMaxScaler()
        Subjects_Data_train = normalize.fit_transform(Subjects_Data_train)
        Subjects_Data_test = normalize.transform(Subjects_Data_test)

        clf = linear_model.ElasticNet(alpha=Optimal_Alpha, l1_ratio=Optimal_L1_ratio)
        clf.fit(Subjects_Data_train, Subjects_Score_train)
        Fold_J_Score = clf.predict(Subjects_Data_test)
        Predicted_Score[j] = Fold_J_Score[0]

    Corr = np.corrcoef(Predicted_Score, Subjects_Score)
    Corr = Corr[0,1]
    MAE = np.mean(np.abs(np.subtract(Predicted_Score, Subjects_Score)))
    
    Res_NFold = {'Corr':Corr, 'MAE':MAE, 'Test_Score':Subjects_Score, 'Predicted_Score':Predicted_Score};
    ResultantFile = os.path.join(ResultantFolder, 'Res_NFold.mat')
    sio.savemat(ResultantFile, Res_NFold)
    return (Corr, MAE)  

def ElasticNet_OptimalAlpha_LOOCV(Training_Data, Training_Score, Alpha_Range, L1_ratio_Range, ResultantFolder, Parallel_Quantity):
    #
    # Select optimal regularization parameter using LOOCV
    #
    # Training_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Training_Score:
    #     n*1 vector, n is subjects quantity
    # Alpha_Range:
    #     Range of alpha, the regularization parameter balancing the training error and L2 penalty
    #     Our previous paper used (2^(-10), 2^(-9), ..., 2^4, 2^5), see Cui and Gong (2018), NeuroImage
    # L1_ratio_Range:
    #     Range of l1 ratio, the parameter balancing l1 and l2 penalty
    #     Our previous paper 10 values in the range [0.2,1], see Cui et al., (2018), Cerebral Cortex
    # ResultantFolder:
    #     Path of the folder storing the results
    # Parallel_Quantity:
    #     Parallel multi-cores on one single computer, at least 1
    #
   
    Subjects_Quantity = len(Training_Score)
     
    Parameter_Combination_Quantity = len(Alpha_Range) * len(L1_ratio_Range) 
    Inner_Predicted_Score = np.zeros((Subjects_Quantity, Parameter_Combination_Quantity))
    for k in np.arange(Subjects_Quantity):
        
        Inner_Fold_K_Data_test = Training_Data[k, :]
        Inner_Fold_K_Data_test = Inner_Fold_K_Data_test.reshape(1,-1)
        Inner_Fold_K_Score_test = Training_Score[k]
        Inner_Fold_K_Data_train = np.delete(Training_Data, k, axis=0)
        Inner_Fold_K_Score_train = np.delete(Training_Score, k)
        
        Scale = preprocessing.MinMaxScaler()
        Inner_Fold_K_Data_train = Scale.fit_transform(Inner_Fold_K_Data_train)
        Inner_Fold_K_Data_test = Scale.transform(Inner_Fold_K_Data_test)    
        
        Parallel(n_jobs=Parallel_Quantity,backend="threading")(delayed(ElasticNet_SubAlpha_LOOCV)(Inner_Fold_K_Data_train, Inner_Fold_K_Score_train, Inner_Fold_K_Data_test, Inner_Fold_K_Score_test, Alpha_Range, L1_ratio_Range, l, ResultantFolder) for l in np.arange(Parameter_Combination_Quantity))        
        for l in np.arange(Parameter_Combination_Quantity):         
            print(l)
            Fold_l_Mat_Path = ResultantFolder + '/Parameter_' + str(l) + '.mat';
            Fold_l_Mat = sio.loadmat(Fold_l_Mat_Path)
            Inner_Predicted_Score[k, l] = Fold_l_Mat['Predicted_Score']
            os.remove(Fold_l_Mat_Path)
        
    Inner_Evaluation = np.zeros((1, Parameter_Combination_Quantity))
    Inner_Evaluation = Inner_Evaluation[0]
    for l in np.arange(Parameter_Combination_Quantity):
        Corr_tmp = np.corrcoef(Inner_Predicted_Score[:,l], Training_Score)
        Inner_Evaluation[l] = Corr_tmp[0,1]
    
    Inner_Evaluation_Mat = {'Inner_Evaluation':Inner_Evaluation}
    sio.savemat(ResultantFolder + '/Inner_Evaluation.mat', Inner_Evaluation_Mat)
    
    Optimal_Combination_Index = np.argmax(Inner_Evaluation) 
    
    Optimal_Alpha_Index = np.int64(np.ceil((Optimal_Combination_Index + 1) / len(L1_ratio_Range))) - 1
    Optimal_Alpha = Alpha_Range[Optimal_Alpha_Index]
    Optimal_L1_ratio_Index = np.mod(Optimal_Combination_Index, len(L1_ratio_Range))
    Optimal_L1_ratio = L1_ratio_Range[Optimal_L1_ratio_Index]
    return (Optimal_Alpha, Optimal_L1_ratio)

def ElasticNet_SubAlpha_LOOCV(Training_Data, Training_Score, Testing_Data, Testing_Score, Alpha_Range, L1_ratio_Range, Parameter_Combination_Index, ResultantFolder):
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
    Predicted_Score = clf.predict(Testing_Data)
    Fold_result = {'Predicted_Score': Predicted_Score}
    ResultantFile = ResultantFolder + '/Parameter_' + str(Parameter_Combination_Index) + '.mat'
    sio.savemat(ResultantFile, Fold_result)
    
def ElasticNet_Weight(Subjects_Data, Subjects_Score, Alpha_Range, L1_ratio_Range, ResultantFolder, Parallel_Quantity):
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
    # ResultantFolder:
    #     Path of the folder storing the results
    # Parallel_Quantity:
    #     Parallel multi-cores on one single computer, at least 1
    #

    if not os.path.exists(ResultantFolder):
        os.mkdir(ResultantFolder)

    # Select optimal alpha using inner fold cross validation
    Optimal_Alpha, Optimal_L1_ratio = ElasticNet_OptimalAlpha_LOOCV(Subjects_Data, Subjects_Score, Alpha_Range, L1_ratio_Range, ResultantFolder, Parallel_Quantity)
    
    Scale = preprocessing.MinMaxScaler()
    Subjects_Data = Scale.fit_transform(Subjects_Data)
    clf = linear_model.ElasticNet(alpha=Optimal_Alpha, l1_ratio=Optimal_L1_ratio)
    clf.fit(Subjects_Data, Subjects_Score)
    Weight = clf.coef_ / np.sqrt(np.sum(clf.coef_ **2))
    Weight_result = {'w_Brain':Weight, 'alpha':Optimal_Alpha}
    sio.savemat(ResultantFolder + '/w_Brain.mat', Weight_result)
    return;
