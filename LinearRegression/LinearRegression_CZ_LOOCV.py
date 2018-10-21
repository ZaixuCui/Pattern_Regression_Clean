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
  
def LinearRegression_LOOCV_Permutation(Subjects_Data, Subjects_Score, Times_IDRange, ResultantFolder, Max_Queued, QueueOptions):
    
    #
    # Linear regression with leave-one-out cross-validation (LOOCV)
    #
    # Subjects_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Subjects_Score:
    #     n*1 vector, n is subjects quantity
    # Times_IDRange:
    #     The index of permutation test, for example np.arange(1000)
    # ResultantFolder:
    #     Path of the folder storing the results
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
            Configuration_Mat = {'Subjects_Data_Mat_Path': Subjects_Data_Mat_Path, 'Subjects_Score': Subjects_Score, 'ResultantFolder_I': ResultantFolder_I};
            sio.savemat(ResultantFolder_I + '/Configuration.mat', Configuration_Mat)
            system_cmd = 'python3 -c ' + '\'import sys;\
                sys.path.append("' + os.getcwd() + '");\
                from LinearRegression_CZ_Sort import LinearRegression_KFold_Sort_Permutation_Sub;\
                import os;\
                import scipy.io as sio;\
                configuration = sio.loadmat("' + ResultantFolder_I + '/Configuration.mat");\
                Subjects_Data_Mat_Path = configuration["Subjects_Data_Mat_Path"];\
                Subjects_Score = configuration["Subjects_Score"];\
                ResultantFolder_I = configuration["ResultantFolder_I"];\
                LinearRegression_LOOCV_Permutation_Sub(Subjects_Data_Mat_Path[0], Subjects_Score[0], ResultantFolder_I[0])\' ';
            system_cmd = system_cmd + ' > "' + ResultantFolder_I + '/LinearRegression.log" 2>&1\n'
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

def LinearRegression_LOOCV_Permutation_Sub(Subjects_Data_Mat_Path, Subjects_Score, ResultantFolder, Parallel_Quantity):
    #
    # For permutation test, This function will call 'LinearRegression_LOOCV' function
    #
    # Subjects_Data_Mat_Path:
    #     The path of .mat file that contain a variable named 'Subjects_Data'
    #     Variable 'Subjects_Data' is a n*m matrix, n is subjects quantity, m is features quantity
    # Other variables are the same with function 'LinearRegression_KFold_Sort'
    #

    data = sio.loadmat(Subjects_Data_Mat_Path)
    Subjects_Data = data['Subjects_Data']
    LinearRegression_LOOCV(Subjects_Data, Subjects_Score, ResultantFolder, 1);

def LinearRegression_LOOCV(Subjects_Data, Subjects_Score, ResultantFolder, Permutation_Flag):
    #
    # LinearRegression regression with leave-one-out cross-validation (LOOCV)   
    # Subjects_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Subjects_Score:
    #     n*1 vector, n is subjects quantity
    # ResultantFolder:
    #     Path of the folder storing the results
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

        normalize = preprocessing.MinMaxScaler()
        Subjects_Data_train = normalize.fit_transform(Subjects_Data_train)
        Subjects_Data_test = normalize.transform(Subjects_Data_test)

        clf = linear_model.LinearRegression()
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
