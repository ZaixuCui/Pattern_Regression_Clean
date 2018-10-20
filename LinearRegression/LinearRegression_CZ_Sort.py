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
  
def LinearRegression_KFold_Sort_Permutation(Subjects_Data, Subjects_Score, Times_IDRange, Fold_Quantity, ResultantFolder, Max_Queued, QueueOptions):
    #
    # Linear regression with K-fold cross-validation
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
            Configuration_Mat = {'Subjects_Data_Mat_Path': Subjects_Data_Mat_Path, 'Subjects_Score': Subjects_Score, 'Fold_Quantity': Fold_Quantity, \
                'ResultantFolder_I': ResultantFolder_I};
            sio.savemat(ResultantFolder_I + '/Configuration.mat', Configuration_Mat)
            system_cmd = 'python3 -c ' + '\'import sys;\
                sys.path.append("' + os.getcwd() + '");\
                from LeastSquares_CZ_Sort import LinearRegression_KFold_Sort_Permutation_Sub;\
                import os;\
                import scipy.io as sio;\
                configuration = sio.loadmat("' + ResultantFolder_I + '/Configuration.mat");\
                Subjects_Data_Mat_Path = configuration["Subjects_Data_Mat_Path"];\
                Subjects_Score = configuration["Subjects_Score"];\
                Fold_Quantity = configuration["Fold_Quantity"];\
                ResultantFolder_I = configuration["ResultantFolder_I"];\
                LinearRegression_KFold_Sort_Permutation_Sub(Subjects_Data_Mat_Path[0], Subjects_Score[0], Fold_Quantity[0][0], ResultantFolder_I[0])\' ';
            system_cmd = system_cmd + ' > "' + ResultantFolder_I + '/LeastSquares.log" 2>&1\n'
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

def LinearRegression_KFold_Sort_Permutation_Sub(Subjects_Data_Mat_Path, Subjects_Score, Fold_Quantity, ResultantFolder):
    #
    # For permutation test, This function will call 'LinearRegression_KFold_Sort' function
    #
    # Subjects_Data_Mat_Path:
    #     The path of .mat file that contain a variable named 'Subjects_Data'
    #     Variable 'Subjects_Data' is a n*m matrix, n is subjects quantity, m is features quantity
    # Other variables are the same with function 'LinearRegression_KFold_Sort'
    #
    data = sio.loadmat(Subjects_Data_Mat_Path)
    Subjects_Data = data['Subjects_Data']
    LinearRegression_KFold_Sort(Subjects_Data, Subjects_Score, Fold_Quantity, ResultantFolder, 1);

def LinearRegression_KFold_Sort(Subjects_Data, Subjects_Score, Fold_Quantity, ResultantFolder, Permutation_Flag):
    #
    # Linear regression with K-fold cross-validation
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
    # ResultantFolder:
    #     Path of the folder storing the results
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
    
def LinearRegression_APredictB_Permutation(Training_Data, Training_Score, Testing_Data, Testing_Score, Times_IDRange, ResultantFolder):
    #
    # Permutation test for 'LinearRegression_APredictB'
    #
   
    if not os.path.exists(ResultantFolder):
        os.mkdir(ResultantFolder)
    for i in np.arange(len(Times_IDRange)):
        ResultantFolder_I = ResultantFolder + '/Time_' + str(Times_IDRange[i])
        if not os.path.exists(ResultantFolder_I):
            os.mkdir(ResultantFolder_I)
        if not os.path.exists(ResultantFolder_I + '/APredictB.mat'):
            LinearRegression_APredictB(Training_Data, Training_Score, Testing_Data, Testing_Score, ResultantFolder_I, 1)
    
def LinearRegression_APredictB(Training_Data, Training_Score, Testing_Data, Testing_Score, ResultantFolder, Permutation_Flag):
    #
    # Linear regression with training data to predict testing data
    #
    # Training_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Training_Score:
    #     n*1 vector, n is subjects quantity
    # Testing_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Testing_Score:
    #     n*1 vector, n is subjects quantity
    # ResultantFolder:
    #     Path of the folder storing the results
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

    normalize = preprocessing.MinMaxScaler()
    Training_Data = normalize.fit_transform(Training_Data)
    Testing_Data = normalize.transform(Testing_Data)  
    
    clf = linear_model.LinearRegression()
    clf.fit(Training_Data, Training_Score)
    Predict_Score = clf.predict(Testing_Data)

    Predict_Corr = np.corrcoef(Predict_Score, Testing_Score)
    Predict_Corr = Predict_Corr[0,1]
    Predict_MAE = np.mean(np.abs(np.subtract(Predict_Score, Testing_Score)))
    Predict_result = {'Test_Score': Testing_Score, 'Predict_Score': Predict_Score, 'Weight': clf.coef_, 'Predict_Corr': Predict_Corr, 'Predict_MAE': Predict_MAE}
    sio.savemat(ResultantFolder+'/APredictB.mat', Predict_result)
    return
    
def LinearRegression_Weight(Subjects_Data, Subjects_Score, ResultantFolder):
    #
    # Function to generate the contribution weight of all features
    # We generally use all samples to construct a new model to extract the weight of all features
    #
    # Training_Data:
    #     n*m matrix, n is subjects quantity, m is features quantity
    # Training_Score:
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
