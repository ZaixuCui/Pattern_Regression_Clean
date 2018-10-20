
import scipy.io as sio
import numpy as np
import os
import sys
sys.path.append('/Users/zaixucui/Dropbox/Pattern_Regression_Clean/LinearRegression');
sys.path.append('/Users/zaixucui/Dropbox/Pattern_Regression_Clean/ElasticNet');
sys.path.append('/Users/zaixucui/Dropbox/Pattern_Regression_Clean/Lasso');
sys.path.append('/Users/zaixucui/Dropbox/Pattern_Regression_Clean/Ridge');
import Ridge_CZ_Sort
import Ridge_CZ_RandomCV
import LinearRegression_CZ_Sort
import LinearRegression_CZ_RandomCV
import Lasso_CZ_Sort
import Lasso_CZ_RandomCV
import ElasticNet_CZ_Sort
import ElasticNet_CZ_RandomCV
import Ridge_CZ_LOOCV
import Lasso_CZ_LOOCV
import ElasticNet_CZ_LOOCV
import LinearRegression_CZ_LOOCV

Subjects_Data = np.random.rand(20, 5);
Subjects_Score = np.random.rand(20, 1);
Subjects_Score = np.transpose(Subjects_Score);
Subjects_Score = Subjects_Score[0];
# Range of parameters
Alpha_Range = np.exp2(np.arange(5) - 10);
L1_ratio_Range = np.linspace(0.2, 1, 5);
Fold_Quantity = 2;
Parallel_Quantity = 1;
Permutation_Flag = 0;
ResultantFolder = '/Users/zaixucui/Dropbox/Pattern_Regression_Clean/res';
Ridge_CZ_Sort.Ridge_KFold_Sort(Subjects_Data, Subjects_Score, Fold_Quantity, Alpha_Range, ResultantFolder, Parallel_Quantity, Permutation_Flag);
Lasso_CZ_Sort.Lasso_KFold_Sort(Subjects_Data, Subjects_Score, Fold_Quantity, Alpha_Range, ResultantFolder, Parallel_Quantity, Permutation_Flag);
LinearRegression_CZ_Sort.LinearRegression_KFold_Sort(Subjects_Data, Subjects_Score, Fold_Quantity, ResultantFolder, Permutation_Flag)
ElasticNet_CZ_Sort.ElasticNet_KFold_Sort(Subjects_Data, Subjects_Score, Fold_Quantity, Alpha_Range, L1_ratio_Range, ResultantFolder, Parallel_Quantity, Permutation_Flag);
CVRepeatTimes_ForInner = 5;
Ridge_CZ_RandomCV.Ridge_KFold_RandomCV_MultiTimes(Subjects_Data, Subjects_Score, Fold_Quantity, Alpha_Range, CVRepeatTimes_ForInner, ResultantFolder, Parallel_Quantity)
Lasso_CZ_RandomCV.Lasso_KFold_RandomCV(Subjects_Data, Subjects_Score, Fold_Quantity, Alpha_Range, CVRepeatTimes_ForInner, ResultantFolder, Parallel_Quantity)
ElasticNet_CZ_RandomCV.ElasticNet_KFold_RandomCV(Subjects_Data, Subjects_Score, Fold_Quantity, Alpha_Range, L1_ratio_Range, CVRepeatTimes_ForInner, ResultantFolder, Parallel_Quantity)
LinearRegression_CZ_RandomCV.LinearRegression_KFold_RandomCV(Subjects_Data, Subjects_Score, Fold_Quantity, ResultantFolder)
Ridge_CZ_LOOCV.Ridge_LOOCV(Subjects_Data, Subjects_Score, Alpha_Range, ResultantFolder, Parallel_Quantity, Permutation_Flag);
Lasso_CZ_LOOCV.Lasso_LOOCV(Subjects_Data, Subjects_Score, Alpha_Range, ResultantFolder, Parallel_Quantity, Permutation_Flag);
ElasticNet_CZ_LOOCV.ElasticNet_LOOCV(Subjects_Data, Subjects_Score, Alpha_Range, L1_ratio_Range, ResultantFolder, Parallel_Quantity, Permutation_Flag)
LinearRegression_CZ_LOOCV.LinearRegression_LOOCV(Subjects_Data, Subjects_Score, ResultantFolder, Permutation_Flag)

Training_Data = Subjects_Data;
Training_Score = Subjects_Score;
Testing_Data = Subjects_Data;
Testing_Score = Subjects_Score;
Nested_Fold_Quantity = 5;
Times_IDRange = np.arange(3);
AlphaRange = np.exp2(np.arange(16) - 10);
Ridge_CZ_Sort.Ridge_APredictB(Training_Data, Training_Score, Testing_Data, Testing_Score, AlphaRange, Nested_Fold_Quantity, ResultantFolder, Parallel_Quantity, Permutation_Flag);
Ridge_CZ_Sort.Ridge_APredictB_Permutation(Training_Data, Training_Score, Testing_Data, Testing_Score, Times_IDRange, AlphaRange, Nested_Fold_Quantity, ResultantFolder, Parallel_Quantity)
Lasso_CZ_Sort.Lasso_APredictB(Training_Data, Training_Score, Testing_Data, Testing_Score, AlphaRange, Nested_Fold_Quantity, ResultantFolder, Parallel_Quantity, Permutation_Flag);
Lasso_CZ_Sort.Lasso_APredictB_Permutation(Training_Data, Training_Score, Testing_Data, Testing_Score, Times_IDRange, AlphaRange, Nested_Fold_Quantity, ResultantFolder, Parallel_Quantity)
LinearRegression_CZ_Sort.LinearRegression_APredictB(Training_Data, Training_Score, Testing_Data, Testing_Score, ResultantFolder, Permutation_Flag)
LinearRegression_CZ_Sort.LinearRegression_APredictB_Permutation(Training_Data, Training_Score, Testing_Data, Testing_Score, Times_IDRange, ResultantFolder)
ElasticNet_CZ_Sort.ElasticNet_APredictB(Training_Data, Training_Score, Testing_Data, Testing_Score, AlphaRange, L1_ratio_Range, Nested_Fold_Quantity, ResultantFolder, Parallel_Quantity, Permutation_Flag)
ElasticNet_CZ_Sort.ElasticNet_APredictB_Permutation(Training_Data, Training_Score, Testing_Data, Testing_Score, Times_IDRange, AlphaRange, L1_ratio_Range, Nested_Fold_Quantity, ResultantFolder, Parallel_Quantity)

# Ridge weight
Ridge_CZ_Sort.Ridge_Weight(Subjects_Data, Subjects_Score, Alpha_Range, Nested_Fold_Quantity, ResultantFolder, Parallel_Quantity)
Lasso_CZ_Sort.Lasso_Weight(Subjects_Data, Subjects_Score, Alpha_Range, Nested_Fold_Quantity, ResultantFolder, Parallel_Quantity)
LinearRegression_CZ_Sort.LinearRegression_Weight(Subjects_Data, Subjects_Score, ResultantFolder)
Lasso_CZ_Sort.Lasso_Weight(Subjects_Data, Subjects_Score, Alpha_Range, Nested_Fold_Quantity, ResultantFolder, Parallel_Quantity)
ElasticNet_CZ_Sort.ElasticNet_Weight(Subjects_Data, Subjects_Score, Alpha_Range, L1_ratio_Range, Nested_Fold_Quantity, ResultantFolder, Parallel_Quantity);
Ridge_CZ_LOOCV.Ridge_Weight(Subjects_Data, Subjects_Score, Alpha_Range, ResultantFolder, Parallel_Quantity)
Lasso_CZ_LOOCV.Lasso_Weight(Subjects_Data, Subjects_Score, Alpha_Range, ResultantFolder, Parallel_Quantity)
ElasticNet_CZ_LOOCV.ElasticNet_Weight(Subjects_Data, Subjects_Score, Alpha_Range, L1_ratio_Range, ResultantFolder, Parallel_Quantity)
LinearRegression_CZ_LOOCV.LinearRegression_Weight(Subjects_Data, Subjects_Score, ResultantFolder)


