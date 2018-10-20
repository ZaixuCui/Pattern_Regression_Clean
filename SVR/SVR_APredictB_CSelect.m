function Prediction = SVR_APredictB_CSelect(Training_Data, Training_Scores, Testing_Data, Testing_Scores, Pre_Method, C_Range, Nested_Fold_Quantity, Permutation_Flag, ResultantFolder)
%
% Training_Data:
%           m*n matrix
%           m is the number of subjects
%           n is the number of features
%
% Training_Scores:
%           m*1 vector, the continuous variable of training subjects
%
% Testing_Data:
%           m*n matrix
%           m is the number of subjects
%           n is the number of features
%
% Testing_Scores:
%           m*1 vector, the continuous variable of testing subjects
%
% Pre_Method:
%           'Normalize', 'Scale', 'None'
%
% C_Range:
%           The range of parameter C, 
%           We used (2^-5, 2^-4, ..., 2^9, 2^10) in our previous paper, see
%           Cui and Gong, 2018, NeuroImage, also see Hsu et al., 2003, A
%           practical guide to support vector classification. 
%
% Nested_Fold_Quantity:
%           Fold quantity for cross-validation that was used for parameter
%           selection
%
% Permutation_Flag:
%           1: do permutation testing, if permutation, we will permute the
%           scores acorss all subjects
%           0: not permutation testing
%
% ResultantFolder:
%           the path of folder storing resultant files
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Written by Zaixu Cui: zaixucui@gmail.com;
%                       Zaixu.Cui@pennmedicine.upenn.edu
%
% If you use this code, please cite: 
%                       Cui et al., 2018, Cerebral Cortex; 
%                       Cui and Gong et al., 2018, NeuroImage; 
%                       Cui et al., 2016, Human Brain Mapping.
% (google scholar: https://scholar.google.com.hk/citations?user=j7amdXoAAAAJ&hl=zh-TW&oi=ao)
%

if nargin >= 9
    if ~exist(ResultantFolder, 'dir')
        mkdir(ResultantFolder);
    end
end

if Permutation_Flag
    Training_Quantity = length(Training_Scores);
    RandIndex = randperm(Training_Quantity);
    Training_Scores = Training_Scores(RandIndex);
end

if strcmp(Pre_Method, 'Normalize')
    % Normalizing
    MeanValue = mean(Training_Data);
    StandardDeviation = std(Training_Data);
    [~, columns_quantity] = size(Training_Data);
    for k = 1:columns_quantity
        Training_Data(:, k) = (Training_Data(:, k) - MeanValue(k)) / StandardDeviation(k);
    end
elseif strcmp(Pre_Method, 'Scale')
    % Scaling to [0 1]
    MinValue = min(Training_Data);
    MaxValue = max(Training_Data);
    [~, columns_quantity] = size(Training_Data);
    for k = 1:columns_quantity
        Training_Data(:, k) = (Training_Data(:, k) - MinValue(k)) / (MaxValue(k) - MinValue(k));
    end
end
Training_Data_final = double(Training_Data);
        
% Select optimal C
for m = 1:length(C_Range)
    Prediction_Inner = SVR_NFolds_Sort(Training_Data_final, Training_Scores, Nested_Fold_Quantity, Pre_Method, C_Range(m), 0);
    Inner_Corr_Array(m) = Prediction_Inner.Mean_Corr;
    Inner_MAE_Array(m) = Prediction_Inner.Mean_MAE;
end
Inner_MAE_inv_Array = 1./Inner_MAE_Array;
Inner_Corr_norm_Array = (Inner_Corr_Array - mean(Inner_Corr_Array)) / std(Inner_Corr_Array);
Inner_MAE_inv_norm_Array = (Inner_MAE_inv_Array - mean(Inner_MAE_inv_Array)) / std(Inner_MAE_inv_Array);
Inner_Evaluation = Inner_Corr_norm_Array + Inner_MAE_inv_norm_Array;
[~, Max_Index] = max(Inner_Evaluation);
C_Optimal = C_Range(Max_Index);

% SVR training
model = svmtrain(Training_Scores, Training_Data_final, ['-s 3 -t 0 -c ' num2str(C_Optimal)]);

% Normalize test data
if strcmp(Pre_Method, 'Normalize')
    % Normalizing
    MeanValue_New = repmat(MeanValue, length(Testing_Scores), 1);
    StandardDeviation_New = repmat(StandardDeviation, length(Testing_Scores), 1);
    Testing_Data = (Testing_Data - MeanValue_New) ./ StandardDeviation_New;
elseif strcmp(Pre_Method, 'Scale')
    % Scale
    MaxValue_New = repmat(MaxValue, length(Testing_Scores), 1);
    MinValue_New = repmat(MinValue, length(Testing_Scores), 1);
    Testing_Data = (Testing_Data - MinValue_New) ./ (MaxValue_New - MinValue_New);
end
Testing_Data_final = double(Testing_Data);

% Predict test data
[Predicted_Scores, ~, ~] = svmpredict(Testing_Scores, Testing_Data_final, model);
Prediction.Score = Predicted_Scores;
Prediction.Corr = corr(Predicted_Scores, Testing_Scores);
Prediction.MAE = mean(abs(Predicted_Scores - Testing_Scores));
Prediction.C_Optimal = C_Optimal;

disp(['The correlation is ' num2str(Prediction.Corr)]);
disp(['The MAE is ' num2str(Prediction.MAE)]);
    
if nargin >= 9
    save([ResultantFolder filesep 'Prediction.mat'], 'Prediction');
end
