function Prediction = RVR_APredictB(Training_Data, Training_Scores, Testing_Data, Testing_Scores, Pre_Method, Permutation_Flag, ResultantFolder)
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
% Permutation_Flag:
%           1: do permutation testing, if permutation, we will permute the
%           scores acorss all subjects
%           0: not permutation testing
%
% Weight_Flag:
%           whether to compute the weight, 1 or 0
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
%                       Cui and Gong, 2018, NeuroImage; 
%                       Cui et al., 2016, Human Brain Mapping.
% (google scholar: https://scholar.google.com.hk/citations?user=j7amdXoAAAAJ&hl=zh-TW&oi=ao)
%

if nargin >= 7
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
    
% RVR training & predicting
d.train{1} = Training_Data_final * Training_Data_final';
d.test{1} = Testing_Data_final * Training_Data_final';
d.tr_targets = Training_Scores;
d.use_kernel = 1;
d.pred_type = 'regression';
output = prt_machine_rvr(d, []);

Prediction.Score = output.predictions;
Prediction.Corr = corr(output.predictions, Testing_Scores);
Prediction.MAE = mean(abs(output.predictions - Testing_Scores));
disp(['The correlation is ' num2str(Prediction.Corr)]);
disp(['The MAE is ' num2str(Prediction.MAE)]); 

if nargin >= 7
   save([ResultantFolder filesep 'Prediction.mat'], 'Prediction');
end
