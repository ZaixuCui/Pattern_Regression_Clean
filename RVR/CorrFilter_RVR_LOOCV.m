function Prediction = CorrFilter_RVR_LOOCV(Subjects_Data, Subjects_Scores, CoefThreshold, Pre_Method, ResultantFolder)
%
% Subject_Data:
%           m*n matrix
%           m is the number of subjects
%           n is the number of features
%
% Subject_Scores:
%           the continuous variable to be predicted
%
% CoefThreshold:
%           Threshold for the feature selection
%           We used pearson correlation for feature selection here. 
%           Here we shold set a threshold of pearson correlation
%           coefficient, and all features with correlation smaller than
%           this value will be removed
%
% Pre_Method:
%           'Normalize' or 'Scale'
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
if nargin >= 5
    if ~exist(ResultantFolder, 'dir')
        mkdir(ResultantFolder);
    end
end

[Subjects_Quantity, Feature_Quantity] = size(Subjects_Data);

Feature_Frequency = zeros(1, Feature_Quantity);
for i = 1:Subjects_Quantity
    
    disp(['The ' num2str(i) ' iteration!']);
    
    Training_data = Subjects_Data;
    Training_scores = Subjects_Scores;
    
    % Select training data and testing data
    test_data = Training_data(i, :);
    test_score = Training_scores(i);
    Training_data(i, :) = [];
    Training_scores(i) = [];

    coef = corr(Training_data, Training_scores);
    coef(find(isnan(coef))) = 0;
    RetainID{i} = find(abs(coef) > CoefThreshold);
    Training_data_New = Training_data(:, RetainID{i});
    Selected_Mask = zeros(1, Feature_Quantity);
    Selected_Mask(RetainID{i}) = 1;
    Feature_Frequency = Feature_Frequency + Selected_Mask;
    
    % Normalizing
    if strcmp(Pre_Method, 'Normalize')
        % Normalizing
        MeanValue = mean(Training_data_New);
        StandardDeviation = std(Training_data_New);
        [~, columns_quantity] = size(Training_data_New);
        for j = 1:columns_quantity
            Training_data_New(:, j) = (Training_data_New(:, j) - MeanValue(j)) / StandardDeviation(j);
        end
    elseif strcmp(Pre_Method, 'Scale')
        % Scaling to [0 1]
        MinValue = min(Training_data_New);
        MaxValue = max(Training_data_New);
        [~, columns_quantity] = size(Training_data_New);
        for j = 1:columns_quantity
            Training_data_New(:, j) = (Training_data_New(:, j) - MinValue(j)) / (MaxValue(j) - MinValue(j));
        end
    end
    
    test_data_New = test_data(:, RetainID{i});
    % Normalizing
    if strcmp(Pre_Method, 'Normalize')
        % Normalizing
        test_data_New = (test_data_New - MeanValue) ./ StandardDeviation;
    elseif strcmp(Pre_Method, 'Scale')
        % Scale
        test_data_New = (test_data_New - MinValue) ./ (MaxValue - MinValue);
    end
    
    % RVR training & predicting
    d.train{1} = Training_data_New * Training_data_New';
    d.test{1} = test_data_New * Training_data_New';
    d.tr_targets = Training_scores;
    d.use_kernel = 1;
    d.pred_type = 'regression';
    output = prt_machine_rvr(d, []);
    
    Predicted_Scores(i) = output.predictions;
    
end
Prediction.Score = Predicted_Scores;
[Prediction.Corr, ~] = corr(Predicted_Scores', Subjects_Scores);
Prediction.MAE = mean(abs((Predicted_Scores - Subjects_Scores)));
Prediction.Feature_Frequency = Feature_Frequency;
if nargin >= 5
    save([ResultantFolder filesep 'Prediction_res.mat'], 'Prediction');
    disp(['The correlation is ' num2str(Prediction.Corr)]);
    disp(['The MSE is ' num2str(Prediction.MAE)]);
end

