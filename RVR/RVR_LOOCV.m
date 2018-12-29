function Prediction = RVR_LOOCV(Subjects_Data, Subjects_Scores, Pre_Method, Weight_Flag, Permutation_Flag, ResultantFolder)
%
% Subject_Data:
%           m*n matrix
%           m is the number of subjects
%           n is the number of features
%
% Subject_Scores:
%           the continuous variable to be predicted, [1*m]
%
% Pre_Method:
%          'Normalize', 'Scale', 'None'
%
% Weight_Flag:
%           whether to compute the weight, 1 or 0
%
% Permutation_Flag:
%           1: do permutation testing, if permutation, we will permute the
%           scores acorss all subjects
%           0: not for permutation
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

if nargin >= 5
    if ~exist(ResultantFolder, 'dir')
        mkdir(ResultantFolder);
    end
end

[Subjects_Quantity, Feature_Quantity] = size(Subjects_Data);

for i = 1:Subjects_Quantity
    
    disp(['The ' num2str(i) ' subject!']);
    
    Training_data = Subjects_Data;
    Training_scores = Subjects_Scores;
    
    % Select training data and testing data
    test_data = Training_data(i, :);
    test_score = Training_scores(i);
    Training_data(i, :) = [];
    Training_scores(i) = [];

    if Permutation_Flag
        RandIndex = randperm(Subjects_Quantity - 1);
        Training_scores = Training_scores(RandIndex);
    end
        
    if strcmp(Pre_Method, 'Normalize')
        %Normalizing
        MeanValue = mean(Training_data);
        StandardDeviation = std(Training_data);
        [~, columns_quantity] = size(Training_data);
        for j = 1:columns_quantity
            Training_data(:, j) = (Training_data(:, j) - MeanValue(j)) / StandardDeviation(j);
        end
    elseif strcmp(Pre_Method, 'Scale')
        % Scaling to [0 1]
        MinValue = min(Training_data);
        MaxValue = max(Training_data);
        [~, columns_quantity] = size(Training_data);
        for j = 1:columns_quantity
            Training_data(:, j) = (Training_data(:, j) - MinValue(j)) / (MaxValue(j) - MinValue(j));
        end
    end
    Training_data_final = double(Training_data);

    % Normalize test data
    if strcmp(Pre_Method, 'Normalize')
        % Normalizing
        test_data = (test_data - MeanValue) ./ StandardDeviation;
    elseif strcmp(Pre_Method, 'Scale')
        % Scale
        test_data = (test_data - MinValue) ./ (MaxValue - MinValue);
    end
    test_data_final = double(test_data);

    % RVR training & predicting
    d.train{1} = Training_data_final * Training_data_final';
    d.test{1} = test_data_final * Training_data_final';
    d.tr_targets = Training_scores;
    d.use_kernel = 1;
    d.pred_type = 'regression';
    output = prt_machine_rvr(d, []);
    
    Predicted_Scores(i) = output.predictions; 
    
end

Prediction.Score = Predicted_Scores;
[Prediction.Corr, ~] = corr(Predicted_Scores', Subjects_Scores);
Prediction.MAE = mean(abs((Predicted_Scores' - Subjects_Scores)));

if nargin >= 5
    save([ResultantFolder filesep 'Prediction_res.mat'], 'Prediction');
    disp(['The correlation is ' num2str(Prediction.Corr)]);
    disp(['The MAE is ' num2str(Prediction.MAE)]);
    % Calculating w
    if Weight_Flag
        W_Calculate_RVR(Subjects_Data, Subjects_Scores, Pre_Method, ResultantFolder); 
    end
end
