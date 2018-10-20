function Prediction = SVR_NFolds_Sort_CSelect(Subjects_Data, Subjects_Scores, FoldQuantity, Pre_Method, C_Range, Weight_Flag, Permutation_Flag, ResultantFolder)
%
% Subject_Data:
%           m*n matrix
%           m is the number of subjects
%           n is the number of features
%
% Subject_Scores:
%           the continuous variable to be predicted
%
% FoldQuantity: 
%           The quantity of folds, 5 or 10 is recommended
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
% Weight_Flag:
%           1: Create the brain map of each region's contribution
%           0: don't need this map
%           If 1, we will train a new model using all subjects and extract
%           this map
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

if nargin >= 8
    if ~exist(ResultantFolder, 'dir')
        mkdir(ResultantFolder);
    end
end

[Subjects_Quantity, ~] = size(Subjects_Data);
    
% Split into N folds according to the behavioral scores
EachPart_Quantity = fix(Subjects_Quantity / FoldQuantity);
[~, SortedID] = sort(Subjects_Scores);
for j = 1:FoldQuantity
    Origin_ID{j} = SortedID([j : FoldQuantity : Subjects_Quantity]); 
end

for j = 1:FoldQuantity

    disp(['The ' num2str(j) ' fold!']);
    
    Training_data = Subjects_Data;
    Training_scores = Subjects_Scores;
    
    % Select training data and testing data
    test_data = Training_data(Origin_ID{j}, :);
    test_score = Training_scores(Origin_ID{j});
    Training_data(Origin_ID{j}, :) = [];
    Training_scores(Origin_ID{j}) = [];

    if Permutation_Flag
        Training_Quantity = length(Training_scores);
        RandIndex = randperm(Training_Quantity);
        Training_scores = Training_scores(RandIndex);
    end
    
    % Select optimal C
    for m = 1:length(C_Range)
        Prediction_Inner = SVR_NFolds_Sort(Training_data, Training_scores, FoldQuantity, Pre_Method, C_Range(m), 0);
        Inner_Corr_Array(m) = Prediction_Inner.Mean_Corr;
        Inner_MAE_Array(m) = Prediction_Inner.Mean_MAE;
    end
    Inner_MAE_inv_Array = 1./Inner_MAE_Array;
    Inner_Corr_norm_Array = (Inner_Corr_Array - mean(Inner_Corr_Array)) / std(Inner_Corr_Array);
    Inner_MAE_inv_norm_Array = (Inner_MAE_inv_Array - mean(Inner_MAE_inv_Array)) / std(Inner_MAE_inv_Array);
    Inner_Evaluation = Inner_Corr_norm_Array + Inner_MAE_inv_norm_Array;
    [~, Max_Index] = max(Inner_Evaluation);
    C_Optimal = C_Range(Max_Index);
    
    if strcmp(Pre_Method, 'Normalize')
        % Normalizing
        MeanValue = mean(Training_data);
        StandardDeviation = std(Training_data);
        [~, columns_quantity] = size(Training_data);
        for k = 1:columns_quantity
            Training_data(:, k) = (Training_data(:, k) - MeanValue(k)) / StandardDeviation(k);
        end
    elseif strcmp(Pre_Method, 'Scale')
        % Scaling to [0 1]
        MinValue = min(Training_data);
        MaxValue = max(Training_data);
        [~, columns_quantity] = size(Training_data);
        for k = 1:columns_quantity
            Training_data(:, k) = (Training_data(:, k) - MinValue(k)) / (MaxValue(k) - MinValue(k));
        end
    end
    
    % SVR training
    Training_data_final = double(Training_data);
    model = svmtrain(Training_scores, Training_data_final, ['-s 3 -t 0 -c ' num2str(C_Optimal)]);
    
    % Normalize test data
    if strcmp(Pre_Method, 'Normalize')
        % Normalizing
        MeanValue_New = repmat(MeanValue, length(test_score), 1);
        StandardDeviation_New = repmat(StandardDeviation, length(test_score), 1);
        test_data = (test_data - MeanValue_New) ./ StandardDeviation_New;
    elseif strcmp(Pre_Method, 'Scale')
        % Scale
        MaxValue_New = repmat(MaxValue, length(test_score), 1);
        MinValue_New = repmat(MinValue, length(test_score), 1);
        test_data = (test_data - MinValue_New) ./ (MaxValue_New - MinValue_New);
    end
    test_data_final = double(test_data);
    % Predict test data
    [Predicted_Scores, ~, ~] = svmpredict(test_score, test_data_final, model);
    Prediction.Origin_ID{j} = Origin_ID{j};
    Prediction.Score{j} = Predicted_Scores;
    Prediction.Corr(j) = corr(Predicted_Scores, test_score);
    Prediction.MAE(j) = mean(abs(Predicted_Scores - test_score));
    Prediction.C_Optimal(j) = C_Optimal;

end

Prediction.Mean_Corr = mean(Prediction.Corr);
Prediction.Mean_MAE = mean(Prediction.MAE);
if nargin >= 8
    save([ResultantFolder filesep 'Prediction.mat'], 'Prediction');
    disp(['The correlation is ' num2str(Prediction.Mean_Corr)]);
    disp(['The MAE is ' num2str(Prediction.Mean_MAE)]);
    % Calculating w
    if Weight_Flag
        W_Calculate_SVR_CSelect(Subjects_Data, Subjects_Scores, Pre_Method, C_Range, ResultantFolder); 
    end
end
