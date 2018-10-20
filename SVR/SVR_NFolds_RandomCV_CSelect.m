function Prediction = SVR_NFolds_RandomCV_CSelect(Subjects_Data, Subjects_Scores, FoldQuantity, CVRepeatTimes, Pre_Method, C_Range, ResultantFolder)
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
% CVRepeatTimes:
%           Because for N-fold cross validation, we need to randomly split
%           all subjects into N subsets. As it is random, we need to
%           repeat multiple times. Maybe 20, 50, 100, depending on the
%           computational resources. 
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

[Subjects_Quantity, ~] = size(Subjects_Data);

for i = 1:CVRepeatTimes

    % Split into N folds randomly
    EachPart_Quantity = fix(Subjects_Quantity / FoldQuantity);
    RandID = randperm(Subjects_Quantity);
    for j = 1:FoldQuantity
        Origin_ID{j} = RandID([(j - 1) * EachPart_Quantity + 1: j * EachPart_Quantity])';
    end
    Reamin = mod(Subjects_Quantity, FoldQuantity);
    for j = 1:Reamin
        Origin_ID{j} = [Origin_ID{j} ; RandID(FoldQuantity * EachPart_Quantity + j)];
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
    
        % Select optimal C
        for m = 1:length(C_Range)
            for k = 1:CVRepeatTimes
                Prediction_Inner = SVR_NFolds_RandomCV(Training_data, Training_scores, 5, Pre_Method, C_Range(m));
                Mean_Corr(k) = Prediction_Inner.Mean_Corr;
                Mean_MAE(k) = Prediction_Inner.Mean_MAE;
            end
            Inner_Corr_Array(m) = mean(Mean_Corr);
            Inner_MAE_Array(m) = mean(Mean_MAE);
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
        Training_scores = Training_scores;
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
        Prediction.Origin_ID{i, j} = Origin_ID{j};
        Prediction.Score{i, j} = Predicted_Scores;
        Prediction.Corr(i, j) = corr(Predicted_Scores, test_score);
        Prediction.MAE(i, j) = mean(abs(Predicted_Scores - test_score));
    end
    Prediction.Corr(find(isnan(Prediction.Corr))) = 0;
    Prediction.Mean_Corr(i) = mean(Prediction.Corr(i, :));
    Prediction.Mean_MAE(i) = mean(Prediction.MAE(i, :));
end
Prediction.CVMean_Corr = mean(Prediction.Mean_Corr);
Prediction.CVMean_MAE = mean(Prediction.Mean_MAE);
if nargin >= 6
    save([ResultantFolder filesep 'Prediction.mat'], 'Prediction');
    disp(['The correlation is ' num2str(Prediction.Mean_Corr)]);
    disp(['The MAE is ' num2str(Prediction.Mean_MAE)]);
    % Calculating w
    W_Calculate_SVR(Subjects_Data, Subjects_Scores, Pre_Method, C_Range, ResultantFolder); 
end

