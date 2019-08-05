function Prediction = RVR_NFolds_RandomCV(Subjects_Data, Subjects_Scores, FoldQuantity, CVRepeatTimes, Pre_Method, Permutation_Flag, ResultantFolder)
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
%           The quantity of folds, 10 is recommended
%
% CVRepeatTimes:
%           Repeatition times for the random n-folds cross-validation
%
% Pre_Method:
%           'Normalize', 'Scale', 'None'
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
        Training_data_final = double(Training_data);
    
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
        
        % RVR training & predicting
        d.train{1} = Training_data_final * Training_data_final';
        d.test{1} = test_data_final * Training_data_final';
        d.tr_targets = Training_scores;
        d.use_kernel = 1;
        d.pred_type = 'regression';
        output = prt_machine_rvr(d, []);
        
        Prediction.Origin_ID{i, j} = Origin_ID{j};
        Prediction.Score{i, j} = output.predictions;
        Prediction.Corr(i, j) = corr(output.predictions, test_score);
        Prediction.MAE(i, j) = mean(abs(output.predictions - test_score));  
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
end
