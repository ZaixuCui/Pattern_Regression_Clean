
function [w_Brain, model_All] = W_Calculate_RVR(Subjects_Data, Subjects_Scores, Pre_Method, ResultantFolder)
%
% Subject_Data:
%           m*n matrix
%           m is the number of subjects
%           n is the number of features
%
% Subject_Scores:
%           the continuous variable to be predicted,[1*m]
%
% Pre_Method:
%          'Normalize', 'Scale', 'None'
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

[~, Features_Quantity] = size(Subjects_Data);

if strcmp(Pre_Method, 'Normalize')
    %Normalizing
    MeanValue = mean(Subjects_Data);
    StandardDeviation = std(Subjects_Data);
    [~, columns_quantity] = size(Subjects_Data);
    for j = 1:columns_quantity
        Subjects_Data(:, j) = (Subjects_Data(:, j) - MeanValue(j)) / StandardDeviation(j);
    end
elseif strcmp(Pre_Method, 'Scale')
    % Scaling to [0 1]
    MinValue = min(Subjects_Data);
    MaxValue = max(Subjects_Data);
    [~, columns_quantity] = size(Subjects_Data);
    for j = 1:columns_quantity
        Subjects_Data(:, j) = (Subjects_Data(:, j) - MinValue(j)) / (MaxValue(j) - MinValue(j));
    end
end
    
% SVR
Subjects_Data = double(Subjects_Data);
d.train{1} = Subjects_Data * Subjects_Data';
d.test{1} = Subjects_Data * Subjects_Data';
d.tr_targets = Subjects_Scores;
d.use_kernel = 1;
d.pred_type = 'regression';
model_All = prt_machine_rvr(d, []);

w_Brain = zeros(1, Features_Quantity);
for i = 1:length(model_All.alpha)
    w_Brain = w_Brain + model_All.alpha(i) * Subjects_Data(i, :);
end

w_Brain = w_Brain / norm(w_Brain);

if nargin >= 4
    save([ResultantFolder filesep 'w_Brain.mat'], 'w_Brain');
end
