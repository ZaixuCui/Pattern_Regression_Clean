
function RVR_NFolds_Sort_Permutation(Subjects_Data, Subjects_Scores, Times, FoldQuantity, Pre_Method, ResultantFolder)
%
% Doing permutation test to create the distribution of random model.
% Because permutation is pretty slow, here we parallized the jobs using
% PSOM (http://psom.simexp-lab.org/).
%
% Subjects_Data:
%           m*n matrix
%           m is the number of subjects
%           n is the number of features
%
% Subjects_Scores:
%         m*1 vector, the scores of all subjects
%
% Times:
%         Repetition times for the permutation test
% 
% FoldQuantity:
%         Quantity of the cross validation, 5 or 10 is recommended
%
% Pre_Method:
%           'Normalize', 'Scale', 'None'
%
% ResultantFolder:
%         Path of the folder for stroing the results.
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

for i = 1:Times
   
    ResultantFolder_I = [ResultantFolder filesep 'Time_' num2str(i)];
    mkdir(ResultantFolder_I);
    
    RVR_NFolds_Sort(Subjects_Data, Subjects_Scores, FoldQuantity, Pre_Method, 0, 1, ResultantFolder)
end



