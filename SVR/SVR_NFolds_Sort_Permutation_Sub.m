
function SVR_NFolds_Sort_Permutation_Sub(Subjects_Data_Path, Subjects_Scores, FoldQuantity, Pre_Method, C_Range, ResultantFolder)
%
% Sub-function for the SVR_NFolds_Sort_Permutation.m
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

tmp = load(Subjects_Data_Path);
FieldName = fieldnames(tmp);
Subjects_Data = tmp.(FieldName{1});
SVR_NFolds_Sort_CSelect(Subjects_Data, Subjects_Scores, FoldQuantity, Pre_Method, C_Range, 0, 1, ResultantFolder)
