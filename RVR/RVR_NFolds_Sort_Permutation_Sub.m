
function RVR_NFolds_Sort_Permutation_Sub(Subjects_Data_Path, Subjects_Scores, FoldQuantity, Pre_Method, ResultantFolder)
%
% Sub-function for the RVR_NFolds_Sort_Permutation.m
%
tmp = load(Subjects_Data_Path);
FieldName = fieldnames(tmp);
Subjects_Data = tmp.(FieldName{1});
RVR_NFolds_Sort(Subjects_Data, Subjects_Scores, FoldQuantity, Pre_Method, 0, 1, ResultantFolder)
