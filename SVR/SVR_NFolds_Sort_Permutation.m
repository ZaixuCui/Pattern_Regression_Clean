
function SVR_NFolds_Sort_Permutation(Subjects_Data_Path, Subjects_Scores, Times, FoldQuantity, C_Range, ResultantFolder)
%
% Doing permutation test to create the distribution of random model.
% Because permutation is pretty slow, here we parallized the jobs using
% PSOM (http://psom.simexp-lab.org/).
%
% Subjects_Data_Path:
%         The path of the .mat file that storing the m*n matrix variable 
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
% C_Range:
%           The range of parameter C, 
%           We used (2^-5, 2^-4, ..., 2^9, 2^10) in our previous paper, see
%           Cui and Gong, 2018, NeuroImage, also see Hsu et al., 2003, A
%           practical guide to support vector classification. 
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
%                       Cui and Gong et al., 2018, NeuroImage; 
%                       Cui et al., 2016, Human Brain Mapping.
% (google scholar: https://scholar.google.com.hk/citations?user=j7amdXoAAAAJ&hl=zh-TW&oi=ao)
%

for i = 1:Times
   
    ResultantFolder_I = [ResultantFolder filesep 'Time_' num2str(i)];
    mkdir(ResultantFolder_I);
    
    Job_Name = ['perm_' num2str(i)];
    pipeline.(Job_Name).command = 'SVR_NFolds_Sort_Permutation_Sub(opt.para1, opt.para2, '''', opt.para3, ''Scale'', opt.para4, opt.para5)';
    pipeline.(Job_Name).opt.para1 = Subjects_Data_Path;
    pipeline.(Job_Name).opt.para2 = Subjects_Scores;
    pipeline.(Job_Name).opt.para3 = FoldQuantity;
    pipeline.(Job_Name).opt.para4 = C_Range;
    pipeline.(Job_Name).opt.para5 = ResultantFolder_I;

end

psom_gb_vars;
Pipeline_opt.mode = 'batch'; % 'batch' means parallized within a single computer; If parallizing on a SGE cluster, using 'qsub'
Pipeline_opt.qsub_options = '-q all.q'; % if 'mode' is 'qsub', here specifying the queue name
Pipeline_opt.mode_pipeline_manager = 'batch';
Pipeline_opt.max_queued = 2;
Pipeline_opt.flag_verbose = 1;
Pipeline_opt.flag_pause = 0;
Pipeline_opt.flag_update = 1;
Pipeline_opt.path_logs = [ResultantFolder filesep 'logs'];

psom_run_pipeline(pipeline,Pipeline_opt);


