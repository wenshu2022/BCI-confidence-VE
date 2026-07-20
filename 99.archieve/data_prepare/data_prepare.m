clc;clear;
data_path = 'P:\3026008.02\rawData';
out_path = 'P:\3026008.02\Data';
sub_lst = [2, 3, 5, 6 ,11 ,15 ,20, 25];
%i=1;
%init
allSubData = table();
for i = 1:length(sub_lst)
    %col names automatically fixed during readtable
    %auditory 
    sub_dataA = readtable(fullfile(data_path, ['Subj_'  num2str(sub_lst(i))  '_BimodalAuditoryData_Excel.xlsx']));
    %select only bimodal trials - all with auditory response
    sub_dataA = rmmissing(sub_dataA);
    sub_dataA{:, 'blockType'} = 1; % 1 for auditory, 0 - for visual.
    
    %visual 
    sub_dataV = readtable(fullfile(data_path, ['Subj_'  num2str(sub_lst(i))  '_BimodalVisualData_Excel.xlsx']));
    sub_dataV = rmmissing(sub_dataV);
    sub_dataV{:, 'blockType'} = 0; % 1 for auditory, 0 - for visual.
    
    sub_data = [sub_dataA;sub_dataV];
    %normalize the confidence use David function
    sub_data.ConfLevelNorm = round(normalizeNonParametric(sub_data.ConfLevel) * 100);
    sub_data.ConfClass = discretize(sub_data.ConfLevelNorm, 4); % map into 4 discrete values
    sub_data{:, 'sub_id'}= sub_lst(i);
    
    allSubData = [allSubData;sub_data];
end


writetable(allSubData, fullfile(out_path, ['bimodalAllSub.csv']))


