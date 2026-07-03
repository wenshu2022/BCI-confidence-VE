clc;clearvars;
%define path 
rms_path = 'C:\Users\wenlou\Documents\MATLAB\rbms_Acerbi';
work_dir ="M:\1confiProj\Analysis";

addpath(rms_path)
cd(work_dir)

% read aic csv file from py
aic_file_name = 'all_models_AIC.csv';
df = readtable(fullfile(work_dir, aic_file_name));
df.gamma_var_ind = cell(height(df),1);
df.gamma_var_ind(df.gamma_var==1) = {'GammaNoise'};
df.gamma_var_ind(df.gamma_var==0) = {'NoGammaNoise'};
% get sub list from the table unique sub list
sub_lst = unique(df.sub_id);
n_sub = numel(sub_lst);
%unique(df.model_id)

%% test the bayesian against non bayesian
m_lst = 1:4;
n_model = numel(m_lst);
% sort based on model id
df_sel = sortrows(df(ismember(df.model_id, m_lst),:), 'model_id');
config_sel =  unique(df_sel(:, {'model_id' ,'CI_readout' ,'est_var' ,'Causal_readout' ,'gamma_var_ind'}));

AIC  = reshape(df_sel.aic, n_sub, n_model);
% Run a "fixed effects" (FFX) analysis based on a sum across subjects                                                    %Lower is better (because log-likelihoods were negated)             
bootstrapFFX(-.5*AIC);                                                      %Note multiplication by -.5 to convert information criteria back to log-model-evidence
% Run a "random effects" (RFX) analysis for AIC and BIC
bms_AIC = bms_Acerbi(-.5*AIC);  
bms_AIC.bor
bms_AIC.pxp

%% factor analysi - 2 factors
num_factors = 2; 
num_factor_components = [2, 2];
config_sel.combine_factor = strcat(config_sel.Causal_readout, '-', config_sel.gamma_var_ind);
f_names = cell(1,num_factors);
f_names{1} = ["Bay-GammaNoise", "Bay-NoGammaNoise"];
f_names{2} = ["MS", "MA"];

factors = cell(1,num_factors);
factors{1} =[strcmp(config_sel.combine_factor, f_names{1}{1})';strcmp(config_sel.combine_factor, f_names{1}{2})'];
factors{2} =[strcmp(config_sel.est_var, f_names{2}{1})';strcmp(config_sel.est_var, f_names{2}{2})'];

% Perform group-level Bayesian Model Factor Selection 
[~,bms_fac_AIC] = bms_Acerbi(-.5*AIC,factors);                              %Request the second output argument
bms_fac_AIC{2}
% save data so that it can be plotted in py
jsonData = jsonencode(bms_fac_AIC);
fid = fopen(fullfile(work_dir, 'bms_fac_AIC.json'), 'w');
fprintf(fid, '%s', jsonData);
fclose(fid);





%% factor analysi - 3 factors

num_factors = 3; 
num_factor_components = [2, 2, 2];
f_names = cell(1,num_factors);
f_names{1} = ["GammaNoise", "NoGammaNoise"];
f_names{2} = ["NonBayFixSig", "Bay"];
f_names{3} = ["MS", "MA"];
factor_names = ["Gamma noise", "Causal readout", "Spatial readout"];

factors = cell(1,num_factors);
factors{1} =[strcmp(config_sel.gamma_var_ind, f_names{1}{1})';strcmp(config_sel.gamma_var_ind, f_names{1}{2})'];
factors{2} =[strcmp(config_sel.Causal_readout, f_names{2}{1})';strcmp(config_sel.Causal_readout, f_names{2}{2})'];
factors{3} =[strcmp(config_sel.est_var, f_names{3}{1})';strcmp(config_sel.est_var, f_names{3}{2})'];

% Perform group-level Bayesian Model Factor Selection 
[~,bms_fac_AIC] = bms_Acerbi(-.5*AIC,factors);                              %Request the second output argument
bms_fac_AIC{3}
% save data so that it can be plotted in py
jsonData = jsonencode(bms_fac_AIC);
fid = fopen(fullfile(work_dir, 'bms_fac_AIC.json'), 'w');
fprintf(fid, '%s', jsonData);
fclose(fid);






%% first part compare the two heurustic (fix sig during inference and flex sig during inference)
m_lst = 5:12;
n_model = numel(m_lst);
% sort based on model id
df_sel = sortrows(df(ismember(df.model_id, m_lst),:), 'model_id');

df_sel.gamma_var_ind = cell(height(df_sel),1);
df_sel.gamma_var_ind(df_sel.gamma_var==1) = {'GammaNoise'};
df_sel.gamma_var_ind(df_sel.gamma_var==0) = {'NoGammaNoise'};

AIC  = reshape(df_sel.aic, n_sub, n_model);
% Run a "fixed effects" (FFX) analysis based on a sum across subjects                                                    %Lower is better (because log-likelihoods were negated)             
bootstrapFFX(-.5*AIC);                                                      %Note multiplication by -.5 to convert information criteria back to log-model-evidence
% Run a "random effects" (RFX) analysis for AIC and BIC
bms_AIC = bms_Acerbi(-.5*AIC);                                              %Use David Meijer's bms_Acerbi function 

bms_AIC.bor
bms_AIC.pxp
% bor = 0.32 - evidence not strong 

num_factors = 3; 
num_factor_components = [2, 2, 2];
f_names = cell(1,num_factors);
f_names{1} = ["GammaNoise", "NoGammaNoise"];
f_names{2} = ["NonBayFixSig", "NonBayFleSig"];
f_names{3} = ["MS", "MA"];
factor_names = ["Gamma noise", "sigma for inference", "Spatial readout"];


factors = cell(1,num_factors);
factors{1} =[strcmp(df_sel.gamma_var_ind, f_names{1}{1})';strcmp(df_sel.gamma_var_ind, f_names{1}{2})'];
factors{2} =[strcmp(df_sel.Causal_readout, f_names{2}{1})';strcmp(df_sel.Causal_readout, f_names{2}{2})'];
factors{3} =[strcmp(df_sel.est_var, f_names{3}{1})';strcmp(df_sel.est_var, f_names{3}{2})'];

% Perform group-level Bayesian Model Factor Selection 
[~,bms_fac_AIC] = bms_Acerbi(-.5*AIC,factors);                              %Request the second output argument
% bor = 0.2308. not too strong

m_lst = [70, 95]; 
% init a matrix, save all the AIC, BIC
n_model = numel(m_lst);
n_sub = numel(sub_lst);
res_mat = zeros(n_model * n_sub, 4);


for i = 1:n_model
    
    m_id = m_lst(i);
    res_mat((i-1)*n_sub+1:i*n_sub, 1) = zeros(n_sub, 1) + m_id;
    res_mat((i-1)*n_sub+1:i*n_sub, 2) = sub_lst';
    
    for j = 1:n_sub
        % read in the AIC and BIC
        sub_data = load(fullfile(est_path, ['m_' num2str(m_id) '_sub_' num2str(sub_lst(j)) '.mat' ]));
        [~, idx] = min(sub_data.minNLL);
        res_mat((i-1)*n_sub+j, 3) = sub_data.ev.AIC(idx);
        res_mat((i-1)*n_sub+j, 4) = sub_data.ev.BIC(idx);
    end
end


res_T = array2table(res_mat, 'VariableNames', {'m_id', 'sub_id', 'AIC', 'BIC'});

%writetable(res_T, 'AIC_BIC_new.csv')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% relative comparison
% transfor into n_sub * nmodel
AIC  = reshape(res_mat(ismember(res_T.m_id, m_lst), 3), n_sub, n_model);
BIC  = reshape(res_mat(ismember(res_T.m_id, m_lst), 4), n_sub, n_model);


% Run a "fixed effects" (FFX) analysis based on a sum across subjects
summed_AIC = sum(AIC);                                                       %Lower is better (because log-likelihoods were negated)             
summed_BIC = sum(BIC);                                                       
bootstrapFFX(-.5*AIC);                                                      %Note multiplication by -.5 to convert information criteria back to log-model-evidence
bootstrapFFX(-.5*BIC);                                                      %The bootstrap ensures that outlier participants are taken care of

    
% Run a "random effects" (RFX) analysis for AIC and BIC
bms_AIC = bms_Acerbi(-.5*AIC);                                              %Use David Meijer's bms_Acerbi function 
bms_BIC = bms_Acerbi(-.5*BIC);    

bms_BIC.bor
bms_BIC.pxp


% simple bar plot 
% Initialize figure
% figure;
% hold on;
% % Box Plot
% boxplot(BIC, 'Positions', m_lst, 'Widths', 0.5);
% for i = 1:size(BIC, 1)
%     plot(m_lst, BIC(i, :), '-o', 'LineWidth', 1, 'MarkerSize', 4, 'Color', [0.7 0.7 0.7]);
% end
% % Customize plot appearance
% grid on;
% xticks(m_lst); % Set x-ticks to match column numbers

% % Adjust the box plot to stay on top
% h = findobj(gca, 'tag', 'Box');
% for j = 1:length(h)
%     patch(get(h(j), 'XData'), get(h(j), 'YData'), [0 0.4470 0.7410], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
% end
% hold off;



% expected posterior model frequencies - EXP_R
% covariance matrix of posterior model - cov_r
% Bayesian Omnibus Risk - BOR
% protected exceedance probabilities - PXP








