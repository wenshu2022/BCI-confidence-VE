clc;clear;
work_dir ='M:\1confiProj\codeCompare';
cd(work_dir)
addpath('M:\MATLAB\Model')

rng(42)
t = randn(12800, 1);
writematrix(t, 'rand_number.csv')

param_names = {'sig_V_AV', ...
 'sig_A_AV', ...
 'sig_P',...
 'p_com',...
 'conf_bin_k1',...
 'conf_bin_k2',...
 'alp_shift_A',...
 'b_A',...
 'alp_shift_V'};

m_id =70;
par = [2, 12, 100, 0.75, 0.55, 0.65, 0.55, 10, 0.15]
x = [3, 12, 50, 0.35, 0.55, 0.65, 0.5, 10, 0.15]

[predData, sim_post, updatedShift_A, bi_xA, bi_xV, bi_sA_hat, bi_sV_hat, cond] = run_sim(m_id,par, param_names, 100);
cond_rep = reshape(repmat(cond', 100, 1), 4, [])';
writematrix(predData, 'mat_pred.csv')
writematrix(cond_rep, 'trailcond.csv')
writematrix(sim_post, 'sim_post.csv')
writematrix(bi_xA, 'bi_xA.csv')
writematrix(bi_xV, 'bi_xV.csv')

[config, stiPar] = fconfig(m_id);
[predData2, ~] = run_sim(m_id,x, param_names, 10000);
writematrix(predData2, 'mat_pred2.csv')
nll=NLLsimAllConds(x, stiPar, param_names, config, 10000, predData, cond);
nll

function [config, stiPar] = fconfig(m_id)
    [config, stiPar, ~, ~] = model_config('M:\MATLAB\Model', 0, m_id, 0);
    stiPar.a_A                  =        1; 
    stiPar.mu_P                 =        0;
    %stiPar.alp_shift_V          =        0;
    stiPar.laps                 =        0.01;
end

function [predData, sim_post, updatedShift_A, bi_xA, bi_xV, bi_sA_hat, bi_sV_hat, cond] = run_sim(m_id,par, param_names, nsim)
    
    [config, stiPar] = fconfig(m_id);
    
    %generate simulations
    for i = 1:length(param_names)
        stiPar.(param_names{i}) = par(i); 
    end
    %if the sigmas in two phases are the same
    if config.same_var_2phase 
        stiPar.sig_A_A = stiPar.sig_A_AV;
        stiPar.sig_V_V = stiPar.sig_V_AV;
    end 
    cond_A = [combvec(stiPar.resp_loc, stiPar.resp_loc, stiPar.resp_loc)' ones(stiPar.resp_loc_len *stiPar.resp_loc_len *stiPar.resp_loc_len , 1)]; % [A in AV, V in AV, A/V in uni, A(1) or V(0) block]
    cond_V = [combvec(stiPar.resp_loc, stiPar.resp_loc, stiPar.resp_loc)' zeros(stiPar.resp_loc_len *stiPar.resp_loc_len *stiPar.resp_loc_len , 1)]; % [A in AV, V in AV, A/V in uni, A(1) or V(0) block]
    cond = [cond_A; cond_V];
    [predData, sim_post, updatedShift_A, bi_xA, bi_xV, bi_sA_hat, bi_sV_hat] = simAllConds(config, stiPar, nsim,cond);      
end
