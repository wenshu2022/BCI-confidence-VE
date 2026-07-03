# general imports
import numpy as np
import torch

CTE = 1/10000
def compute_NLL(theta, estimator, data, eps=1e-3):
    '''
    inputs:
    theta: np.array() parameters at which to evaluate the NLL
    data: beh data (including trial conditions)
    '''
    ntrials = data.shape[0] # number of trials
    cond = data[:, :4]
    theta = torch.reshape(torch.tensor(theta), (1, len(theta))).to(torch.float32)
    theta = theta.repeat(ntrials, 1)
    theta = torch.column_stack((theta, torch.tensor(cond).to(torch.float32)))
    #reorg data
    x_o = data[:, 4:]
    x_o = torch.reshape(torch.tensor(x_o), (1, ntrials, 4)).to(torch.float32)
    log_liks = estimator.log_prob(x_o, condition=theta).detach().numpy()  #take log prob from MNLE
    log_liks = np.exp(log_liks)*(1-eps) + eps*CTE  # add contaminants
    log_liks = np.log(log_liks)  # take log prob

    return -np.nansum(log_liks)

def compute_NLL_detail(theta, estimator, data, eps=1e-3):
    '''
    inputs:
    theta: np.array() parameters at which to evaluate the NLL
    data: beh data (including trial conditions)
    '''
    ntrials = data.shape[0] # number of trials
    cond = data[:, :4]
    theta = torch.reshape(torch.tensor(theta), (1, len(theta))).to(torch.float32)
    theta = theta.repeat(ntrials, 1)
    theta = torch.column_stack((theta, torch.tensor(cond).to(torch.float32)))
    #reorg data
    x_o = data[:, 4:]
    x_o = torch.reshape(torch.tensor(x_o), (1, ntrials, 4)).to(torch.float32)
    log_liks = estimator.log_prob(x_o, condition=theta).detach().numpy()  #take log prob from MNLE
    log_liks = np.exp(log_liks)*(1-eps) + eps*CTE  # add contaminants
    log_liks = np.log(log_liks)  # take log prob

    # --- Aggregate per condition ---
    cond_tuples = [tuple(c) for c in cond]
    unique_conds = sorted(set(cond_tuples))

    nll_summary = []
    for c in unique_conds:
        mask = np.array([ct == c for ct in cond_tuples])
        avg_nll = -np.mean(log_liks[mask])  # mean NLL for that condition
        nll_summary.append(list(c) + [avg_nll])

    nll_summary = np.array(nll_summary)  # shape: (n_conditions, cond_dim+1)

    return log_liks, cond, nll_summary


