import numpy as np
import cProfile
import pstats
import time 
from functools import partial
def get_cond_n_bin_data(data, bin_data, condition, cix):
    """
    help function to organize simulated data and behaviour bin data

    Parameters
    ----------
    data : np.array nsims * 8
        direct output from model simulation.
    bin_data : dict
        bin edges and bin indecis from behaviour data.
    condition : 1D np.array 
        one condition [A , V, V reliability 1 - high, 2 - low, A(1) or V(0) block].
    cix : int
        idx for the conditions.

    Returns
    -------
    1. filtered simulation data for the given condition
    2, 3, 4, 5. unpack the beh bin dict and return the grid and bin indices from beh data

    """
    assert np.array_equal(bin_data[cix]['cond'], condition), "Condition index are inconsistent!"
    cond_mask = np.all(data[:, :4] == condition, axis=1)
    
    return data[cond_mask, 4:], bin_data[cix]['loc_grid'], bin_data[cix]['ci_grid'], bin_data[cix]['bin_indices']


def comput_LL(pred_resp, loc_grid, ci_grid, beh_bin_indices, num_sim, eps, detail= False):
    '''
    the function computes log liklihood of the beh data for one condition

    Parameters
    ----------
    pred_resp : simulated/predicted response.
    loc_grid : bin edges for localization response from behaviour data.
    ci_grid : bin edges for confidence interval from behaviour data.
    beh_bin_indices : np.array ntrial * 4
        the bin indices for beh data
    num_sim : int
    detail : bool, optional
        If True, return detailed log likelihood information. The default is False.
    Returns
    -------
    LL
    '''
    # first get the bin indice for each 4D entry
    loc_binned = np.digitize(pred_resp[:, 0], bins=loc_grid, right=True) - 1  # subtract 1 to get 0-based bin indices
    ci_binned  = np.digitize(pred_resp[:, 1], bins=ci_grid, right=True) - 1  

    bin_indices = np.column_stack((loc_binned, ci_binned))
    #compute the occurence of the sim_dat in each bin
    bin_sizes = (len(loc_grid)-1, len(ci_grid)-1)

    # some instances are out of bound and removed
    valid_mask = (
        (bin_indices[:, 0] >= 0) & (bin_indices[:, 0] < bin_sizes[0]) &
        (bin_indices[:, 1] >= 0) & (bin_indices[:, 1] < bin_sizes[1]) 
    )

    # Filter only valid indices
    valid_bin_indices = bin_indices[valid_mask]

    # Map 4D indices to 1D index
    multi_indices = np.ravel_multi_index(valid_bin_indices.T, bin_sizes)
    # Count occurrences
    counts_in_bins = np.bincount(multi_indices, minlength=np.prod(bin_sizes)).reshape(bin_sizes)
    prob_mass = counts_in_bins / num_sim
    # add eps and also avoid 0 prob
    prob_mass = (1-eps)*prob_mass + eps/(np.prod(bin_sizes)) 
    #log_probs = np.log(prob_mass[tuple(beh_bin_indices.T)])  

    # -------------------------
    # joint log likelihood
    # -------------------------
    joint_probs = prob_mass[tuple(beh_bin_indices.T)]
    logp_joint = np.log(joint_probs)

    if detail:
        # -------------------------
        # marginal p(loc)
        # -------------------------

        prob_loc = prob_mass.sum(axis=1) # sum over ci dimension to get marginal p(loc)

        loc_idx = beh_bin_indices[:,0]
        logp_loc = np.log(prob_loc[loc_idx])

        # -------------------------
        # conditional p(ci|loc)
        # -------------------------

        logp_ci_given_loc = logp_joint - logp_loc

        ll_joint = logp_joint
        ll_loc = logp_loc
        ll_ci_given_loc = logp_ci_given_loc

        return ll_joint, ll_joint.mean(), ll_loc, ll_ci_given_loc

    else:

        return logp_joint.sum()



def func_compute_NLL(model, bin_data, num_sim=10000, eps=1e-2):
    '''
    generate the objective function for optimize with maxium likelihood estimation

    Parameters
    ----------
    model : object
    bin_data : dict
        bin edges and indices from behaviour data.
    num_sim : int, optional
        DESCRIPTION. The default is 10000.

    Returns
    -------
    a function to compute NLL.

    '''
    def neg_log_likelihood(params):
        simulated_data = model.run_simulation(params, model.getCondRep(nIntSamples=num_sim))
        LL = 0
        for cidx in range(model.conditions.shape[0]):
            # get data for this condition
            c_ = model.conditions[cidx, :]
            pred_resp, loc_grid, ci_grid, beh_bin_indices= get_cond_n_bin_data(simulated_data, bin_data, c_, cidx)
        
            # Compute LL
            LL += comput_LL(pred_resp, loc_grid, ci_grid, beh_bin_indices, num_sim, eps)

        # return negative log-likelihood
        return -LL
    
    return neg_log_likelihood


def func_compute_NLL_detail(model, bin_data, num_sim=10000, eps=1e-2):
    '''
    generate the objective function for optimize with maxium likelihood estimation

    Parameters
    ----------
    model : object
    bin_data : dict
        bin edges and indices from behaviour data.
    num_sim : int, optional
        DESCRIPTION. The default is 10000.

    Returns
    -------
    a function to compute NLL.

    '''
    def neg_log_likelihood_lst(params):
        simulated_data = model.run_simulation(params, model.getCondRep(nIntSamples=num_sim))
        NLL_lst = []
        beh_bins = []
        log_probs_lst = []
        lp_loc_lst = []
        NLL_loc_lst = []
        NLL_ci_lst= []
        lp_ci_lst = []
        for cidx in range(model.conditions.shape[0]):
            # get data for this condition
            c_ = model.conditions[cidx, :]
            pred_resp, loc_grid, ci_grid, beh_bin_indices= get_cond_n_bin_data(simulated_data, bin_data, c_, cidx)
        
            log_probs,  LL, lp_loc, lp_ci = comput_LL(pred_resp, loc_grid, ci_grid, beh_bin_indices, num_sim, eps, True)
            beh_bins.append(beh_bin_indices)
            # Compute LL
            NLL_lst.append(-1 * LL)
            NLL_loc_lst.append(-1 * lp_loc.mean())
            NLL_ci_lst.append(-1 * lp_ci.mean())
            log_probs_lst.append(log_probs)
            lp_loc_lst.append(lp_loc)
            lp_ci_lst.append(lp_ci)
            
        # return negative log-likelihood
        return NLL_lst, beh_bins, log_probs_lst, lp_loc_lst, NLL_loc_lst, lp_ci_lst, NLL_ci_lst
            
    return neg_log_likelihood_lst    
