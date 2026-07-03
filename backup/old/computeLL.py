import numpy as np
import pandas as pd

def get_condition_data(data, condition):
    """Helper function to extract rows from data matching a condition."""
    cond_mask = np.all(data[:, :4] == condition, axis=1)
    return data[cond_mask, 4:]

def get_simulate_dist(dat_sim, dat_beh, nbin_loc=20, nbin_ci=10, nbin_conf=10):
    '''
    generate a 4D simulated distribution

    Parameters
    ----------
    dat_sim : array, nsamples * 4
        simulated data for one condition.
    dat_beh : array, ntrials * 4
        behaviour data for one condition.
    nbin_loc : int, optional
        number of bins for localization data. The default is 20.
    nbin_ci : int, optional
        number of bins for confidence interval data. The default is 10.
    nbin_conf : int, optional
        number of bins for causal confidence data. The default is 10.

    Returns
    -------
    prob_mass : 4D array
        save the prob mass in each 4D bin from the simulation.
    loc_grid : 1D array shape (nbin_loc+1, )
        grid/bin edges for simulated distribution.
    ci_grid : 1D array shape (nbin_ci+1, )
        grid/bin edges for simulated distribution.
    conf_grid : 1D array shape (nbin_conf+1, )
        grid/bin edges for simulated distribution.

    '''
    loc_min, loc_max = min(dat_sim[:,0].min(), dat_beh[:,0].min()), max(dat_sim[:,0].max(), dat_beh[:,0].max())
    ci_min, ci_max = min(dat_sim[:,1].min(), dat_beh[:,1].min()), max(dat_sim[:,1].max(), dat_beh[:,1].max())

    # Add small margins, so that it won't be out of bound during np.digititaze()
    loc_grid = np.linspace(loc_min - 0.1, loc_max + 0.1, nbin_loc + 1)
    ci_grid = np.linspace(ci_min - 0.1, ci_max + 0.1, nbin_ci + 1)
    conf_grid = np.linspace(-0.1, 100.1, nbin_conf+1)
    
    # first get the bin indice for each 4D entry
    loc_binned = np.digitize(dat_sim[:, 0], bins=loc_grid) - 1  # subtract 1 to get 0-based bin indices
    ci_binned  = np.digitize(dat_sim[:, 1], bins=ci_grid) - 1  
    conf_binned  = np.digitize(dat_sim[:, 3], bins=conf_grid) - 1  
    
    bin_indices = np.column_stack((loc_binned, ci_binned, dat_sim[:, 2].astype(int)-1, conf_binned))
    #compute the occurence of the sim_dat in each bin
    bin_sizes = (nbin_loc, nbin_ci, 2, nbin_conf)
    # Map 4D indices to 1D index
    multi_indices = np.ravel_multi_index(bin_indices.T, bin_sizes)
    # Count occurrences
    counts_in_bins = np.bincount(multi_indices, minlength=np.prod(bin_sizes)).reshape(bin_sizes)
    prob_mass = counts_in_bins / counts_in_bins.sum()
    prob_mass += np.finfo(float).eps  # Add small value to avoid log(0)
    prob_mass /= prob_mass.sum()
    
    return prob_mass, loc_grid, ci_grid, conf_grid

def comput_ll(dat, prob_mass, loc_grid, ci_grid, conf_grid):
    '''
    The function computes the log likelihood in each condition given the simulated population

    Parameters
    ----------
    dat : array, nsamples * 4
        behaviour data for one condition.
    prob_mass : 4D array
        save the prob mass in each 4D bin from the simulation.
    loc_grid : 1D array shape (nbin_loc+1, )
        grid/bin edges for simulated distribution.
    ci_grid : 1D array shape (nbin_ci+1, )
        grid/bin edges for simulated distribution.
    conf_grid : 1D array shape (nbin_conf+1, )
        grid/bin edges for simulated distribution.

    Returns
    -------
    float:    sum(logp)

    '''
    # get the bin indices from the simulated distribution
    loc_binned = np.digitize(dat[:, 0], bins=loc_grid) - 1  
    ci_binned  = np.digitize(dat[:, 1], bins=ci_grid) - 1  
    conf_binned  = np.digitize(dat[:, 3], bins=conf_grid) - 1  
    
    bin_indice = np.column_stack([loc_binned, ci_binned, dat[:,2].astype(int)-1, conf_binned])
    log_probs = np.log(prob_mass[tuple(bin_indice.T)])  
    return log_probs.sum()


def func_compute_NLL(model, beh_data, num_sim=10000, nbin_loc=20, nbin_ci=10, nbin_conf=10):
    '''
    generate the objective function for optimize with maxium likelihood estimation

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    beh_data : TYPE
        DESCRIPTION.
    num_sim : TYPE, optional
        DESCRIPTION. The default is 10000.
    nbin_loc : TYPE, optional
        DESCRIPTION. The default is 20.
    nbin_ci : TYPE, optional
        DESCRIPTION. The default is 10.
    nbin_conf : TYPE, optional
        DESCRIPTION. The default is 10.

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
            pred_resp = get_condition_data(simulated_data, c_)
            beh_resp = get_condition_data(beh_data, c_)
        
            # Compute LL
            prob_mass, loc_grid, ci_grid, conf_grid =  \
                get_simulate_dist(pred_resp, beh_resp, nbin_loc, nbin_ci , nbin_conf)
            LL += comput_ll(beh_resp, prob_mass, loc_grid, ci_grid, conf_grid)
        # return negative log-likelihood
        return -LL
            
    return neg_log_likelihood


def gen_init_params(bounds, estParamsNames):
    ''' generate x0 for optimization'''
    sig_Vrh_idx = estParamsNames.index('sig_Vrh')
    sig_Vrl_idx = estParamsNames.index('sig_Vrl')
    sig_A_idx = estParamsNames.index('sig_A')
    sig_P_idx = estParamsNames.index('sig_P')
    
    x0s = np.array([np.random.uniform(low, high) for (low, high) in bounds])
    # number 1 : the visual high realibility variance smaller visual low realibility 
    x0s[sig_Vrl_idx] = np.random.uniform(max(x0s[sig_Vrh_idx], bounds[sig_Vrl_idx][0]), bounds[sig_Vrl_idx][1])
    # number 2 : spatial prior should have the biggest variance of all
    x0s[sig_P_idx] = np.random.uniform(max(x0s[sig_Vrl_idx],x0s[sig_Vrh_idx],x0s[sig_A_idx], bounds[sig_P_idx][0]), bounds[sig_P_idx][1])

    return x0s

    
def define_constraints(estParamsNames):
     '''
     # constraints
     # number 1 : the visual high realibility variance smaller visual low realibility 
     # number 2 : spatial prior should have the biggest variance of all

     Parameters
     ----------
     estParamsNames : list of str
         from model class.

     Returns
     -------
     cons 

     '''
     sig_Vrh_idx = estParamsNames.index('sig_Vrh')
     sig_Vrl_idx = estParamsNames.index('sig_Vrl')
     sig_A_idx = estParamsNames.index('sig_A')
     sig_P_idx = estParamsNames.index('sig_P')
     
     cons = [
         {'type': 'ineq', 'fun': lambda x: x[sig_Vrl_idx] - x[sig_Vrh_idx]},
         {'type': 'ineq', 'fun': lambda x: x[sig_P_idx] - x[sig_Vrh_idx]},
         {'type': 'ineq', 'fun': lambda x: x[sig_P_idx] - x[sig_Vrl_idx]},
         {'type': 'ineq', 'fun': lambda x: x[sig_P_idx] - x[sig_A_idx]},
     ]
     return cons
   
    
    
    