import numpy as np

def NLLsimAllConds(par, stiPar, param_names, config, nIntSamples, behData, cond):
    # Update the full parameter set
    for i, param in enumerate(param_names):
        stiPar[param] = par[i]
    
    # If the sigmas in two phases are the same
    if config["same_var_2phase"]:
        stiPar["sig_A_A"] = stiPar["sig_A_AV"]
        stiPar["sig_V_V"] = stiPar["sig_V_AV"]
    
    # Simulate all conditions
    predData = simAllConds(config, stiPar, nIntSamples, cond)[0]  # Only first return value
    
    # Part II: Compute Negative Log-Likelihood (NLL)
    
    # Levels of decisions
    ncat = 2  # Level of causal decisions
    nr = 3    # Level of confidence
    ns = 4    # Level of spatial locations
    
    NLL = 0  # Initialize NLL
    
    for cidx in range(cond.shape[0]):
        # Extract condition values
        c_ = cond[cidx, :]
        
        cond_mask_pred = np.all(predData[:, :4] == c_, axis=1)
        cond_mask_beh = np.all(behData[:, :4] == c_, axis=1)
        
        pred_resp = predData[cond_mask_pred, 4:7]
        data_cond = behData[cond_mask_beh, 4:7]
        
        # Map spatial locations into indices 1,2,3,4
        pred_spatial_mapped = np.searchsorted(stiPar["resp_loc"], pred_resp[:, 2])
        pred_indices = np.ravel_multi_index((pred_resp[:, 0]-1, pred_resp[:, 1]-1, pred_spatial_mapped-1), (ncat, nr, ns))
        
        # Compute predicted probabilities
        pMat_counts = np.bincount(pred_indices, minlength=ncat * nr * ns)
        temp_probs = pMat_counts.reshape(ncat, nr, ns) / nIntSamples
        
        # Apply lapse rate correction
        pMat = (1 - stiPar["laps"]) * temp_probs + stiPar["laps"] / (ncat * nr * ns)
        
        # Ensure non-zero probabilities and normalize
        pMat += np.finfo(float).eps  # Add small value to avoid log(0)
        pMat /= pMat.sum()
        
        # Compute real data probability counts
        data_spatial_mapped = np.searchsorted(stiPar["resp_loc"], data_cond[:, 2])
        data_indices = np.ravel_multi_index((data_cond[:, 0]-1, data_cond[:, 1]-1, data_spatial_mapped-1), (ncat, nr, ns))
        cntMat = np.bincount(data_indices, minlength=ncat * nr * ns).reshape(ncat, nr, ns)
        
        # Compute NLL
        NLL += -np.sum(cntMat * np.log(pMat))
    
    return NLL