#this script is to separate the behaviour data into bins and each bin and roughly the same size. 
import os
wkdir = 'M:/1confiProj/'
os.chdir(wkdir)
import sys
parent_dir = os.path.abspath(os.getcwd())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scripts.modelClassGPU import ConfiModel
from scripts.experiments import Experiment
import pandas as pd
import numpy as np
import os
import pickle

sub_lst = [2, 3, 5, 6 ,11 ,15 ,20, 25]

#get conditions from the model class
experiment = Experiment('base_clamp')
model = ConfiModel(m_id = 1, exp_config = experiment.exp_config)
conds = model.conditions

def get_condition_data(data, condition):
    """Helper function to extract rows from data matching a condition."""
    cond_mask = np.all(data[:, :4] == condition, axis=1)
    return data[cond_mask, 4:]

def safe_qcut(data, max_bins):
    '''
    Helper function to cut the data into quantile-based bins.
    Handles cases where data has only one unique value.
    
    Returns:
        bin_ind: Index of the bin each value belongs to
        bin_edges: The edges of the bins
    '''
    data = np.asarray(data)
    
    # Handle edge case where data has 0 or 1 unique value
    if len(np.unique(data)) <= 1:
        # One bin only, centered on the unique value
        unique_val = data[0] if len(data) > 0 else 0
        bin_edges = np.array([unique_val - 0.1, unique_val + 0.1])
        bin_ind = np.zeros_like(data, dtype=int)
        return bin_ind, bin_edges

    # Use qcut normally
    _, bin_edges = pd.qcut(data, q=max_bins, labels=False, retbins=True, duplicates='drop')

    # Adjust the first and last edges to slightly expand range
    bin_edges[0] -= 0.1
    bin_edges[-1] += 0.1

    bin_ind = np.digitize(data, bins=bin_edges, right=True) - 1
    return bin_ind, bin_edges

# def safe_qcut(data, max_bins):
#     '''
#     helper function to cut the for beh data in each condition. 
#     Return equal size bin edges and bin indices (same way as the sim data would be cut)
#     '''
#     _, bin_edges = pd.qcut(data, q=max_bins, labels=False, retbins = True, duplicates='drop')

#     # add a bit more space at the upper and lower bounds
#     bin_edges[0] = bin_edges[0] - 0.1
#     bin_edges[-1] = bin_edges[-1] + 0.1

#     bin_ind = np.digitize(data, bins=bin_edges, right=True) - 1
#     return bin_ind, bin_edges

# max number for the bins
nbin_loc= 8 #10 #10 #6
nbin_ci= 6 #8 #4 #6
nbin_conf= 6 #4 #8 #6

data_path = 'P:/3026008.02/data_for_fitting/'
fit_data_path = os.path.join(data_path, 'split_data', experiment.exp_config['NLL_method'] + str(experiment.exp_config['nfold_CV']))
if not os.path.exists(fit_data_path):
    os.makedirs(fit_data_path)
# load behave data
# split for high and low visual reliability
for sub_id in sub_lst:
    #sub_id = 2
    sub_data = np.loadtxt(os.path.join(data_path, 'beh_data_sub_' + str(sub_id) + '.csv'), delimiter=',')
    for vrel in [1, 2]:
        sub_data_vrel = sub_data[sub_data[:, 2] == vrel]
        sub_bins = {}
        new_conds = conds[conds[:,2]==vrel] # for this visual reliability only

        for cix in np.arange(new_conds.shape[0]):
            data = get_condition_data(sub_data_vrel, new_conds[cix])
        
            #cut into euqal size bins
            loc_binned, loc_grid = safe_qcut(data[:, 0], nbin_loc)
            ci_binned, ci_grid = safe_qcut(data[:, 1], nbin_ci)
            conf_binned, conf_grid = safe_qcut(data[:, 3], nbin_ci)
        
            bin_indices = np.column_stack((loc_binned, ci_binned, data[:, 2].astype(int)-1, conf_binned))
            # save data
            sub_bins[cix] = {
                    'cond': new_conds[cix],
                    'loc_grid': loc_grid,
                    'ci_grid': ci_grid,
                    'conf_grid': conf_grid,
                    'bin_indices': bin_indices
                }
            
        pickle.dump(sub_bins, open(os.path.join(fit_data_path, 'beh_bin_sub_' + str(sub_id) + '_visrel' + str(vrel) + '.pkl') , 'wb'))
        
    