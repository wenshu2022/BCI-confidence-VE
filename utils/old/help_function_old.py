import pandas as pd
from datetime import datetime
import os

fontsize = 16
sub_lst = [2, 3, 5, 6 ,11 ,15 ,20, 25]

all_data_path = "P:/3026008.02/AllData/"
plot_path = "M:/1confiProj/plots/"
fit_data_path = "P:/3026008.02/data_for_fitting/"
fit_res_path = "M:\\1confiProj\\fitting_results\\output\\"

def clean_axs(ax,fontsize=16):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', direction='in', left=True, bottom=True, labelsize=fontsize)

def ax_noXlabel(ax):
    ax.set_xlabel('')
    ax.xaxis.set_ticklabels([])

def ax_noYlabel(ax):
    ax.set_ylabel('')
    ax.yaxis.set_ticklabels([])

def organize_sim_data(dat, sub_id=999):
    
    df = pd.DataFrame(dat)
    df.columns =['AudLoc', 'VisLoc', 'VisRel', 'blockType', 'RespLoc', 'CISize', 'Sources', 'ConfLevel']

    df['sub_id'] = sub_id
    df['delta_VA'] = df['VisLoc'] - df['AudLoc']
    df.loc[ df.VisRel==1, 'VisRelLabel'] = 'High'
    df.loc[ df.VisRel==2, 'VisRelLabel'] = 'Low'
    
    df.loc[df.blockType==1, 'bias'] = df.loc[df.blockType==1, 'RespLoc'] - df.loc[df.blockType==1, 'AudLoc']
    df.loc[df.blockType==0, 'bias'] = df.loc[df.blockType==0, 'RespLoc'] - df.loc[df.blockType==0, 'VisLoc']

    #flip the bias
    df['flipBias'] = df['bias'].copy()
    df.loc[df['delta_VA']<0, 'flipBias'] = -df.loc[df['delta_VA']<0, 'flipBias']
    df['abs_delta_VA'] = df.delta_VA.abs()
    
    return df

def save_model_fit_result(
    model_id: int,
    sub_id: int,
    est_params: list,
    init_params: list,
    nll: float,
    runtime: float,
    result_file: str,
    success: bool = True
):
    """
    Save a single model fitting result to a file (appending if file exists).

    Parameters:
    - model_id: identifier of the model
    - est_params: List of estimated parameter values
    - init_params: List of initial values used for optimization
    - nll: Negative log-likelihood (or other loss) value
    - result_file: Path to file for storing results
    """

    run_data = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_id": model_id,
        "sub_id": sub_id,
        "nll": nll,
        "est_params": est_params,
        "init_params": init_params,
        "runtime": runtime,
        "success": success 
    }

    # Load existing file or create new DataFrame
    if os.path.exists(result_file):
        df = pd.read_pickle(result_file)
    else:
        df = pd.DataFrame()

    # Append new run
    df = pd.concat([df, pd.DataFrame([run_data])], ignore_index=True)

    # Save back
    df.to_pickle(result_file)

def get_model_prediction(model, sub_id, remove_outliers = True, fit_res_path=fit_res_path):
    """
    Load the fitted model results for a specific model and subject.

    Parameters:
    - model: Model object
    - sub_id: Subject identifier
    - fit_res_path: Path to the directory containing fitted results

    Returns:
    - DataFrame containing the fitted prediction for the specified model and subject.
    """
    m_id = 1
    #m_id = model.mid
    res_file = os.path.join(fit_res_path, f"model_{m_id}", f"fitted_sub_{sub_id}_m{m_id}_bads.pkl")
    
    if not os.path.exists(res_file):
        raise FileNotFoundError(f"Result file {res_file} does not exist.")
    
    df = pd.read_pickle(res_file)
    #find the best params for the model and subject from all the runs. 
    df_best_params = df[['sub_id', 'model_id', 'est_params','nll']].groupby(['sub_id', 'model_id']).apply(
        lambda x: x.loc[x['nll'].idxmin()]).reset_index(drop=True)

    par_input = df_best_params.loc[(df_best_params['model_id'] == m_id) & (df_best_params['sub_id'] == sub_id), 'est_params'].values[0]
    trialConds = model.getCondRep(nIntSamples=500)

    dat_sim = model.run_simulation(par_input,trialConds)
    df_sim = organize_sim_data(dat_sim, sub_id=sub_id)
    # remove outliers
    if remove_outliers:
        df_sim['CISize_zscore'] = (df_sim['CISize'] - df_sim['CISize'].mean()) / df_sim['CISize'].std()
        df_sim = df_sim[df_sim.CISize_zscore.abs() < 3]  
    return df_sim