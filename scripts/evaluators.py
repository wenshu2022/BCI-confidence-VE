import pandas as pd
import numpy as np
import os 
# import pickle
from scripts.modelClassGPU import ConfiModel
from scripts.experiments import Experiment, DataHandler
from scripts.run_optimizers import gen_init_params
from typing import List, Optional
from tqdm import tqdm
import time
# from scripts.computeQLL import func_compute_NLL, func_compute_NLL_detail
# from scripts.computeMNLE import compute_NLL, compute_NLL_detail

class ModelEvaluator:
    def __init__(self, exp_name:str, cluster=False, cross_test=False):   
        self.exp = Experiment(exp_name, cluster=cluster, cross_test=cross_test)
        self.dh = DataHandler(cluster=cluster)
        self.cross_test = cross_test

    def get_model(self, m_id:int):
        model = ConfiModel(m_id, self.exp.exp_config)
        return model
    
    def test_model_exp(self, m_id:int, x0=None):
        ''' for testing whether the models (at least) runs error free  '''
        model = self.get_model(m_id)
        bayes = model.model_config['Causal_readout']=='Bay'
        if x0 is None:
            x0 = gen_init_params(model.p_bounds, model.estParamsNames, bayes) 
        model.display_parameters()

        trialConds = model.getCondRep(nIntSamples=500)
        start_time = time.time()
        dat_sim = model.run_simulation(x0,trialConds)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.6f} seconds") 
        return dat_sim, x0
    
    def remap_params(self, model, param_values:list):
        # the order of parameters in estParamsNames may be different from the order in which they are stored in the fitted results. Here we remap them.
        param_names_fit = self.dh.get_fit_param_names(m_id=model.m_id, exp_name=self.exp.exp_name)
        param_names_model = model.estParamsNames
        param_values = list(param_values)
        if 'a' in param_names_model and 'b' in param_names_model and 'a' not in param_names_fit and 'b' not in param_names_fit:
            # fix legacy problems
            param_values.extend([1, 0])
            param_names_fit.extend(['a', 'b'])

        if param_names_fit == param_names_model:
            return param_values
        else:
            # Create a mapping from fitted parameter names to model parameter names
            param_mapping = {name: i for i, name in enumerate(param_names_fit)}
            # Reorder the param_values according to the model's parameter order
            remapped_params = [param_values[param_mapping[name]] for name in param_names_model]
            return remapped_params

    def get_best_param_nll(self, m_id:int, sub_id:int, by:str = 'test'):
        df = self.dh.get_sub_fit_data(m_id=m_id, exp_name=self.exp.exp_name, sub_id=sub_id)
        #sAMPLE 30
        #df = df.tail(n=10)
        if 'model_id' in df.columns and 'm_id' not in df.columns: # for some legacy reason
            df.rename(columns={'model_id': 'm_id'}, inplace=True)
       
        #find the best params for the model and subject from all the runs. 
        # df_best_params = df[['sub_id', 'm_id', 'est_params', by + '_nll']].groupby(['sub_id', 'm_id']).apply(
        #     lambda x: x.loc[x[by + '_nll'].idxmin()]).reset_index(drop=True)
        best_idx = df.groupby(['sub_id', 'm_id'])[by + '_nll'].idxmin()
        df_best_params = df.loc[best_idx, ['sub_id', 'm_id', 'est_params', by + '_nll']].reset_index(drop=True)

        return df_best_params.loc[(df_best_params['m_id'] == m_id) & (df_best_params['sub_id'] == sub_id), 'est_params'].values[0], \
                df_best_params.loc[(df_best_params['m_id'] == m_id) & (df_best_params['sub_id'] == sub_id), by + '_nll'].values[0]
    
    def get_loss(self, m_id : int, sub_id:int, save:bool = False):
        nll_method = self.exp.exp_config['NLL_method']
        model = self.get_model(m_id)
        # If cross_test=True, swap QML <-> MNLE
        if self.cross_test:
            nll_method = 'MNLE' if nll_method == 'QML' else 'QML'
        beh_data, _ = self.exp.load_data_for_fitting(sub_id, i_fold=1, nll_method=nll_method, full_data=True)
        func_nll = self.exp.get_objf(beh_data, model, nll_method)
            
        df = self.dh.get_sub_fit_data(m_id=m_id, exp_name=self.exp.exp_name, sub_id=sub_id)
        df['test_nll'] = df.apply(lambda row: func_nll(row['est_params']), axis=1)
        print('finish')
        if save:
            df.to_pickle(self.dh.get_sub_fit_filename(m_id=m_id, exp_name=self.exp.exp_name, sub_id=sub_id))
        return df

    def get_loss_detail_MNLE(self, m_id : int, sub_id:int, by:str='train'):
        pass

    def get_allsub_pred_data(
        self, 
        m_id: int, 
        sub_lst: Optional[List[int]] = None, 
        save_csv: bool = False
    ) -> pd.DataFrame:

        if sub_lst is None:
            sub_lst = self.dh.sub_lst_all

        if not sub_lst:
            raise ValueError("sub_lst is empty.")

        df_list = []

        # Progress bar here
        for sub_id in tqdm(sub_lst, desc=f"Model {m_id} subjects", unit="subject"):
            df_sub = self.get_model_prediction(
                m_id=m_id,
                sub_id=sub_id,
                remove_outliers=True,
                nsim=1000
            )
            df_sub = df_sub.copy()
            df_sub["m_id"] = m_id
            df_list.append(df_sub)

        df_sim = pd.concat(df_list, ignore_index=True)

        if save_csv:
            save_dir = os.path.join(self.dh.fit_res_path, "agg_data")
            os.makedirs(save_dir, exist_ok=True)
            df_sim.to_csv(os.path.join(save_dir, "model_pred_aggregated.csv"), index=False)

        return df_sim
    # def get_allsub_pred_data(self, m_id:int, sub_lst=None,save_csv:bool=False):
    #     df_sim = []
    #     if sub_lst is None:
    #         sub_lst = self.dh.sub_lst_all
    #     # aggreagte all the subjects
    #     for sub_id in sub_lst:
    #         df_sim_t = self.get_model_prediction(m_id = m_id, sub_id = sub_id, remove_outliers = True, nsim=1000)
    #         df_sim_t['m_id'] = m_id
    #         df_sim.append(df_sim_t) 

    #     df_sim = pd.concat(df_sim, ignore_index = True)
    #     if save_csv:
    #         df_sim.to_csv(os.path.join(self.dh.fit_res_path, 'agg_data', 'model_pred_aggregated.csv'), index = False)
    #     return df_sim
    
    def get_loss_detail_QML(self, m_id:int, sub_id:int, by:str='train'):
        model = self.get_model(m_id)
        beh_data, _ = self.exp.load_data_for_fitting(sub_id, i_fold=1, nll_method = self.exp.exp_config['NLL_method'], full_data=True)
        func_nll = self.exp.func_loss_detail(model, beh_data, self.exp.exp_config['num_sim'])
        best_param,_ = self.get_best_param_nll(m_id, sub_id, by)
        # Get the results
        NLL_lst, beh_bins, log_probs_lst, lp_loc_lst, NLL_loc_lst, lp_ci_lst, NLL_ci_lst = func_nll(best_param)
        # Create DataFrame
        df_loss_detail = pd.DataFrame()
        df_loss_detail[['AudLoc', 'VisLoc', 'VisRel', 'blockType']] = model.conditions.tolist()
        df_loss_detail['NLL_joint'] = NLL_lst  # This works - list of scalars
        df_loss_detail['NLL_loc'] = NLL_loc_lst  
        df_loss_detail['NLL_ci'] = NLL_ci_lst  

        # For complex data structures, store as objects
        df_loss_detail['beh_bins'] = [beh_bins[i] for i in range(len(beh_bins))]
        df_loss_detail['lp_joint'] = [log_probs_lst[i] for i in range(len(log_probs_lst))]
        df_loss_detail['lp_loc'] = [lp_loc_lst[i] for i in range(len(lp_loc_lst))]
        df_loss_detail['lp_ci'] = [lp_ci_lst[i] for i in range(len(lp_ci_lst))]

        return df_loss_detail
    
    def get_model_prediction(self, m_id, sub_id, nsim:int = 1000, remove_outliers = False, by:str='train', print_param:bool = False, interimOutput:bool=False):
        """
        Load the fitted model results for a specific model and subject.

        Parameters:
        - model: Model object
        - sub_id: Subject identifier
        - fit_res_path: Path to the directory containing fitted results

        Returns:
        - DataFrame containing the fitted prediction for the specified model and subject.
        """
        model = self.get_model(m_id)

        par_input,_ = self.get_best_param_nll(m_id = m_id, sub_id = sub_id, by = by)
        par_input = self.remap_params(model, par_input) # remap the order of parameters if necessary. This is to prevent imcompatibility caused by code restruct. 
        trialConds = model.getCondRep(nIntSamples=nsim)
        if print_param:
            model.display_parameters() 
        if interimOutput:
            dat_sim, sim_post, sim_post_conf, xA, xV, post_cdf, post_pdf, bin_est, trial_indices, binSize, nbin,ntrial = model.run_simulation(inputs=par_input,trialConds=trialConds,interimOutput=interimOutput)
            df_sim = self.dh.organize_dat(dat = dat_sim, remove_outliers = remove_outliers, sub_id=sub_id, part_id=model.exp_config['split_data_f'])
            return df_sim, sim_post, sim_post_conf, xA, xV, post_cdf, post_pdf, bin_est, trial_indices, binSize, nbin,ntrial
        
        else:
            dat_sim = model.run_simulation(inputs=par_input,trialConds=trialConds,interimOutput=interimOutput)
            df_sim = self.dh.organize_dat(dat = dat_sim, remove_outliers = remove_outliers, sub_id=sub_id, part_id=model.exp_config['split_data_f'])

        # # remove outliers
        # if remove_outliers:
        #     df_sim['CISize_zscore'] = (df_sim['CISize'] - df_sim['CISize'].mean()) / df_sim['CISize'].std()
        #     df_sim = df_sim[df_sim.CISize_zscore.abs() < 3]  
            return df_sim

    def getdf_AICcBIC(self, m_id:int, sub_id:int, by:str='train'):
        model = self.get_model(m_id)
        k = len(model.estParamsNames)
        best_par, nll = self.get_best_param_nll(m_id, sub_id, by=by)
        aic = 2*k + 2 * nll
        beh_dat = self.dh.get_beh_data(sub_id, MNLE=False)
        n = beh_dat.shape[0] # sample size
        aicc = aic + (2* (k**2) + 2 * k)/(n - k - 1)
        bic = k * np.log(n) + 2* nll

        # put to a df
        df_ev = pd.DataFrame([{
                "m_id": m_id,
                "exp": self.exp.exp_name, # or subsititute with attribute of this expriment
                "sub_id": sub_id,
                "AIC": aic,
                "AICc": aicc,
                "BIC": bic,
                "nll": nll,
                "npar": k,
                "best_par": best_par,
                "nll_method" : self.exp.exp_config['NLL_method'],
                'CI_readout': model.model_config['CI_readout'],
                'est_var': model.model_config['est_var'],
                'Causal_readout': model.model_config['Causal_readout'],
                }])
        return df_ev




class CompareModels:
    def __init__(self, exp1_name:str, exp2_name=None):   
        self.ev_exp1 = ModelEvaluator(exp1_name)
        self.ev_exp2 = ModelEvaluator(exp2_name) if exp2_name is not None else None
        self.sub_lst = self.ev_exp1.dh.sub_lst_all
    
    def get_evMlist(self, m1_list:list, m2_list:list=None):
        if self.ev_exp2 is not None and m2_list is not None:
            # pair each experiment with its corresponding model list
            combos = [(self.ev_exp1, m1_list), (self.ev_exp2, m2_list)]
            pairs = [(ev, m_id) for ev, models in combos for m_id in models]

        elif self.ev_exp2 is None and m2_list is None:
            pairs = [(self.ev_exp1, m_id) for m_id in m1_list]

        else:
            raise ValueError("Both m2_list and ev_exp2 should be provided together, or neither.")

        return pairs
        
    def compare_models(self, m1_list:list, best_by:str, m2_list:list=None, sub_lst:list=None, by_nll:str='test', save_csv=False):
        '''
        compare by model metrics
        best_by : str : 'AIC', 'BIC', 'AICc'
        '''
        if sub_lst is None:
            sub_lst = self.sub_lst
        pairs = self.get_evMlist(m1_list, m2_list)
        # compute BIC/AIC
        df_ev= []
        for ev, m_id in pairs:
            for sub_id in sub_lst:
                # get the test nll - TO DO
                df_metrics = ev.getdf_AICcBIC(m_id, sub_id, by_nll)
                df_ev.append(df_metrics)
        df_ev = pd.concat(df_ev, ignore_index=True)

        best_idx = df_ev.groupby(['sub_id'])[best_by].idxmin()
        df_best_model = df_ev.loc[best_idx, :].reset_index(drop=True)

        if save_csv:
            df_ev.to_csv('M:\\1confiProj\\Analysis\\output\\model_compare.csv', index = False)
        return df_ev, df_best_model
