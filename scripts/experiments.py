# the goal of this script is to prepare data for one experimental modeling attempts. 
import pandas as pd
import numpy as np
import os 
import pickle
from scipy import stats
# from scripts.modelClassGPU import ConfiModel
# from scripts.run_optimizers import gen_init_params
# import time
# from scripts.computeQLL import func_compute_NLL, func_compute_NLL_detail
from scripts.computeMNLE import compute_NLL, compute_NLL_detail
class Experiment:
    def __init__(self, exp_name:str, cluster=True, cross_test=False):
        self.exp_name = exp_name
        self.data_handler = DataHandler(cluster=cluster)
        self.exp_config = self.read_config()
        self.func_loss, self.func_loss_detail= self.define_loss_function(cross_test)
    
    def define_loss_function(self, cross_test):
        if self.exp_config['split_data_f'] ==0:
            from scripts.computeQLL import func_compute_NLL, func_compute_NLL_detail
        elif self.exp_config['split_data_f'] ==2:
            from scripts.computeQLL_split2 import func_compute_NLL, func_compute_NLL_detail

        nll_method = self.exp_config['NLL_method']
        # Base mapping (no cross-test)
        base_mapping = {
            'QML': (func_compute_NLL, func_compute_NLL_detail),
            'MNLE': (compute_NLL, compute_NLL_detail),
        }
        if nll_method not in base_mapping:
            raise ValueError(f"Unsupported NLL method: {nll_method}")
        # If cross_test=True, swap QML <-> MNLE
        if cross_test:
            nll_method = 'MNLE' if nll_method == 'QML' else 'QML'
        return base_mapping[nll_method]
   
    def get_objf(self, data, model, NLL_method):
        '''get a function that takes params as compute NLL as output'''
        if NLL_method == 'QML':
            objf = self.func_loss(model, data, self.exp_config['num_sim'])
        elif NLL_method == 'MNLE':
            estimator = self.get_MNLE_estimator(model.m_id)
            def objf(x):
                return self.func_loss(x, estimator, data)
        return objf
    
    def get_MNLE_estimator(self, m_id):
        from scripts.utils import CPU_Unpickler
        with open(os.path.join(self.data_handler.MNLE_est_path, self.exp_name , 'estimators', 'model_' + str(m_id) + '_estimator.pickle'), "rb") as f:
            estimator = CPU_Unpickler(f).load()
        return estimator
    
    def read_config(self):
        exp_config = pd.read_csv('exp_config.csv')
        return exp_config[exp_config.exp_name==self.exp_name].iloc[0].to_dict()
    
    def make_exp_dir(self, m_id:int, sub_lst:list):
        #make dir to store the partial results
        for sub_id in sub_lst:
            os.makedirs(os.path.join(self.data_handler.fit_res_path, self.exp_name, 'partial', 'model_' + str(m_id), 'subject_' + str(sub_id)), exist_ok=True)

    def load_data_for_fitting(self, sub_id:int, i_fold:int, nll_method:str, full_data = False, vrel=0):
         if self.exp_config['split_data_f'] ==1:
            exp_data_path = os.path.join(self.data_handler.data_path, 'split_data_vrel', nll_method + str(self.exp_config['nfold_CV']))
            train_data = pickle.load(open(os.path.join(exp_data_path, 'beh_bin_sub_' + str(sub_id) + '_visrel' + str(vrel) +'.pkl'), 'rb'))
            return train_data, train_data
         elif self.exp_config['split_data_f'] ==0:
            if nll_method == 'MNLE':
                if self.exp_config['nfold_CV']==1 or full_data:
                    # no cross validation, train and test is the same
                    train_data = self.data_handler.get_beh_data(sub_id = sub_id, MNLE = True)
                    return train_data, train_data
                else:
                    pass

            elif nll_method == 'QML':
                exp_data_path = os.path.join(self.data_handler.data_path, nll_method + str(self.exp_config['nfold_CV']))
                if self.exp_config['nfold_CV']==1 or full_data:
                    # no cross validation, train and test is the same
                    train_data = pickle.load(open(os.path.join(exp_data_path, 'beh_bin_sub_' + str(sub_id) +'.pkl'), 'rb'))
                    return train_data, train_data
                else:
                    pass
         elif self.exp_config['split_data_f'] ==2:
            exp_data_path = os.path.join(self.data_handler.data_path, 'split_data2', nll_method + str(self.exp_config['nfold_CV']) +'_cols01')
            train_data = pickle.load(open(os.path.join(exp_data_path, 'beh_bin_sub_' + str(sub_id) +'.pkl'), 'rb'))
            return train_data, train_data
            #return train_data, test_data
            # if no cross-validation (n_fold=1), the train and test are the same

    def safe_qcut(self, data, max_bins):
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

    def get_condition_data(self, data, condition, cols=None):
        if cols is None:
            cols = np.array([0, 1, 2, 3])

        cond_mask = np.all(data[:, :4] == condition, axis=1)
        return data[cond_mask][:, cols + 4]

    def prepare_bin_edges(self, cols=None, nbin_loc:int = 10, nbin_ci:int = 8, nbin_conf:int=6):
        '''
        cut the data into bins for each subject and each condition, and save the bin edges and bin indices for each trial. 
        
        cols: list of column indices to be used for binning
        nbin_loc: int, number of bins for location
        nbin_ci: int, number of bins for ci
        nbin_conf: int, number of bins for confidence
        '''
        if cols is None:
            cols = np.array([0, 1, 2, 3]) 
            fit_data_path = os.path.join(self.data_handler.data_path, self.exp_config['NLL_method'] + str(self.exp_config['nfold_CV']))
        else:
            fit_data_path = os.path.join(self.data_handler.data_path, 'split_data' + str(self.exp_config['split_data_f']), self.exp_config['NLL_method'] + str(self.exp_config['nfold_CV']) + '_cols' + ''.join(map(str, cols)))
        # make dir
        os.makedirs(fit_data_path, exist_ok=True)
        conds = self.data_handler.get_model_condition()
        
        for sub_id in self.data_handler.sub_lst_all:
            #sub_id = 2
            sub_data = self.data_handler.get_beh_data(sub_id=sub_id, MNLE=False)

            sub_bins = {}
            for cix in np.arange(conds.shape[0]):
                data = self.get_condition_data(data = sub_data, condition = conds[cix], cols=cols)
            
                #cut into euqal size bins
                if np.array_equal(cols, np.array([0, 1, 2, 3])):
                    # the full dataset
                    loc_binned, loc_grid = self.safe_qcut(data[:, 0], nbin_loc)
                    ci_binned, ci_grid = self.safe_qcut(data[:, 1], nbin_ci)
                    conf_binned, conf_grid = self.safe_qcut(data[:, 3], nbin_conf)
                    bin_indices = np.column_stack((loc_binned, ci_binned, data[:, 2].astype(int)-1, conf_binned))
                    # save data
                    sub_bins[cix] = {
                            'cond': conds[cix],
                            'loc_grid': loc_grid,
                            'ci_grid': ci_grid,
                            'conf_grid': conf_grid,
                            'bin_indices': bin_indices
                        }
                    
                elif np.array_equal(cols, np.array([0, 1])):
                    # perceptual data only
                    loc_binned, loc_grid = self.safe_qcut(data[:, 0], nbin_loc)
                    ci_binned, ci_grid = self.safe_qcut(data[:, 1], nbin_ci)
            
                    bin_indices = np.column_stack((loc_binned, ci_binned))
                    # save data
                    sub_bins[cix] = {
                            'cond': conds[cix],
                            'loc_grid': loc_grid,
                            'ci_grid': ci_grid,
                            'bin_indices': bin_indices
                        }

            pickle.dump(sub_bins, open(os.path.join(fit_data_path, 'beh_bin_sub_' + str(sub_id) + '.pkl') , 'wb'))


           

class DataHandler:
    def __init__(self,  cluster: bool = True):
        self.cluster = cluster
        self.sub_lst_all = [2, 3, 5, 6 ,11 ,15 ,20, 25]
        self.data_path, self.fit_res_path, self.MNLE_est_path = self.get_path()

    def get_path(self):
        if self.cluster:
            data_path = '/project/3026008.02/data_for_fitting/'
            fit_res_path = '/project/3026008.02/fit_res/'
            MNLE_est_path = '/project/3026008.02/MNLE/'
        else: 
            data_path = 'P:/3026008.02/data_for_fitting/'
            fit_res_path = 'P:/3026008.02/fit_res/'
            MNLE_est_path = 'P:/3026008.02/MNLE/'
        return data_path, fit_res_path, MNLE_est_path

    def get_model_condition(self):
        with open(os.path.join(self.fit_res_path, 'model_condition.pkl'), 'rb') as f:
            conds = pickle.load(f)
        return conds
    
    def get_beh_data(self, sub_id, MNLE=False):
        sub_data = np.loadtxt(os.path.join(self.data_path, 'beh_data_sub_' + str(sub_id) + '.csv'), delimiter=',')
        if MNLE:
            # reorg data to be consistent with the training input
            sub_data = sub_data[:, [0, 1, 2, 3, 4, 5, 7, 6]] # the catrgorical condition is the last column
            sub_data[:, -1] = 2 - sub_data[:, -1] # convert the 1 - 2 choice to 0 - 1 choice
            sub_data[:,-2] = sub_data[:,-2]/100.0
        return sub_data
    
    def get_sub_fit_data(self, m_id:int, exp_name:str, sub_id:int):
        sub_file_name = self.get_sub_fit_filename(m_id=m_id, exp_name = exp_name , sub_id=sub_id)
        if not os.path.exists(sub_file_name):
            self.organize_parial_data(m_id=m_id, exp_name=exp_name, sub_id=sub_id)
            #raise FileNotFoundError(f"Result file {sub_file_name} does not exist.")
        df = pd.read_pickle(sub_file_name)
        return df

    def get_fit_param_names(self, m_id:int, exp_name:str):
        param_file_name = os.path.join(self.fit_res_path, exp_name, 'partial', 'model_' + str(m_id), 'est_param_names.txt')
        with open(param_file_name, 'r') as f:
            param_names = f.read().strip().split('\n')
        return param_names


    def get_sub_fit_filename(self, m_id:int, exp_name:str, sub_id:int):
        sub_file_name = os.path.join(self.fit_res_path, exp_name, 'full', 'model_' + str(m_id), 'fitted_model_' + str(m_id) +  '_subject_' + str(sub_id) + '.pkl')
        return sub_file_name
    
    def organize_parial_data(self, m_id:int, exp_name:str, sub_id:int):
        # this function organize the partial data from fitting results into a full file for each subject
        # make a folder to store the aggregated data
        full_file_path = os.path.join(self.fit_res_path, exp_name, 'full', 'model_' + str(m_id))
        os.makedirs(full_file_path, exist_ok=True)
        part_path = os.path.join(self.fit_res_path, exp_name, 'partial', 'model_' + str(m_id))

        print(f"\n ------- Start Aggregating for Subject {sub_id} ---------- ")
        sub_path = os.path.join(part_path, 'subject_' + str(sub_id))
        all_files = [f.path for f in os.scandir(sub_path) if f.is_file() and f.path.endswith('.pkl')]

        if os.path.isdir(sub_path) and all_files:
            df_full = []
            for f in all_files:
                df_part = pd.read_pickle(f)
                df_full.append(df_part)
            df_full = pd.concat(df_full, ignore_index=True)
            df_full.to_pickle(os.path.join(full_file_path, 'fitted_model_' + str(m_id) +  '_subject_' + str(sub_id) + '.pkl'))
            
        else:
            raise RuntimeError(f"Folder '{sub_path}' does not exist or has no files")
        # for sub_id in self.sub_lst_all:
        #     print(f"\n ------- Start Aggregating for Subject {sub_id} ---------- ")
        #     sub_path = os.path.join(part_path, 'subject_' + str(sub_id))
        #     all_files = [f.path for f in os.scandir(sub_path) if f.is_file() and f.path.endswith('.pkl')]

        #     if os.path.isdir(sub_path) and all_files:
        #         df_full = []
        #         for f in all_files:
        #             df_part = pd.read_pickle(f)
        #             df_full.append(df_part)
        #         df_full = pd.concat(df_full, ignore_index=True)
        #         df_full.to_pickle(os.path.join(full_file_path, 'fitted_model_' + str(m_id) +  '_subject_' + str(sub_id) + '.pkl'))
               
        #     else:
        #         raise RuntimeError(f"Folder '{sub_path}' does not exist or has no files")
            
    def organize_dat(self, dat, remove_outliers:bool=False, sub_id=999, part_id:int=0):
        # organize the simulated data (or beh data) into a dataframe
        if part_id ==2:
            # if only produce perceptual prediction, add two additional zeros columns. 
            dat = np.column_stack((dat, np.zeros((dat.shape[0], 2))))
        df = pd.DataFrame(dat)

        df.columns =['AudLoc', 'VisLoc', 'VisRel', 'blockType', 'RespLoc', 'CISize', 'Sources', 'ConfLevel']

        df['sub_id'] = sub_id
        df['delta_VA'] = df['VisLoc'] - df['AudLoc']
        df['VisRelLabel'] = df['VisRel'].map({1:'High', 2:'Low'})
        df['Modality'] = df['blockType'].map({0:'V', 1:'A'})

        # get the perceptual bias for bi-modal congruent trials
        df_stat = df.loc[df['AudLoc'] == df['VisLoc']].groupby(['sub_id', 'blockType', 'VisRel', 'AudLoc', 'VisLoc'])['RespLoc'].agg(['mean', 'std']).reset_index()
        df_aud = df.loc[df.blockType==1].merge(df_stat.loc[df_stat.blockType==1, ['sub_id', 'blockType', 'VisRel', 'AudLoc', 'mean']], how='left', on=['sub_id', 'blockType', 'VisRel', 'AudLoc'])
        df_vis = df.loc[df.blockType==0].merge(df_stat.loc[df_stat.blockType==0, ['sub_id', 'blockType', 'VisRel', 'VisLoc', 'mean']], how='left', on=['sub_id', 'blockType', 'VisRel', 'VisLoc'])
        df = pd.concat([df_aud, df_vis], ignore_index=True)
        df['bias'] = df['RespLoc'] - df['mean']

        #flip bias
        df['flipBias'] = df['bias'].copy()
        df.loc[df['delta_VA']<0, 'flipBias'] = -df.loc[df['delta_VA']<0, 'flipBias']
        df['abs_delta_VA'] = df.delta_VA.abs()

        # other bias measurement - from Tim's paper
        df.loc[df.Modality=='A', 'bias_abs'] = df.loc[df.Modality=='A', 'RespLoc'] - df.loc[df.Modality=='A', 'AudLoc']
        df.loc[df.Modality=='V', 'bias_abs'] = df.loc[df.Modality=='V', 'RespLoc'] - df.loc[df.Modality=='V', 'VisLoc']
        df['bias_abs_rel'] = df['bias_abs'] / df['delta_VA']
        df['bias_rel'] =  df['bias'] / df['delta_VA']

        # df['flipBias_abs'] = df['bias_abs'].copy()
        # df.loc[df['delta_VA']<0, 'flipBias_abs'] = -df.loc[df['delta_VA']<0, 'flipBias_abs']

        group_cols = ['sub_id', 'Modality', 'VisRel']
        #group_cols = ['sub_id', 'Modality', 'VisRel', 'abs_delta_VA']
        df = df.copy()

        df['bias_abs_zscore'] = (
            df.groupby(group_cols)['bias_abs']
            .transform( lambda x: stats.zscore(x, ddof=1))
        )

        df['bias_zscore'] = (
            df.groupby(group_cols)['bias']
            .transform( lambda x: stats.zscore(x, ddof=1))
        )

        df['CISize_zscore'] = (
            df.groupby(group_cols)['CISize']
            .transform( lambda x: stats.zscore(x, ddof=1))
        )
        
        if remove_outliers:
            df = df[
                (df['bias_abs_zscore'].abs() <= 3) &
                (df['CISize_zscore'].abs() <= 3)
            ]
        
        # def remove_outliers_func(df,bias_col='bias'):
        #     df = df.copy()
        #     df.loc[:,'bias_zscore'] = (df[bias_col] - df[bias_col].mean()) / df[bias_col].std()
        #     df.loc[:,'CISize_zscore'] = (df['CISize'] - df['CISize'].mean()) / df['CISize'].std()
        #     return df[(df.bias_zscore.abs() <= 3) & (df.CISize_zscore.abs() <= 3)]
       
        # #if remove_outliers_f:
        # df = (df.groupby(['sub_id', 'Modality', 'VisRel', 'VisLoc', 'AudLoc'], group_keys=False)
        #         .apply(remove_outliers_func)
        #         .reset_index(drop=True))
            
        return df


    def get_allsub_beh_data(self, save_csv:bool=False):
        df_beh_filename = os.path.join(self.fit_res_path, 'agg_data', 'beh_data_aggregated.csv')
        if os.path.exists(df_beh_filename):
            df_beh = pd.read_csv(df_beh_filename)
            return df_beh   
        else:
            df_beh = []
            # aggreagte all the subjects
            for sub_id in self.sub_lst_all:
                # load beh data differently??
                df_beh_t = self.organize_dat(dat = self.get_beh_data(sub_id, MNLE=False),sub_id=sub_id)
                df_beh.append(df_beh_t)

            df_beh = pd.concat(df_beh, ignore_index = True)
            if save_csv:
                df_beh.to_csv(df_beh_filename, index = False)
            return df_beh
 