import numpy as np
import itertools
from dataclasses import dataclass, field
import pandas as pd
from scipy.stats import norm
#import time

@dataclass
class ModelParams:
    """Stores numerical model parameters with defaults."""
    sig_Vrh: float = 1
    sig_Vrl: float = 20
    sig_A: float = 8
    sig_P: float = 30
    sig_r: float = 0.01
    mu_P: float = 0
    p_com: float = 0.5
    rng: float = 0.4
    logbeta: float = 0
    logbeta_conf: float = 0
    epsilon: float = 1
    a:float = 5
    b:float = 2
    kC1:float = 5
    mC1:float = 1
    kC2:float = 5
    mC2:float = 1
    stim_loc: np.ndarray = field(default_factory=lambda: np.array([-10, -5, 0, 5, 10]))
    stim_loc_len: int = 5
    laps: float = 0.01

class ConfiModel:
    def __init__(self, mid:int, params: ModelParams = ModelParams()):
        self.params = params
        self.mid = mid
        self.conditions = self.generate_conditions()
        self.config = self.load_config()
        self.estParamsNames, self.bounds, self.p_bounds = self.est_params()

    def display_parameters(self):
        for param, value in self.params.__dict__.items():
            print(f"{param}: {value}")
        print("\nParams for estimation are")
        for item in self.estParamsNames:
            print(item)

    def generate_conditions(self):
        """Generate all possible conditions based on stim_loc.
        each row represent a trial condition = [A , V, V reliability 1 - high, 2 - low, A(1) or V(0) block]
        """
        combs = np.array(list(itertools.product(self.params.stim_loc, repeat=2)))
        num_combs = combs.shape[0]
        # Create condition arrays
        cond_A = np.column_stack((np.tile(combs, (2, 1)), np.repeat([1, 2], num_combs), np.ones(num_combs*2)))  # both visual reliability for A(1) block
        cond_V = np.column_stack((np.tile(combs, (2, 1)), np.repeat([1, 2], num_combs), np.zeros(num_combs*2))) # both visual reliability for V(0) block
        # Concatenate both conditions
        return np.vstack((cond_A, cond_V)).astype(int)
    
    def load_config(self):
        """
        Load model configuration from an Excel file and filter by model_id.
        """
        config_df = pd.read_csv('m_config.csv')  # Read full config file
        model_config = config_df[config_df["mid"] == self.mid]  # Filter by model_id

        if model_config.empty:
            raise ValueError(f"Model ID {self.mid} not found in configuration file.")

        return model_config.iloc[0].to_dict()  # Convert to dictionary
    
    def update_est_params(self, x):
        # Check if the length of x and self.estParamsNames is the same
        if len(x) != len(self.estParamsNames):
            raise ValueError("Length of x does not match the number of estimated parameters.")
        
        for i, paramName in enumerate(self.estParamsNames):
            setattr(self.params, paramName, x[i])

    def est_params(self):
        params = ['sig_Vrh',  'sig_Vrl',   'sig_A',    'sig_P', 'p_com',        ' sig_r',           'a',           'b',       'kC1',           'mC1']
        bounds = [(1e-3, 5),  (1e-1, 20),  (1e-1, 20), (1, 60), (1e-4, 1-1e-4),  (1e-10, 1),   (1e-8, 50),   (1e-8, 10), (1e-8, 50),   (-10, 10)]  #bounds = []
        p_bounds=[(1e-3, 2),  (2, 10),     (2, 15),    (5, 30), (0.3, 0.7),      (1e-10, 0.5), (1e-8, 50),   (1e-8, 5), (1e-8, 20),    (-5, 5)] # plausible bounds [PLB, PUB], more narrow than the bounds
        if self.config['nConfK'] == 4:
            params.extend(['kC2', 'mC2'])
            bounds.extend([(1e-8, 50),   (-10, 10)])  
            p_bounds.extend([(1e-8, 20), (-5, 5)])
        if self.config['laps_flag'] == 1:
            params.append('laps')
            bounds.append((1e-8, 0.5)) 
            p_bounds.append((1e-8, 0.5)) 
        if self.config['Causal_readout'] == 'Bay' and self.config['dec_noise_flag'] in (1, -1):
            params.append('logbeta')
            bounds.append((-4, 4))  
            p_bounds.append((-0.5, 1.5))
        if self.config['Causal_readout'] == 'Bay' and self.config['dec_noise_flag'] in (2, -1):
            params.append('logbeta_conf')
            bounds.append((-4, 4))  
            p_bounds.append((-0.5, 1.5))
        if self.config['Causal_readout'] != 'Bay':
            params.append('epsilon')
            bounds.append((1e-1, 20))  
            p_bounds.append((1e-1, 20))  
        
        return params, bounds, p_bounds

    def getCondRep(self, nIntSamples:int =10000):
        condRep = np.repeat(self.conditions, repeats=nIntSamples, axis=0)
        return condRep

    def add_decision_noise(self, sim_post, conf):
        '''help fuction to add decision noise for simulation functions'''
        if conf==0:
            beta = 10**self.params.logbeta
        elif conf==1:
            beta = 10**self.params.logbeta_conf
        sim_post = np.random.gamma(np.clip(sim_post * beta, 1e-16 , None), 1) # add gamma noise
        sim_post /= sim_post.sum(axis=1, keepdims=True) # normalize
        return sim_post
    
    def compute_post_cdf_bins(self, s_hat_common, sV_hat_indep, sA_hat_indep, varVA_hat, varV_hat, varA_hat, Atrial, sim_post, Range = 100, nbin = 200):
        binSize = Range / (nbin - 2)
        half_range = Range/2 + binSize/2
        responseLoc_boundaries = np.concatenate((
            [-np.inf],
            np.linspace(-half_range, half_range, nbin - 1),
            [np.inf]
        )) # 1D array : Num of edges 

        #start_time = time.time()
        postd_c1 = np.diff(norm.cdf(responseLoc_boundaries[:, np.newaxis], s_hat_common[np.newaxis, :], np.sqrt(varVA_hat[np.newaxis, :])), axis = 0)
        postd_c2v = np.diff(norm.cdf(responseLoc_boundaries[:, np.newaxis], sV_hat_indep[np.newaxis, :], np.sqrt(varV_hat[np.newaxis, :])), axis = 0)
        postd_c2a = np.diff(norm.cdf(responseLoc_boundaries[:, np.newaxis], sA_hat_indep[np.newaxis, :], np.sqrt(varA_hat)), axis = 0)   
        #end_time = time.time()
        #print(f"Execution time: {end_time - start_time:.6f} seconds")  
        
        postd_a = sim_post[:,0] * postd_c1 + sim_post[:,1] * postd_c2a
        postd_v = sim_post[:,0] * postd_c1 + sim_post[:,1] * postd_c2v
        #normalize 
        postd_a = postd_a/np.sum(postd_a, axis = 0)
        postd_v = postd_v/np.sum(postd_v, axis = 0)
        map_a_bin =  np.argmax(postd_a, axis = 0) # from 0 - nbin-1
        map_v_bin =  np.argmax(postd_v, axis = 0)
        post_pdf = np.where(Atrial, postd_a, postd_v)
        
        # make it into cdf for computation
        post_cdf = np.cumsum(post_pdf, axis=0)
        
        return post_cdf, post_pdf, binSize, nbin, map_a_bin, map_v_bin, responseLoc_boundaries, Range
       
    def run_simulation(self, inputs, trialConds=None, interimOutput = False):
        '''
        Parameters
        ----------
        inputs : the value of the tuning parameters 
        trialConds : numpy array
        each row represents the condition of a trial [A, V, V reliability 1 - high, 2 - low, A(1) or V(0) block]
        dimention - ntrial * 4
        alternative:  model.getCondRep(nIntSamples=5000)
        interimOutput: boolean
            default: False
        Returns
        -------
        predData: numpy.array
            trialConds + response data
        All intermediate data if interimOutput is True

        '''
        if trialConds is None:
            trialConds = self.getCondRep()
        # can be called with a different nsim with 
        # custom_trialConds = model.getCondRep(nIntSamples=5000)
        # model.run_simulation(trialConds=custom_trialConds)

        # update est params
        self.update_est_params(inputs)
        #np.random.seed(42)
        ## simple parameter transformation for easier and cleaner computation
        ntrial = trialConds.shape[0]
        trial_indices = np.arange(ntrial)
        # indicator for trial types
        Atrial = trialConds[:, 3] == 1
        resp = np.zeros((ntrial, 4), dtype=float) #init a np array to save response data
        
        # two levels of visual reliability  1 - high, 2 - low, correpond to two visual variance 'sig_Vrh' and 'sig_Vrl'
        varV = np.zeros(ntrial)
        varV[trialConds[:, 2]==1] = self.params.sig_Vrh**2
        varV[trialConds[:, 2]==2] = self.params.sig_Vrl**2

        varA = self.params.sig_A**2
        varP = self.params.sig_P**2
        # if to keep the same shape
        # varA = np.full((ntrial, 1), self.params.sig_A**2)
        # varP = np.full((ntrial, 1), self.params.sig_P**2)

        varVA_hat = 1 / (1/varV + 1/varA + 1/varP)
        varV_hat = 1 / (1/varV + 1/varP)
        varA_hat = 1 / (1/varA + 1/varP)
        var_common = varV * varA + varV * varP + varA * varP
        varV_indep = varV + varP
        varA_indep = varA + varP

        ###########################################
        #############  AV Phase  ##################
        ###########################################

        # sample A and V measurements from gaussion distribution
        xA = trialConds[:, 0] + self.params.sig_A * np.random.randn(ntrial)
        xV = trialConds[:, 1] + np.sqrt(varV) * np.random.randn(ntrial)

        # compute the posterior causal structure based on BCI model
        s_hat_common = (xV/varV + xA/varA + self.params.mu_P/varP) * varVA_hat
        sV_hat_indep = (xV/varV + self.params.mu_P/varP) * varV_hat
        sA_hat_indep = (xA/varA + self.params.mu_P/varP) * varA_hat

        quad_common = (xV - xA)**2 * varP + (xV - self.params.mu_P)**2 * varA + (xA - self.params.mu_P)**2 * varV
        quadV_indep = (xV - self.params.mu_P)**2
        quadA_indep = (xA - self.params.mu_P)**2

        likelihood_common = np.exp(-quad_common / (2 * var_common)) / (2 * np.pi * np.sqrt(var_common))
        likelihoodV_indep = np.exp(-quadV_indep / (2 * varV_indep)) / np.sqrt(2 * np.pi * varV_indep)
        likelihoodA_indep = np.exp(-quadA_indep / (2 * varA_indep)) / np.sqrt(2 * np.pi * varA_indep)
        likelihood_indep = likelihoodV_indep * likelihoodA_indep

        post_common = likelihood_common * self.params.p_com
        post_indep = likelihood_indep * (1 - self.params.p_com)
        post_C1 = post_common / (post_common + post_indep)
        sim_post = np.column_stack((post_C1, 1 - post_C1))
        
        # final A, V spatial read-out for various strategies
        if self.config['est_var'] == 'PME':
            sV_hat = post_C1 * s_hat_common + sim_post[:,1] * sV_hat_indep
            sA_hat = post_C1 * s_hat_common + sim_post[:,1] * sA_hat_indep

        elif self.config['est_var'] == 'MS':
            sV_hat = (post_C1 > 0.5) * s_hat_common + (post_C1 <= 0.5) * sV_hat_indep
            sA_hat = (post_C1 > 0.5) * s_hat_common + (post_C1 <= 0.5) * sA_hat_indep

            var_V_hat_MS = (post_C1 > 0.5) * varVA_hat + (post_C1 <= 0.5) * varV_hat
            var_A_hat_MS = (post_C1 > 0.5) * varVA_hat + (post_C1 <= 0.5) * varA_hat
            sig_hat_MS = np.sqrt(np.where(Atrial, var_A_hat_MS, var_V_hat_MS))

        elif self.config['est_var'] == 'MAP':      
            post_cdf, post_pdf, binSize, nbin, map_a_bin, map_v_bin, responseLoc_boundaries, Range =  self.compute_post_cdf_bins(s_hat_common, sV_hat_indep, sA_hat_indep, varVA_hat, varV_hat, varA_hat, Atrial, sim_post)           
            # map is the middle of the bin in which the max probability lays in 
            bin_est = np.where(Atrial, map_a_bin, map_v_bin)

            sA_hat =  responseLoc_boundaries[map_a_bin] + binSize/2
            sV_hat =  responseLoc_boundaries[map_v_bin] + binSize/2
            #deal with first/last bin and infinite values
            sA_hat[np.isposinf(sA_hat)] = Range/2
            sA_hat[np.isneginf(sA_hat)] = -Range/2
            sV_hat[np.isposinf(sV_hat)] = Range/2
            sV_hat[np.isneginf(sV_hat)] = -Range/2

        elif self.config['est_var'] == 'PM':
            eta = np.random.rand(ntrial)
            sV_hat = (post_C1 > eta) * s_hat_common + (post_C1 <= eta) * sV_hat_indep
            sA_hat = (post_C1 > eta) * s_hat_common + (post_C1 <= eta) * sA_hat_indep

        else:
            sA_hat = np.nan
            sV_hat = np.nan

        # report the spatial location
        #Atrial = trialConds[:, 3] == 1
        shat = np.where(Atrial, sA_hat, sV_hat)
        #add motor noise before reporting
        resp[:, 0] = np.random.normal(shat, self.params.sig_r)

        
        ######
        #read out the confidence intevral
        # prepare posterior distribution for PME 
        if self.config['est_var'] == 'PME' and self.config['CI_readout'] in ['BayInt', 'BayEst', 'Entro']:
            # get the bin posterior distribution
            post_cdf, post_pdf, binSize, nbin, _, _, responseLoc_boundaries,_ =  self.compute_post_cdf_bins(s_hat_common, sV_hat_indep, sA_hat_indep, varVA_hat, varV_hat, varA_hat, Atrial, sim_post) 
            # find which bin has the pme
            bin_est = np.digitize(shat, responseLoc_boundaries, right=False) - 1 # bins can goes from 0 to nbin-1
                    
                    
        if self.config['CI_readout'] == 'BayInt':
            #interval-based Bayesian estimation            
            if self.config['est_var'] == 'MS':
                # the distribution is either p(s_a|x_a, x_v, c=1) or p(s_a|x_a, x_v, c=2). Both are gaussians. 
                alpha = 1 - self.params.rng
                z = norm.ppf(1 - alpha / 2)  # e.g., ≈ 1.96 for 95% confidence interval
                resp[:, 1] = self.params.a * 2 * z * sig_hat_MS + self.params.b

            else:
                #trial_indices = np.arange(ntrial)
                center_cdf = post_cdf[bin_est, trial_indices]
                # the cap trials will have bigger intervals
                cap_trial_l = (center_cdf < self.params.rng/2) | (bin_est == 0)
                cap_trial_r = (center_cdf > 1-self.params.rng/2) | (bin_est == nbin - 1)
                
                non_cap_trial = ~(cap_trial_l | cap_trial_r)
                abnormal_ratio = np.sum(~non_cap_trial)/len(non_cap_trial)
                
                # init a array to save the best bin size (to cover rng%)
                best_bin_n = np.ones(ntrial)
                #start_time = time.time()
                while True:
                    half_width = ((best_bin_n - 1) / 2).astype(int)
                    start_bin = np.clip(bin_est - half_width - 1, 0, nbin - 1)
                    end_bin   = np.clip(bin_est + half_width    , 0, nbin - 1)
                    prob_rng = post_cdf[end_bin, trial_indices] - post_cdf[start_bin, trial_indices]
            
                    need_more_bins = (prob_rng < self.params.rng) & (~cap_trial_l)
                    if not np.any(need_more_bins):
                        break
                    best_bin_n[need_more_bins] += 2
                #end_time = time.time()
                #print(f"Execution time: {end_time - start_time:.6f} seconds")    
                resp[~cap_trial_l, 1] = self.params.a * binSize *  (end_bin[~cap_trial_l] - start_bin[~cap_trial_l] + 1)  + self.params.b
                
                # compute separately for cap_trial_l 
                best_bin_n = np.ones(ntrial) 
                # situation 1:  first bin has the MAP and also center_cdf at that bin is greater than the range - best_bin_n = 1
                # situation 2:  first bin has the MAP and  center_cdf at that bin is not greater than the range - 
                # situation 3: the left side is not greater than half of the range. 
                #((bin_est == 0) & (center_cdf < self.params.rng)) | (center_cdf < self.params.rng/2)
                #start_time = time.time()
                while True:
                    half_width = ((best_bin_n - 1) / 2).astype(int)
                    end_bin = np.clip(bin_est + half_width, 0, nbin-1)
                    prob_rng = post_cdf[end_bin.astype(int), trial_indices] 
                    
                    need_more_bins = (prob_rng < self.params.rng) & cap_trial_l
                    if not np.any(need_more_bins):
                        break  
                    best_bin_n[need_more_bins] += 2
                #end_time = time.time()
                #print(f"Execution time: {end_time - start_time:.6f} seconds")    
                resp[cap_trial_l, 1] = self.params.a * binSize *  (end_bin[cap_trial_l] + 1)  + self.params.b
            
        elif self.config['CI_readout'] == 'BayEst':
            #estimate-based Bayesian estimation
            if self.config['est_var'] == 'MS':
                # the distribution is either p(s_a|x_a, x_v, c=1) or p(s_a|x_a, x_v, c=2). Both are gaussians. 
                resp[:, 1] = self.params.b - self.params.a /(sig_hat_MS * np.sqrt(2 * np.pi))

            else:
                # bin_est could be the first or last bin and that would be problematic
                # if np.sum((bin_est == 0) | (bin_est == nbin - 1)) > 0:
                #     raise ValueError("MAP/PME best bin contains out-of-bound bin indices.")
                
                prob_rng = post_pdf[bin_est, trial_indices]
                resp[:, 1] = self.params.b - self.params.a * prob_rng

        elif self.config['CI_readout'] == 'Entro':
            #entropy-based estimation (non-bayesian)
            if self.config['est_var'] == 'MS':
                # the distribution is either p(s_a|x_a, x_v, c=1) or p(s_a|x_a, x_v, c=2). Both are gaussians. 
                resp[:, 1] = self.params.b + self.params.a * np.log(np.sqrt(2 * np.pi * np.e) * sig_hat_MS)

            else:
                # bin_est could be the first or last bin and that would be problematic
                # if np.sum((bin_est == 0) | (bin_est == nbin - 1)) > 0:
                #     raise ValueError("MAP/PME best bin contains out-of-bound bin indices.")
                
                diff_entropy = -np.sum(post_pdf * np.log(np.where(post_pdf > 0, post_pdf, np.finfo(float).eps)), axis=0) + np.log(binSize)  # shape: (trials,)
                resp[:, 1] = self.params.b + self.params.a * diff_entropy

        elif self.config['CI_readout'] == 'DistXdiff':
            #distance-based estimation (non-bayesian)
            d = np.abs(xA - xV)
            #resp[:, 1] = self.params.a * d + self.params.b
            resp[:, 1] = self.params.a * d**2 * np.exp(-d) + self.params.b
            
        elif self.config['CI_readout'] == 'DistSdiff':
            #distance-based estimation (non-bayesian)
            d = np.abs(s_hat_common - np.where(Atrial, sA_hat_indep, sV_hat_indep))
            #resp[:, 1] = self.params.a * d + self.params.b
            resp[:, 1] = self.params.a * d**2 * np.exp(-d) + self.params.b

        # add motor noise
        resp[:, 1] = np.random.normal(resp[:, 1], self.params.sig_r)

        # add decision noise for causal decision and/or confidence - 
        # output: sim_post used in causal decision read out, sim_post_conf for confidence read-out
        if self.config['dec_noise_flag'] == 1:
            # same noise for causal decision/confidence
            sim_post = self.add_decision_noise(sim_post, 0)
            sim_post_conf = sim_post # same for confidence
        elif self.config['dec_noise_flag'] == 2:
            # only noise for causal confidence
            sim_post_conf = self.add_decision_noise(sim_post, 1)
        elif self.config['dec_noise_flag'] == -1:
            # separate noises for confidence and decision
            sim_post_conf = self.add_decision_noise(sim_post, 1)
            sim_post = self.add_decision_noise(sim_post, 0)
        else:
            sim_post_conf = sim_post

        # read-out the causal decision/confidence 
        if self.config['Causal_readout'] == 'Bay' :
            # Bayesian - sim_post
            # Output: 1 - common source, 2 - separate source
            resp[:, 2] = np.argmax(sim_post, axis=1) + 1 # add lapse here or during estimation????

            #confidence
            if self.config['nConfK'] == 2:
                sim_c_map = np.max(sim_post_conf, axis=1) # max posterior prob from both category
                # symmetrical boundary
                #resp[:, 3] = 1 / (1 + np.exp(-1* sim_c_map * self.params.kC1 + self.params.mC1))
                # kC1>0 , mC1>0
                
                # alternative : linear
                resp[:, 3] = sim_c_map * self.params.kC1 + self.params.mC1
                # kC1>0 , mC1 can be positive or negative

            elif self.config['nConfK'] == 4:
                # asymmetrical boundary
                com_trial = resp[:, 2] == 1
                #resp[com_trial, 3] = 1 / (1 + np.exp(-1* sim_post_conf[com_trial, 0] * self.params.kC1 + self.params.mC1))
                #resp[~com_trial, 3] = 1 / (1 + np.exp(-1* sim_post_conf[~com_trial, 1] * self.params.kC2 + self.params.mC2))
                # kC1>0 , mC1>0, same for c1

                # alternative : linear
                resp[com_trial, 3] = sim_post_conf[com_trial, 0] * self.params.kC1 + self.params.mC1
                resp[~com_trial, 3] = sim_post_conf[~com_trial, 1] * self.params.kC2 + self.params.mC2
                # kC1>0 , mC1 can be positive or negative
        else:
            # Non-Bayesian read-out
            if self.config['Causal_readout'].lower() == 'xdiff':
                # Measurement difference
                d = np.abs(xA - xV)
            elif self.config['Causal_readout'].lower() == 'sdiff':
                # Spatial estimate difference
                d = np.abs(s_hat_common - np.where(Atrial, sA_hat_indep, sV_hat_indep))
            # Output: 1 - common source, 2 - separate source
            resp[:, 2] = (d >= self.params.epsilon) + 1
            com_trial = resp[:, 2] == 1

            if self.config['nConfK'] == 2:
                # symmetrical boundary - a positive $\beta_{C2}$ and a negative $\beta_{C1}$
                #non-linear
                #[com_trial, 3] = 1 / (1 + np.exp(d[com_trial] * self.params.kC1 - self.params.mC1))
                #[~com_trial, 3] = 1 / (1 + np.exp(-1 * d[~com_trial] * self.params.kC1 + self.params.mC1))
                #kC1>0, mc1>0

                # linear
                resp[com_trial, 3] = -1 * d[com_trial] * self.params.kC1 + self.params.mC1
                resp[~com_trial, 3] = d[~com_trial] * self.params.kC1 + self.params.mC1
                #kC1>0, mc1 can be either positive or negative
                
            elif self.config['nConfK'] == 4:
                # asymmetrical boundary
                #resp[com_trial, 3] = 1 / (1 + np.exp(d[com_trial] * self.params.kC1 - self.params.mC1))
                #resp[~com_trial, 3] = 1 / (1 + np.exp(-1 * d[~com_trial] * self.params.kC2 + self.params.mC2))
                #kC1>0, mc1>0, kc2>0, mc2>0

                # linear
                resp[com_trial, 3] = -1 * d[com_trial] * self.params.kC1 + self.params.mC1
                resp[~com_trial, 3] = d[~com_trial] * self.params.kC2 + self.params.mC2
                #kC1, kc2>0, mc1. mc2 can be either positive or negative


        # Objective to subjective confidence level tranform?
        # transform to 0 - 100 ???
        resp[:, 3] = (resp[:, 3] - resp[:, 3].min()) / (resp[:, 3].max() -  resp[:, 3].min()) * 100
        resp[:, 3] = np.clip(np.round(np.random.normal(resp[:, 3], self.params.sig_r)), 0, 100)
        
        # add motor lapse for causal decision
        lapse_trials = np.random.rand(ntrial) < self.params.laps
        resp[lapse_trials, 2] = np.random.choice([1, 2], size=np.sum(lapse_trials))

        # concat the trial conditions and response
        predData = np.column_stack((trialConds, resp))
        #predData.dtype -'int32'
        if interimOutput:
            return predData, sim_post, sim_post_conf, xA, xV,s_hat_common, sV_hat_indep, sA_hat_indep,quad_common, quadV_indep, quadA_indep
        else:
            return predData
    
