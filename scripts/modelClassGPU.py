import numpy as np
import itertools
from dataclasses import dataclass, field
import pandas as pd
import torch 
import cProfile
import pstats
print('device is :' , torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

@dataclass
class ModelParams:
    """Stores numerical model parameters with defaults."""
    sig_Vrh: float = 1
    sig_Vrl: float = 7
    sig_V: float = 5
    sig_A: float = 8
    sig_P: float = 30
    sig_P_h: float = 30
    sig_P_l: float = 30
    sig_rs: float = 0.5
    sig_rconf: float = 0.5 
    sig_rc: float = 0.001
    gamma_rate: float = 2
    mu_P: float = 0
    p_com: float = 0.5
    p_com_h: float = 0.5
    p_com_l: float = 0.5
    rng: float = 0.3
    rngA: float = 0.3
    rngV: float = 0.3
    logbeta: float = 0
    logbeta_conf: float = 0
    k_emp: float = 1
    a:float = 1
    b:float = 0
    kC1:float = 5
    mC1:float = 1
    kC2:float = 5
    mC2:float = 1
    ci_bin_k1:float = 0
    ci_bin_k2:float = 0
    ci_bin_k3:float = 0
    cc_bin_k1:float = 0
    cc_bin_k2:float = 0
    cc_bin_k3:float = 0
    w_vis: float = 0 
    wb_v_h: float = 0
    wb_v_l: float = 0
    w_vis_h: float = 0
    w_vis_l: float = 0
    wb_v: float = 0
    wb_a: float = 0
    stim_loc: np.ndarray = field(default_factory=lambda: np.array([-10, -5, 0, 5, 10]))
    stim_loc_len: int = 5

class ConfiModel:
    def __init__(self, m_id:int, exp_config : dir, params: ModelParams = ModelParams(), vrel:int = 0):
        self.params = params
        self.m_id = m_id
        self.exp_config = exp_config
        self.vrel=vrel
        self.conditions = self.generate_conditions()
        self.model_config = self.load_config()
        self.estParamsNames, self.bounds, self.p_bounds = self.est_params()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def display_parameters(self):
        # for param, value in self.params.__dict__.items():
        #     print(f"{param}: {value}")
        print("\nParams for estimation are")
        for param in self.estParamsNames:
            print(f"{param}: {getattr(self.params, param)}")
        # for item in self.estParamsNames:
        #     print(item)

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
        cond = np.vstack((cond_A, cond_V)).astype(int)
        if self.exp_config['split_data_f'] ==1:
            return cond[cond[:,2]==self.vrel]
        else:
            return cond
    
    def load_config(self):
        """
        Load model configuration from an Excel file and filter by model_id.
        """
        config_df = pd.read_csv('m_config.csv')  # Read full config file
        model_config = config_df[config_df["m_id"] == self.m_id]  # Filter by model_id

        if model_config.empty:
            raise ValueError(f"Model ID {self.m_id} not found in configuration file.")

        return model_config.iloc[0].to_dict()  # Convert to dictionary
    
    def update_est_params(self, x):
        # Check if the length of x and self.estParamsNames is the same
        if len(x) != len(self.estParamsNames):
            raise ValueError("Length of x does not match the number of estimated parameters.")
        
        for i, paramName in enumerate(self.estParamsNames):
            setattr(self.params, paramName, x[i])

    def est_params(self):
        if self.exp_config['split_data_f'] == 0:
            params = ['sig_Vrh',  'sig_Vrl'   ,'sig_A'      ,'kC1'           ,'mC1'   , 'a'            ,'b']
            bounds = [(1e-3, 5),  (1e-1, 20)  ,(1e-1, 20)   ,(1e-8, 50)      ,(-10, 10), (1e-8, 50)     ,(1e-8, 10)]  # bounds = []
            p_bounds=[(1e-3, 2),  (2, 10)     ,(2, 15)      ,(1e-8, 20)      ,(-5, 5)  , (1e-8, 50)   ,(1e-8, 5)]  # plausible bounds [PLB, PUB], more narrow than the bounds
        elif self.exp_config['split_data_f'] == 1:
            params = [ 'sig_V'   ,'sig_A'      ,'kC1'              ,'mC1'   , 'a'            ,'b' ]
            bounds = [ (1e-1, 20)  ,(1e-1, 20)   ,(1e-8, 50)      ,(-10, 10), (1e-8, 50)     ,(1e-8, 10)]  # bounds = []
            p_bounds=[ (2, 10)     ,(2, 15)      ,(1e-8, 20)      ,(-5, 5)  , (1e-8, 50)   ,(1e-8, 5)]  # plausible bounds [PLB, PUB], more narrow than the bounds
        elif self.exp_config['split_data_f'] == 2:
            params = ['sig_Vrh',  'sig_Vrl'   ,'sig_A'   , 'a'            ,'b' ]
            bounds = [(1e-3, 5),  (1e-1, 20)  ,(1e-1, 20), (1e-8, 50)     ,(1e-8, 10)]  # bounds = []
            p_bounds=[(1e-3, 2),  (2, 10)     ,(2, 15)   , (1e-8, 50)   ,(1e-8, 5)]  # plausible bounds [PLB, PUB], more narrow than the bounds

        # else:
        #     params.extend(['a'            ,'b'])
        #     bounds.extend([(1e-8, 50)     ,(1e-8, 10)])  
        #     p_bounds.extend([(1e-8, 50)   ,(1e-8, 5)])

        if self.exp_config['eccent_sig_flag'] == 1:
            params.append('w_vis')
            bounds.append((1e-2, 1))  
            p_bounds.append((1e-1, 0.7))
        elif self.exp_config['eccent_sig_flag'] == 2:
            params.extend(['w_vis_h', 'w_vis_l'])
            bounds.extend([(1e-2, 1), (1e-2, 1)])  
            p_bounds.extend([(1e-1, 0.7), (1e-1, 0.7)])

        if self.exp_config['eccent_mu_flag'] == 1:
            params.extend(['wb_v', 'wb_a'])
            bounds.extend([(0, 0.5),   (0, 0.5)])  
            p_bounds.extend([(0, 0.2), (0, 0.2)])
        elif self.exp_config['eccent_mu_flag'] == 2:
            params.extend(['wb_v_h', 'wb_v_l', 'wb_a'])
            bounds.extend([(0, 0.5),   (0, 0.5), (0, 0.5)])  
            p_bounds.extend([(0, 0.2), (0, 0.2), (0, 0.2)]) 

        if self.model_config['Causal_readout'] == 'Bay' or self.model_config['perc_readout'] == 'Bay':
            if self.exp_config['share_prior_f']==1:
                params.extend(['sig_P'    ,'p_com'])
                bounds.extend([(1, 60)    ,(1e-4, 1-1e-4)])  
                p_bounds.extend([(15, 30)  ,(0.3, 0.7)])
            elif self.exp_config['share_prior_f']==0:
                params.extend(['sig_P'    ,'p_com_h',       'p_com_l'])
                bounds.extend([(1, 60)    ,(1e-4, 1-1e-4),  (1e-4, 1-1e-4)])  
                p_bounds.extend([(15, 30)  ,(0.1, 0.9),      (0.1, 0.9)])
            elif self.exp_config['share_prior_f']==2:
                params.extend(['sig_P_h'  ,'sig_P_l', 'p_com'])
                bounds.extend([(1, 60)    ,(1, 60)  ,  (1e-4, 1-1e-4)])  
                p_bounds.extend([(15, 30)  ,(15, 30)  ,  (0.1, 0.9)])
            elif self.exp_config['share_prior_f']==3:
                params.extend(['sig_P_h'  ,'sig_P_l','p_com_h',       'p_com_l'])
                bounds.extend([(1, 60)    ,(1, 60)  ,(1e-4, 1-1e-4)   ,(1e-4, 1-1e-4)])  
                p_bounds.extend([(15, 30)  ,(15, 30)  ,(0.1, 0.9)       ,(0.1, 0.9)])

        if self.model_config['CI_readout'] == 'BayInt' and self.exp_config['rng_flag'] == 0 and self.exp_config['split_data_f']<=2 :
            params.append('rng')
            bounds.append((1e-8, 0.8)) 
            p_bounds.append((0.1, 0.3)) 
        elif self.model_config['CI_readout'] == 'BayInt' and self.exp_config['rng_flag'] == 1 and self.exp_config['split_data_f']<=2:
            params.extend(['rngA', 'rngV'])
            bounds.extend([(1e-8, 0.8), (1e-8, 0.8)]) 
            p_bounds.extend([(0.1, 0.3), (0.1, 0.3)])   

        if self.model_config['Causal_readout'] != 'Bay' or self.model_config['perc_readout'] != 'Bay':
            params.append('k_emp')
            bounds.append((1, 50))  
            p_bounds.append((1, 20))

        if self.exp_config['gamma_noise_flag'] == 1:
            params.append('gamma_rate')
            bounds.append((1, 50))  
            p_bounds.append((2, 20))

        if self.exp_config['symetrical_causal_flag'] == 0:
            params.extend(['kC2', 'mC2'])
            bounds.extend([(1e-8, 50),   (-10, 10)])  
            p_bounds.extend([(1e-8, 20), (-5, 5)])

        if self.exp_config['continous_flag'] == 0:
            params.extend(['ci_bin_k1', 'ci_bin_k2', 'ci_bin_k3', 'cc_bin_k1', 'cc_bin_k2', 'cc_bin_k3'])
            bounds.extend([(1e-8, 30),  (1e-8, 30),   (1e-8, 30), (1e-8, 1),  (1e-8, 1),  (1e-8, 1)])  
            p_bounds.extend([(1e-8, 30),  (1e-8, 30), (1e-8, 30), (1e-8, 1),  (1e-8, 1),  (1e-8, 1)])
        elif self.exp_config['continous_flag'] == 1:
            params.extend(['sig_rs', 'sig_rconf', 'sig_rc'])
            bounds.extend([(1e-8, 0.8),  (1e-8, 0.8), (1e-8, 0.8)])  
            p_bounds.extend([(1e-8, 0.5), (1e-8, 0.5), (1e-8, 0.5)])

        if self.model_config['Causal_readout'] == 'Bay' and self.exp_config['decision_noise_flag'] in (1, -1, -2):
            params.append('logbeta')
            bounds.append((-4, 4))  
            p_bounds.append((-0.5, 1.5))
        if self.model_config['Causal_readout'] == 'Bay' and self.exp_config['decision_noise_flag'] in (2, -1):
            params.append('logbeta_conf')
            bounds.append((-4, 4))  
            p_bounds.append((-0.5, 1.5))

        return params, bounds, p_bounds
    
    def getCondRep(self, nIntSamples: int = 10000) -> torch.Tensor:
        """
        Returns a repeated version of self.conditions as a torch.Tensor on the appropriate device.
        Shape: [n_conditions * nIntSamples, 4]
        """
        # Repeat along rows
        condRep_np = np.repeat(self.conditions, repeats=nIntSamples, axis=0)
        # Convert to torch.Tensor and send to device
        condRep_tensor = torch.tensor(condRep_np, dtype=torch.float32, device=self.device)
        return condRep_tensor

    def add_decision_noise(self, sim_post, conf):
        """
        Add gamma-distributed noise to the simulated posterior using PyTorch.
        `sim_post` should be a torch.Tensor (n_trials, 2), on the correct device.
        """
        #device = sim_post.device
        if conf == 0:
            beta = 10 ** self.params.logbeta
        elif conf == 1:
            beta = 10 ** self.params.logbeta_conf

        # Clamp to avoid zero
        sim_post = torch.clamp(sim_post * beta, min=1e-6)

        # Gamma sampling: concentration (shape) = sim_post, rate = 1.0
        gamma_noise = torch.distributions.Gamma(sim_post, torch.ones_like(sim_post, device=self.device)).sample()

        # Normalize
        sim_post_noisy = gamma_noise / gamma_noise.sum(dim=1, keepdim=True)

        return sim_post_noisy
    
    def compute_post_cdf_bins(self, s_hat_common, sV_hat_indep, sA_hat_indep, 
                            varVA_hat, varV_hat, varA_hat, Atrial, sim_post, 
                            Range=100, nbin=200):
        # Calculate bin parameters
        binSize = Range / (nbin - 2)
        half_range = Range / 2 + binSize / 2
        
        # Create bin boundaries on GPU
        responseLoc_boundaries = torch.cat([
            torch.tensor([-float('inf')], device=self.device),
            torch.linspace(-half_range, half_range, steps=nbin - 1, device=self.device),
            torch.tensor([float('inf')], device=self.device)
        ])

        # Standard Normal distribution for CDF calculations
        norm_dist = torch.distributions.Normal(0, 1)

        # Helper function to compute bin probabilities
        def compute_bin_probs(mean, std):
            # Standardize boundaries: (nbin+1, 1) - (1, n_trials) -> (nbin+1, n_trials)
            standardized = (responseLoc_boundaries.unsqueeze(1) - mean.unsqueeze(0)) / std.unsqueeze(0)
            cdf_vals = norm_dist.cdf(standardized)
            return cdf_vals[1:] - cdf_vals[:-1]  # Differences give bin probs

        # Compute probabilities for each distribution
        postd_c1 = compute_bin_probs(s_hat_common, torch.sqrt(varVA_hat))
        postd_c2v = compute_bin_probs(sV_hat_indep, torch.sqrt(varV_hat))
        postd_c2a = compute_bin_probs(sA_hat_indep, torch.sqrt(varA_hat))

        # Mix posteriors using sim_post weights
        w1 = sim_post[:, 0]
        w2 = sim_post[:, 1]
        postd_a = w1 * postd_c1 + w2 * postd_c2a
        postd_v = w1 * postd_c1 + w2 * postd_c2v

        # Normalize probabilities
        postd_a = postd_a / postd_a.sum(dim=0, keepdim=True)
        postd_v = postd_v / postd_v.sum(dim=0, keepdim=True)

        # Find MAP estimates
        map_a_bin = torch.argmax(postd_a, dim=0)
        map_v_bin = torch.argmax(postd_v, dim=0)

        # Select PDF based on Atrial condition
        post_pdf = torch.where(Atrial, postd_a, postd_v)

        # Convert to CDF
        post_cdf = torch.cumsum(post_pdf, dim=0)

        return (
            post_cdf, post_pdf, binSize, nbin,
            map_a_bin, map_v_bin,
            responseLoc_boundaries, Range
        )
    
    def discretize_loc(self, resp: torch.Tensor) -> torch.Tensor:
        # Ensure stim_loc is a tensor on the same device
        stim_loc = torch.as_tensor(self.params.stim_loc, device=resp.device, dtype=resp.dtype)

        # Compute absolute differences (resp: [N], stim_loc: [M])
        # Result shape: [N, M]
        diffs = torch.abs(resp.unsqueeze(1) - stim_loc.unsqueeze(0))

        # Get index of minimum along stim_loc axis
        t = torch.argmin(diffs, dim=1)

        # Gather the discretized response
        dis_resp = stim_loc[t]

        return dis_resp

    def perform_BCI(self, xV, xA, sigV, sigA, ntrial, Atrial, trialConds):
        """ 
        Perform Bayesian inference for the BCI model.
        """
        # Variance for A, V, and P
        varV = sigV**2
        varA = sigA**2

        if self.exp_config['share_prior_f']==0:
            varP = torch.full((ntrial,), self.params.sig_P**2, device=self.device)
            p_com = self.params.p_com
        elif self.exp_config['share_prior_f']==1:
            p_com = torch.where(trialConds[:, 2] == 1, self.params.p_com_h, self.params.p_com_l)
            varP = torch.full((ntrial,), self.params.sig_P**2, device=self.device)
        elif self.exp_config['share_prior_f']==2:
            p_com = self.params.p_com
            varP = torch.where(trialConds[:, 2] == 1, self.params.sig_P_h**2, self.params.sig_P_l**2)
        elif self.exp_config['share_prior_f']==3:
            p_com = torch.where(trialConds[:, 2] == 1, self.params.p_com_h, self.params.p_com_l)
            varP = torch.where(trialConds[:, 2] == 1, self.params.sig_P_h**2, self.params.sig_P_l**2)

        #varP = torch.full((ntrial,), self.params.sig_P**2, device=self.device)
        varVA_hat = 1 / (1/varV + 1/varA + 1/varP)
        varV_hat = 1 / (1/varV + 1/varP)
        varA_hat = 1 / (1/varA + 1/varP)
        var_common = varV * varA + varV * varP + varA * varP
        varV_indep = varV + varP
        varA_indep = varA + varP

        # Compute the posterior causal structure based on BCI model
        s_hat_common = (xV/varV + xA/varA + self.params.mu_P/varP) * varVA_hat
        sV_hat_indep = (xV/varV + self.params.mu_P/varP) * varV_hat
        sA_hat_indep = (xA/varA + self.params.mu_P/varP) * varA_hat

        quad_common = (xV - xA)**2 * varP + (xV - self.params.mu_P)**2 * varA + (xA - self.params.mu_P)**2 * varV
        quadV_indep = (xV - self.params.mu_P)**2
        quadA_indep = (xA - self.params.mu_P)**2

        likelihood_common = torch.exp(-quad_common / (2 * var_common)) / (2 * torch.pi * torch.sqrt(var_common))
        likelihoodV_indep = torch.exp(-quadV_indep / (2 * varV_indep)) / torch.sqrt(2 * torch.pi * varV_indep)
        likelihoodA_indep = torch.exp(-quadA_indep / (2 * varA_indep)) / torch.sqrt(2 * torch.pi * varA_indep)
        likelihood_indep = likelihoodV_indep * likelihoodA_indep
            
        post_common = likelihood_common * p_com
        post_indep = likelihood_indep * (1 - p_com)
        post_C1 = post_common / (post_common + post_indep)
        sim_post = torch.stack((post_C1, 1 - post_C1), dim=1)
        
        # Final A, V spatial read-out for various strategies
        if self.model_config['est_var'] == 'MA' and self.model_config['perc_readout']=='Bay':
            sV_hat = post_C1 * s_hat_common + sim_post[:, 1] * sV_hat_indep
            sA_hat = post_C1 * s_hat_common + sim_post[:, 1] * sA_hat_indep

            # Get the bin posterior distribution
            post_cdf, post_pdf, binSize, nbin, _, _, responseLoc_boundaries, _ = self.compute_post_cdf_bins(
                s_hat_common, sV_hat_indep, sA_hat_indep, varVA_hat, varV_hat, varA_hat, Atrial, sim_post
            ) 
            shat = torch.where(Atrial, sA_hat, sV_hat)
            # Find which bin has the MA
            bin_est = torch.searchsorted(responseLoc_boundaries[1:-1], shat, right=False)
            bin_est = torch.clamp(bin_est, 0, nbin - 1)
            
            return shat, sim_post, post_cdf, post_pdf, bin_est, binSize, nbin
        
        elif self.model_config['est_var'] == 'MS' and self.model_config['perc_readout']=='Bay':
            sV_hat = torch.where(post_C1 > 0.5, s_hat_common, sV_hat_indep)
            sA_hat = torch.where(post_C1 > 0.5, s_hat_common, sA_hat_indep)

            var_V_hat_MS = torch.where(post_C1 > 0.5, varVA_hat, varV_hat)
            var_A_hat_MS = torch.where(post_C1 > 0.5, varVA_hat, varA_hat)
            sig_hat_MS = torch.sqrt(torch.where(Atrial, var_A_hat_MS, var_V_hat_MS))
            shat = torch.where(Atrial, sA_hat, sV_hat)

            return shat, sig_hat_MS, sim_post
        
        elif self.model_config['est_var'] == 'MAP' and self.model_config['perc_readout']=='Bay':      
            post_cdf, post_pdf, binSize, nbin, map_a_bin, map_v_bin, responseLoc_boundaries, Range = self.compute_post_cdf_bins(
                s_hat_common, sV_hat_indep, sA_hat_indep, varVA_hat, varV_hat, varA_hat, Atrial, sim_post
            )           
            # MAP is the middle of the bin in which the max probability lays in 
            bin_est = torch.where(Atrial, map_a_bin, map_v_bin)

            sA_hat = responseLoc_boundaries[map_a_bin] + binSize/2
            sV_hat = responseLoc_boundaries[map_v_bin] + binSize/2
            
            # Deal with first/last bin and infinite values
            sA_hat = torch.where(torch.isposinf(sA_hat), Range/2, sA_hat)
            sA_hat = torch.where(torch.isneginf(sA_hat), -Range/2, sA_hat)
            sV_hat = torch.where(torch.isposinf(sV_hat), Range/2, sV_hat)
            sV_hat = torch.where(torch.isneginf(sV_hat), -Range/2, sV_hat)

            shat = torch.where(Atrial, sA_hat, sV_hat)
            return shat, sim_post, post_cdf, post_pdf, bin_est, binSize, nbin
        
        else:
            return sim_post
        
    def perform_heuristic_inference(self, xV, xA, varV, Atrial):

        w_c1 = torch.exp(-torch.abs(xA - xV) / (self.params.k_emp * varV))
        w_c2 = 1 - w_c1
    
        if self.model_config['est_var'] == 'MA' and self.model_config['perc_readout']!='Bay':
            sV_hat = w_c2 * xV + w_c1 * (xV/varV + (1-1/varV) * xA )
            sA_hat = w_c2 * xA + w_c1 * (xV/varV + (1-1/varV) * xA )
            shat = torch.where(Atrial, sA_hat, sV_hat)
            return shat, w_c1
        
        elif self.model_config['est_var'] == 'MS' and self.model_config['perc_readout']!='Bay':
            shat_common = xV/varV + (1-1/varV) * xA 
            sV_hat = torch.where(w_c1 > 0.5, shat_common, xV)
            sA_hat = torch.where(w_c1 > 0.5, shat_common, xA)
            shat = torch.where(Atrial, sA_hat, sV_hat)
            return shat, w_c1
        
        else:
            return w_c1
        
    def define_sensory_noise(self, trialConds, ntrial):
        # Two levels of visual reliability  1 - high, 2 - low, correspond to two visual variance 'sig_Vrh' and 'sig_Vrl'
        sigV = torch.zeros(ntrial, device=self.device)
        if not self.exp_config['split_data_f']==1:
            sigV[trialConds[:, 2] == 1] = self.params.sig_Vrh
            sigV[trialConds[:, 2] == 2] = self.params.sig_Vrl
        else:
            sigV = torch.full((ntrial,), self.params.sig_V, device=self.device)

        sigA = torch.full((ntrial,), self.params.sig_A, device=self.device)

        if self.exp_config['eccent_sig_flag'] == 1:
            sigV = sigV * torch.sqrt(1 + (self.params.w_vis * trialConds[:, 1]) ** 2)

        elif self.exp_config['eccent_sig_flag'] == 2:
            w_vis = torch.zeros(ntrial, device=self.device)
            w_vis[trialConds[:,2]==1] = self.params.w_vis_h
            w_vis[trialConds[:,2]==2] = self.params.w_vis_l
            sigV = sigV * torch.sqrt(1 + (w_vis * trialConds[:, 1]) ** 2)

        if self.exp_config['gamma_noise_flag'] == 1:
            # the gamma distribution - mean sigV, variance sigV/gamma_rate
            sigV = torch.distributions.Gamma(concentration=sigV * self.params.gamma_rate , rate=self.params.gamma_rate).sample()
            sigA = torch.distributions.Gamma(concentration=sigA * self.params.gamma_rate , rate=self.params.gamma_rate).sample()
            sigV = torch.clamp(sigV, min=1e-6) # make sure sigV is not zero
            sigA = torch.clamp(sigA, min=1e-6)
        
        return sigV, sigA
    
    def define_sensory_mean(self, trialConds):
        if self.exp_config['eccent_mu_flag'] == 0:
            muV = trialConds[:, 1]
            muA = trialConds[:, 0]

        elif self.exp_config['eccent_mu_flag'] == 1:
            muV = trialConds[:, 1] * (1 - self.params.wb_v) # bias towards center
            muA = trialConds[:, 0] * (1 + self.params.wb_a) # bias away from center

        elif self.exp_config['eccent_mu_flag'] == 2:
            wb_v = torch.zeros(trialConds.shape[0], device=self.device)
            wb_v[trialConds[:,2]==1] = self.params.wb_v_h
            wb_v[trialConds[:,2]==2] = self.params.wb_v_l
            muV = trialConds[:, 1] * (1 - wb_v)
            muA = trialConds[:, 0] * (1 + self.params.wb_a)

        return muA, muV

    def run_simulation(self, inputs, trialConds=None, interimOutput=False):
        '''
        Parameters
        ----------
        inputs : the value of the tuning parameters 
        trialConds : numpy array or torch tensor
        each row represents the condition of a trial [A, V, V reliability 1 - high, 2 - low, A(1) or V(0) block]
        dimention - ntrial * 4
        alternative:  model.getCondRep(nIntSamples=5000)
        interimOutput: boolean
            default: False
        Returns
        -------
        predData: torch.Tensor
            trialConds + response data
        All intermediate data if interimOutput is True
        '''
        # pr = cProfile.Profile()
        # pr.enable()
        # 1. Get trialConds if not passed in
        if trialConds is None:
            trialConds = self.getCondRep() 
        # 2. If NumPy array, convert to torch.Tensor
        elif isinstance(trialConds, np.ndarray):
            trialConds = torch.tensor(trialConds, dtype=torch.float32, device=self.device)
        # 3. If already a tensor but wrong device or dtype
        else:
            if not trialConds.is_floating_point():
                trialConds = trialConds.to(dtype=torch.float32)
            trialConds = trialConds.to(device=self.device)
            

        # Update est params
        self.update_est_params(inputs)
        
        # prepare variables for common use
        ntrial = trialConds.shape[0]
        trial_indices = torch.arange(ntrial, device=self.device)
        Atrial = trialConds[:, 3] == 1  # shape: (ntrial,), dtype: torch.bool
        resp = torch.zeros((ntrial, 4), dtype=torch.float32, device=self.device) # init response
        
        sigV, sigA = self.define_sensory_noise(trialConds, ntrial)
        muA, muV = self.define_sensory_mean(trialConds)
        ########################################################
        ############# Perceptual Inference #####################
        ########################################################

        # Sample A and V measurements from gaussian distribution
        # xA = trialConds[:, 0] + sigA * torch.randn(ntrial, device=self.device)
        # xV = trialConds[:, 1] + sigV * torch.randn(ntrial, device=self.device)
        xA = muA + sigA * torch.randn(ntrial, device=self.device)
        xV = muV + sigV * torch.randn(ntrial, device=self.device)

        ############# Perceptual Inference #####################
        if self.model_config['perc_readout'] == 'Bay':
            ## performe Bayesian inference
            if self.model_config['est_var'] == 'MS':
                shat, sig_hat_MS, sim_post = self.perform_BCI(xV, xA, sigV, sigA, ntrial, Atrial, trialConds)

            elif self.model_config['est_var'] in ['MAP', 'MA']:      
                shat, sim_post, post_cdf, post_pdf, bin_est, binSize, nbin \
                      = self.perform_BCI(xV, xA, sigV, sigA, ntrial, Atrial, trialConds)

        elif self.model_config['perc_readout'] == 'NonBay':
            #  heuristic inference
            if self.exp_config['heur_var'] == 'fix_sig':
                sigV_emp = torch.zeros(ntrial, device=self.device)
                sigV_emp[trialConds[:, 2] == 1] = 2 # cloud std for high reliability visual
                sigV_emp[trialConds[:, 2] == 2] = 14 # cloud std for low reliability visual
                varV = sigV_emp**2
            elif self.exp_config['heur_var'] == 'flex_sig':
                varV = sigV**2

            shat, w_c1 = self.perform_heuristic_inference(xV, xA, varV, Atrial)

        elif self.model_config['perc_readout'] == 'singCue':
            # not integrate but two cues. 
            pass

        ################################
        # Read out the perceptual confidence interval
        if self.model_config['CI_readout'] == 'BayInt' and self.model_config['perc_readout'] == 'Bay':
            if self.exp_config['rng_flag']==1:
                rng = torch.zeros(ntrial, device=self.device)
                rng[trialConds[:, 3] == 1] = self.params.rngA
                rng[trialConds[:, 3] == 0] = self.params.rngV
            else:
                rng = torch.full((ntrial,), self.params.rng, device=self.device)

            # Interval-based Bayesian estimation            
            if self.model_config['est_var'] == 'MS':
                # The distribution is either p(s_a|x_a, x_v, c=1) or p(s_a|x_a, x_v, c=2). Both are gaussians. 
                alpha = 1 - rng  # float
                alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=self.device)
                normal = torch.distributions.Normal(0.0, 1.0)
                z = normal.icdf(1 - alpha_t / 2)
                resp[:, 1] = self.params.a * 2 * z * sig_hat_MS + self.params.b

            else:
                # Complex interval computation - keeping original logic but with PyTorch
                center_cdf = post_cdf[bin_est, trial_indices]
                
                # The cap trials will have bigger intervals
                cap_trial_l = (center_cdf < rng/2) | (bin_est == 0)
                #cap_trial_r = (center_cdf > 1-rng/2) | (bin_est == nbin - 1)

                # Init array to save the best bin size (to cover rng%)
                best_bin_n = torch.ones(ntrial, device=self.device, dtype=torch.long)
                # Main loop for non-capped trials
                while True:
                    half_width = ((best_bin_n - 1) // 2)
                    start_bin = torch.clamp(bin_est - half_width - 1, 0, nbin - 1)
                    end_bin = torch.clamp(bin_est + half_width, 0, nbin - 1)
                    prob_rng = post_cdf[end_bin, trial_indices] - post_cdf[start_bin, trial_indices]
            
                    need_more_bins = (prob_rng < rng) & (~cap_trial_l)
                    if not torch.any(need_more_bins):
                        break
                    best_bin_n[need_more_bins] += 2
                    
                #resp[~cap_trial_l, 1] = binSize * (end_bin[~cap_trial_l] - start_bin[~cap_trial_l] + 1).float() 
                resp[~cap_trial_l, 1] = self.params.a * binSize * (end_bin[~cap_trial_l] - start_bin[~cap_trial_l] + 1).float() + self.params.b
                # Compute separately for cap_trial_l 
                best_bin_n = torch.ones(ntrial, device=self.device, dtype=torch.long)
                while True:
                    half_width = ((best_bin_n - 1) // 2)
                    end_bin = torch.clamp(bin_est + half_width, 0, nbin-1)
                    prob_rng = post_cdf[end_bin, trial_indices]
                    
                    need_more_bins = (prob_rng < rng) & cap_trial_l
                    if not torch.any(need_more_bins):
                        break  
                    best_bin_n[need_more_bins] += 2
                    
                #resp[cap_trial_l, 1] =  binSize * (end_bin[cap_trial_l] + 1).float() 
                resp[cap_trial_l, 1] = self.params.a * binSize * (end_bin[cap_trial_l] + 1).float() + self.params.b
            
        elif self.model_config['CI_readout'] == 'BayEst' and self.model_config['perc_readout'] == 'Bay':
            # Estimate-based Bayesian estimation
            if self.model_config['est_var'] == 'MS':
                # The distribution is either p(s_a|x_a, x_v, c=1) or p(s_a|x_a, x_v, c=2). Both are gaussians. 
                resp[:, 1] = self.params.b + self.params.a * (sig_hat_MS * torch.sqrt(torch.tensor(2 * torch.pi))) * torch.exp(0.5 * (shat/sig_hat_MS)**2)
            else:
                prob_rng = post_pdf[bin_est, trial_indices]
                resp[:, 1] = self.params.b + self.params.a / prob_rng

        elif self.model_config['CI_readout'] == 'Entro' and self.model_config['perc_readout'] == 'Bay':
            # Entropy-based estimation (non-bayesian)
            if self.model_config['est_var'] == 'MS':
                # The distribution is either p(s_a|x_a, x_v, c=1) or p(s_a|x_v, c=2). Both are gaussians. 
                resp[:, 1] = self.params.b + self.params.a * torch.log(torch.sqrt(torch.tensor(2 * torch.pi * torch.e)) * sig_hat_MS)
            else:
                # Differential entropy computation
                eps = torch.finfo(torch.float32).eps
                safe_pdf = torch.where(post_pdf > 0, post_pdf, eps)
                diff_entropy = -torch.sum(post_pdf * torch.log(safe_pdf), dim=0) + torch.log(torch.tensor(binSize, device=self.device))
                resp[:, 1] = self.params.b + self.params.a * diff_entropy

        elif self.model_config['perc_readout'] == 'NonBay' :
            resp[:, 1] = self.params.a /(w_c1 + self.params.b)
            #resp[:, 1] = self.params.b + self.params.a /torch.abs(w_c1-0.5)


        ########################################################
        ################ Causal Inference ######################
        ########################################################
        if self.model_config['Causal_readout'] == 'Bay' and self.exp_config['split_data_f']<=2:
            if self.model_config['perc_readout']!='Bay' or self.exp_config['split_data_f']>2:
                sim_post = self.perform_BCI(xV, xA, sigV, sigA, ntrial, Atrial, trialConds)

            # Add decision noise for causal decision and/or confidence
            if self.exp_config['decision_noise_flag'] == 1:
                sim_post = self.add_decision_noise(sim_post, 0)
                sim_post_conf = sim_post  # same for confidence
            elif self.exp_config['decision_noise_flag'] == 2:
                sim_post_conf = self.add_decision_noise(sim_post, 1) # Only noise for causal confidence
            elif self.exp_config['decision_noise_flag'] == -1:
                sim_post_conf = self.add_decision_noise(sim_post, 1)
                sim_post = self.add_decision_noise(sim_post, 0) # Separate noises for confidence and decision
            elif self.exp_config['decision_noise_flag'] == -2: # only noise for causal inference
                sim_post_conf = sim_post.clone()
                sim_post = self.add_decision_noise(sim_post, 0)
            else:
                sim_post_conf = sim_post

            # Output: 1 - common source, 2 - separate source
            resp[:, 2] = torch.where(sim_post[:, 0]>.5, 1, 2)

            # confidence
            if self.exp_config['symetrical_causal_flag'] == 1:
                sim_c_map = torch.max(sim_post_conf, dim=1)[0]  # max posterior prob from both categories
                resp[:, 3] = 1 / (1 + torch.exp(-sim_c_map * self.params.kC1 + self.params.mC1)).to(resp.dtype)

            elif self.exp_config['symetrical_causal_flag'] == 0:
                com_trial = resp[:, 2] == 1
                resp[com_trial, 3] = 1 / (1 + torch.exp(-sim_post_conf[com_trial, 0] * self.params.kC1 + self.params.mC1)).to(resp.dtype)
                resp[~com_trial, 3] = 1 / (1 + torch.exp(-sim_post_conf[~com_trial, 1] * self.params.kC2 + self.params.mC2)).to(resp.dtype)


        elif self.model_config['Causal_readout'] == 'NonBay' and self.exp_config['split_data_f']<=2 :
            if self.model_config['perc_readout']!='NonBay':
                w_c1 = self.perform_heuristic_inference(xV, xA, varV, Atrial)
            # Output: 1 - common source, 2 - separate source
            resp[:, 2] = torch.where(w_c1>.5, 1, 2)
            w_c2 = 1 - w_c1
            if self.exp_config['symetrical_causal_flag'] == 1:
                resp[:, 3] = 1 / (1 + torch.exp(-w_c1 * self.params.kC1 + self.params.mC1)).to(resp.dtype)

            elif self.exp_config['symetrical_causal_flag'] == 0:
                # losgistics mapping 
                com_trial = resp[:, 2] == 1
                # asymmetrical boundary
                resp[com_trial, 3] = 1 / (1 + torch.exp(-w_c1[com_trial] * self.params.kC1 + self.params.mC1)).to(resp.dtype)
                resp[~com_trial, 3] = 1 / (1 + torch.exp(-w_c2[~com_trial] * self.params.kC2 + self.params.mC2)).to(resp.dtype)
            
        else:
            pass

        ########################################################
        ################## Final readout  ######################
        ########################################################
        if self.exp_config['continous_flag']==0:

            resp[:, 0] = self.discretize_loc(shat)
            # CI discretization
            k_ci = torch.tensor(
                [-0.1, self.params.ci_bin_k1, self.params.ci_bin_k2, self.params.ci_bin_k3, float("inf")],
                device=self.device
                ).sort().values
            resp[:, 1] = torch.clamp(
                torch.bucketize(resp[:, 1], k_ci, right=True),
                min=1, max=4
                )

            # causal confidence discretization
            k_cc = torch.tensor(
                [-0.1, self.params.cc_bin_k1, self.params.cc_bin_k2, self.params.cc_bin_k3, 1.0],
                device=self.device
                ).sort().values
            resp[:, 3] = torch.clamp(
                torch.bucketize(resp[:, 3], k_cc, right=True),
                min=1, max=4
                )
        else:
            # Add motor noise from the tracker ball mouse
            resp[:, 0] = torch.normal(shat, self.params.sig_rs)
            resp[:, 1] = torch.normal(resp[:, 1], self.params.sig_rconf)
            resp[:, 1] = resp[:, 1].clamp(min=0.1)
            # Add random noise and clip (GPU tensor version)
            noise = torch.randn(resp.shape[0], device=self.device) * self.params.sig_rc
            resp[:, 3] = torch.round(torch.clamp(resp[:, 3] + noise, 0, 1) * 100)


        # Concatenate trial conditions and response (assuming trialConds is a tensor on device)
        if self.exp_config['split_data_f']<=1:
            predData = torch.cat((trialConds, resp), dim=1)
        elif self.exp_config['split_data_f']==2:
            predData = torch.cat((trialConds, resp[:, 0:2]), dim=1)
        else:
            pass

        if interimOutput:
            return predData.numpy(), sim_post, sim_post_conf, xA, xV, post_cdf, post_pdf, bin_est, trial_indices, binSize, nbin, ntrial
        else:
            return predData.cpu().numpy()

