import numpy as np
import itertools
from dataclasses import dataclass, field
import pandas as pd
import torch 

print('device is :' , torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

##### CODE ABOVE IS FOR CHECKING CUDA FUNCTIONALITY WITH TORCH!!
@dataclass
class ModelParams:
    """Stores numerical model parameters with defaults."""
    sig_Vrh: float = 1
    sig_Vrl: float = 20
    sig_A: float = 8
    sig_P: float = 30
    sig_rs: float = 0.01
    sig_rconf: float = 0.01
    sig_rc: float = 0.01
    gamma_rate: float = 2
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        config_df = pd.read_csv('m_config0728.csv')  # Read full config file
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
        params = ['sig_Vrh',  'sig_Vrl',   'sig_A',    'sig_P', 'p_com',        'sig_rs',      'sig_rconf',      'sig_rc',       'a',             'b',       'kC1',     'mC1']
        bounds = [(1e-3, 5),  (1e-1, 20),  (1e-1, 20), (1, 60), (1e-4, 1-1e-4),  (1e-10, 1),    (1e-10, 1),    (1e-10, 1),   (1e-8, 50),   (1e-8, 10),   (1e-8, 50),   (-10, 10)]  #bounds = []
        p_bounds=[(1e-3, 2),  (2, 10),     (2, 15),    (5, 30), (0.3, 0.7),      (1e-10, 0.5), (1e-10, 0.5), (1e-10, 0.5),   (1e-8, 50),    (1e-8, 5),   (1e-8, 20),    (-5, 5)] # plausible bounds [PLB, PUB], more narrow than the bounds
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
        if self.config['gamma_var'] == 1:
            params.append('gamma_rate')
            bounds.append((1, 50))  
            p_bounds.append((1, 20))
        return params, bounds, p_bounds
    
    def getCondRep(self, nIntSamples: int = 10000):
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
        sim_post = torch.clamp(sim_post * beta, min=1e-16)

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
        
        # Simple parameter transformation for easier and cleaner computation
        ntrial = trialConds.shape[0]
        trial_indices = torch.arange(ntrial, device=self.device)

        # Boolean mask for A block trials (column 3 == 1)
        Atrial = trialConds[:, 3] == 1  # shape: (ntrial,), dtype: torch.bool

        # Initialize response matrix (ntrial x 4), float32 on GPU
        resp = torch.zeros((ntrial, 4), dtype=torch.float32, device=self.device)
        
        # # Two levels of visual reliability  1 - high, 2 - low, correspond to two visual variance 'sig_Vrh' and 'sig_Vrl'
        sigV = torch.zeros(ntrial, device=self.device)
        sigV[trialConds[:, 2] == 1] = self.params.sig_Vrh
        sigV[trialConds[:, 2] == 2] = self.params.sig_Vrl
        sigA = torch.full((ntrial,), self.params.sig_A, device=self.device)

        # the variable sigmas
        if self.config['gamma_var'] == 1:
            # the gamma distribution - mean sigV, variance sigV/gamma_rate
            sigV = torch.distributions.Gamma(concentration=sigV * self.params.gamma_rate , rate=torch.tensor(self.params.gamma_rate, device=self.device)).sample()
            sigA = torch.distributions.Gamma(concentration=sigA * self.params.gamma_rate , rate=torch.tensor(self.params.gamma_rate, device=self.device)).sample()

        # Variance for A, V, and P
        varV = sigV**2
        varA = sigA**2
        #varP = torch.tensor(self.params.sig_P**2, device=self.device)
        varP = torch.full((ntrial,), self.params.sig_P**2, device=self.device)

        varVA_hat = 1 / (1/varV + 1/varA + 1/varP)
        varV_hat = 1 / (1/varV + 1/varP)
        varA_hat = 1 / (1/varA + 1/varP)
        var_common = varV * varA + varV * varP + varA * varP
        varV_indep = varV + varP
        varA_indep = varA + varP

        ###########################################
        #############  AV Phase  ##################
        ###########################################

        # Sample A and V measurements from gaussian distribution
        xA = trialConds[:, 0] + sigA * torch.randn(ntrial, device=self.device)
        xV = trialConds[:, 1] + sigV * torch.randn(ntrial, device=self.device)

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

        post_common = likelihood_common * self.params.p_com
        post_indep = likelihood_indep * (1 - self.params.p_com)
        post_C1 = post_common / (post_common + post_indep)
        sim_post = torch.stack((post_C1, 1 - post_C1), dim=1)
        
        # Final A, V spatial read-out for various strategies
        if self.config['est_var'] == 'MA':
            sV_hat = post_C1 * s_hat_common + sim_post[:, 1] * sV_hat_indep
            sA_hat = post_C1 * s_hat_common + sim_post[:, 1] * sA_hat_indep

        elif self.config['est_var'] == 'MS':
            sV_hat = torch.where(post_C1 > 0.5, s_hat_common, sV_hat_indep)
            sA_hat = torch.where(post_C1 > 0.5, s_hat_common, sA_hat_indep)

            var_V_hat_MS = torch.where(post_C1 > 0.5, varVA_hat, varV_hat)
            var_A_hat_MS = torch.where(post_C1 > 0.5, varVA_hat, varA_hat)
            sig_hat_MS = torch.sqrt(torch.where(Atrial, var_A_hat_MS, var_V_hat_MS))

        elif self.config['est_var'] == 'MAP':      
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

        elif self.config['est_var'] == 'PM':
            eta = torch.rand(ntrial, device=self.device)
            sV_hat = torch.where(post_C1 > eta, s_hat_common, sV_hat_indep)
            sA_hat = torch.where(post_C1 > eta, s_hat_common, sA_hat_indep)

        else:
            sA_hat = torch.full((ntrial,), float('nan'), device=self.device)
            sV_hat = torch.full((ntrial,), float('nan'), device=self.device)

        # Report the spatial location
        shat = torch.where(Atrial, sA_hat, sV_hat)
        # Add motor noise before reporting
        resp[:, 0] = torch.normal(shat, torch.tensor(self.params.sig_rs, device = self.device))
        
        ######
        # Read out the confidence interval
        # Prepare posterior distribution for MA 
        if self.config['est_var'] == 'MA' and self.config['CI_readout'] in ['BayInt', 'BayEst', 'Entro']:
            # Get the bin posterior distribution
            post_cdf, post_pdf, binSize, nbin, _, _, responseLoc_boundaries, _ = self.compute_post_cdf_bins(
                s_hat_common, sV_hat_indep, sA_hat_indep, varVA_hat, varV_hat, varA_hat, Atrial, sim_post
            ) 
            # Find which bin has the MA
            bin_est = torch.searchsorted(responseLoc_boundaries[1:-1], shat, right=False)
            bin_est = torch.clamp(bin_est, 0, nbin - 1)
                    
        if self.config['CI_readout'] == 'BayInt':
            # Interval-based Bayesian estimation            
            if self.config['est_var'] == 'MS':
                # The distribution is either p(s_a|x_a, x_v, c=1) or p(s_a|x_a, x_v, c=2). Both are gaussians. 
                alpha = torch.tensor(1 - self.params.rng, device=self.device)
                normal = torch.distributions.Normal(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
                z = normal.icdf(1 - alpha / 2)  # Inverse CDF (equivalent to norm.ppf)
                resp[:, 1] = self.params.a * 2 * z * sig_hat_MS + self.params.b

            else:
                # Complex interval computation - keeping original logic but with PyTorch
                center_cdf = post_cdf[bin_est, trial_indices]
                
                # The cap trials will have bigger intervals
                cap_trial_l = (center_cdf < self.params.rng/2) | (bin_est == 0)
                cap_trial_r = (center_cdf > 1-self.params.rng/2) | (bin_est == nbin - 1)
                
                non_cap_trial = ~(cap_trial_l | cap_trial_r)
                
                # Init array to save the best bin size (to cover rng%)
                best_bin_n = torch.ones(ntrial, device=self.device, dtype=torch.long)
                
                # Main loop for non-capped trials
                while True:
                    half_width = ((best_bin_n - 1) // 2)
                    start_bin = torch.clamp(bin_est - half_width - 1, 0, nbin - 1)
                    end_bin = torch.clamp(bin_est + half_width, 0, nbin - 1)
                    prob_rng = post_cdf[end_bin, trial_indices] - post_cdf[start_bin, trial_indices]
            
                    need_more_bins = (prob_rng < self.params.rng) & (~cap_trial_l)
                    if not torch.any(need_more_bins):
                        break
                    best_bin_n[need_more_bins] += 2
                    
                resp[~cap_trial_l, 1] = self.params.a * binSize * (end_bin[~cap_trial_l] - start_bin[~cap_trial_l] + 1).float() + self.params.b
                
                # Compute separately for cap_trial_l 
                best_bin_n = torch.ones(ntrial, device=self.device, dtype=torch.long)
                
                while True:
                    half_width = ((best_bin_n - 1) // 2)
                    end_bin = torch.clamp(bin_est + half_width, 0, nbin-1)
                    prob_rng = post_cdf[end_bin, trial_indices]
                    
                    need_more_bins = (prob_rng < self.params.rng) & cap_trial_l
                    if not torch.any(need_more_bins):
                        break  
                    best_bin_n[need_more_bins] += 2
                    
                resp[cap_trial_l, 1] = self.params.a * binSize * (end_bin[cap_trial_l] + 1).float() + self.params.b
            
        elif self.config['CI_readout'] == 'BayEst':
            # Estimate-based Bayesian estimation
            if self.config['est_var'] == 'MS':
                # The distribution is either p(s_a|x_a, x_v, c=1) or p(s_a|x_a, x_v, c=2). Both are gaussians. 
                resp[:, 1] = self.params.b - self.params.a / (sig_hat_MS * torch.sqrt(torch.tensor(2 * torch.pi)))
            else:
                prob_rng = post_pdf[bin_est, trial_indices]
                resp[:, 1] = self.params.b - self.params.a * prob_rng

        elif self.config['CI_readout'] == 'Entro':
            # Entropy-based estimation (non-bayesian)
            if self.config['est_var'] == 'MS':
                # The distribution is either p(s_a|x_a, x_v, c=1) or p(s_a|x_v, c=2). Both are gaussians. 
                resp[:, 1] = self.params.b + self.params.a * torch.log(torch.sqrt(torch.tensor(2 * torch.pi * torch.e)) * sig_hat_MS)
            else:
                # Differential entropy computation
                eps = torch.finfo(torch.float32).eps
                safe_pdf = torch.where(post_pdf > 0, post_pdf, eps)
                diff_entropy = -torch.sum(post_pdf * torch.log(safe_pdf), dim=0) + torch.log(torch.tensor(binSize, device=self.device))
                resp[:, 1] = self.params.b + self.params.a * diff_entropy

        elif self.config['CI_readout'] == 'DistXdiff':
            # Distance-based estimation (non-bayesian)
            d = torch.abs(xA - xV)
            resp[:, 1] = self.params.a * d**2 * torch.exp(-d) + self.params.b
            
        elif self.config['CI_readout'] == 'DistSdiff':
            # Distance-based estimation (non-bayesian)
            d = torch.abs(s_hat_common - torch.where(Atrial, sA_hat_indep, sV_hat_indep))
            resp[:, 1] = self.params.a * d**2 * torch.exp(-d) + self.params.b

        # Add motor noise
        resp[:, 1] = torch.normal(resp[:, 1], torch.tensor(self.params.sig_rconf, device=self.device))

        # Add decision noise for causal decision and/or confidence
        if self.config['dec_noise_flag'] == 1:
            # Same noise for causal decision/confidence
            sim_post = self.add_decision_noise(sim_post, 0)
            sim_post_conf = sim_post  # same for confidence
        elif self.config['dec_noise_flag'] == 2:
            # Only noise for causal confidence
            sim_post_conf = self.add_decision_noise(sim_post, 1)
        elif self.config['dec_noise_flag'] == -1:
            # Separate noises for confidence and decision
            sim_post_conf = self.add_decision_noise(sim_post, 1)
            sim_post = self.add_decision_noise(sim_post, 0)
        else:
            sim_post_conf = sim_post

        # Read-out the causal decision/confidence 
        if self.config['Causal_readout'] == 'Bay':
            # Output: 1 - common source, 2 - separate source
            resp[:, 2] = torch.argmax(sim_post, dim=1) + 1  # sim_post should be on GPU

            # confidence
            if self.config['nConfK'] == 2:
                sim_c_map = torch.max(sim_post_conf, dim=1)[0]  # max posterior prob from both categories
                # symmetrical boundary
                resp[:, 3] = 1 / (1 + torch.exp(-sim_c_map * self.params.kC1 + self.params.mC1)).to(resp.dtype)
                # alternative: linear
                # resp[:, 3] = sim_c_map * self.params.kC1 + self.params.mC1

            elif self.config['nConfK'] == 4:
                com_trial = resp[:, 2] == 1
                # asymmetrical boundary
                resp[com_trial, 3] = 1 / (1 + torch.exp(-sim_post_conf[com_trial, 0] * self.params.kC1 + self.params.mC1)).to(resp.dtype)
                resp[~com_trial, 3] = 1 / (1 + torch.exp(-sim_post_conf[~com_trial, 1] * self.params.kC2 + self.params.mC2)).to(resp.dtype)
                # alternative: linear
                # resp[com_trial, 3] = sim_post_conf[com_trial, 0] * self.params.kC1 + self.params.mC1
                # resp[~com_trial, 3] = sim_post_conf[~com_trial, 1] * self.params.kC2 + self.params.mC2
        else:
            if self.config['Causal_readout'].lower() == 'xdiff':
                d = torch.abs(xA - xV)
            elif self.config['Causal_readout'].lower() == 'sdiff':
                # Spatial estimate difference
                s_hat_choice = torch.where(Atrial, sA_hat_indep, sV_hat_indep)
                d = torch.abs(s_hat_common - s_hat_choice)

            resp[:, 2] = (d >= self.params.epsilon).long() + 1
            com_trial = resp[:, 2] == 1

            if self.config['nConfK'] == 2:
                # symmetrical boundary
                resp[com_trial, 3] = 1 / (1 + torch.exp(d[com_trial] * self.params.kC1 - self.params.mC1)).to(resp.dtype)
                resp[~com_trial, 3] = 1 / (1 + torch.exp(-d[~com_trial] * self.params.kC1 + self.params.mC1)).to(resp.dtype)
                # alternative: linear
                # resp[com_trial, 3] = -d[com_trial] * self.params.kC1 + self.params.mC1
                # resp[~com_trial, 3] = d[~com_trial] * self.params.kC1 + self.params.mC1

            elif self.config['nConfK'] == 4:
                # asymmetrical boundary
                resp[com_trial, 3] = 1 / (1 + torch.exp(d[com_trial] * self.params.kC1 - self.params.mC1)).to(resp.dtype)
                resp[~com_trial, 3] = 1 / (1 + torch.exp(-d[~com_trial] * self.params.kC2 + self.params.mC2)).to(resp.dtype)
                # alternative: linear
                # resp[com_trial, 3] = -d[com_trial] * self.params.kC1 + self.params.mC1
                # resp[~com_trial, 3] = d[~com_trial] * self.params.kC2 + self.params.mC2

        # Confidence transform (optional: if you want to scale to 0-100, but avoid if you want to keep as probability)
        # resp[:, 3] = (resp[:, 3] - resp[:, 3].min()) / (resp[:, 3].max() - resp[:, 3].min()) * 100

        # Add random noise and clip (GPU tensor version)
        noise = torch.randn(resp.shape[0], device=self.device) * self.params.sig_rc
        resp[:, 3] = torch.clamp(torch.round((resp[:, 3] + noise) * 100), 0, 100)

        # Add motor lapse for causal decision
        # lapse_trials = torch.rand(ntrial, device=self.device) < self.params.laps
        # resp[lapse_trials, 2] = torch.randint(1, 3, (torch.sum(lapse_trials),), device=self.device)
        lapse_trials = torch.rand(ntrial, device=self.device) < self.params.laps
        resp[lapse_trials, 2] = torch.randint(1, 3, (torch.sum(lapse_trials),), device=self.device).to(dtype=resp.dtype)

        # Concatenate trial conditions and response (assuming trialConds is a tensor on device)
        predData = torch.cat((trialConds, resp), dim=1)

        if interimOutput:
            return predData, sim_post, sim_post_conf, xA, xV, s_hat_common, sV_hat_indep, sA_hat_indep, quad_common, quadV_indep, quadA_indep
        else:
            return predData.cpu().numpy()

