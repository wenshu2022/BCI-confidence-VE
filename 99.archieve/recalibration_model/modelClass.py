import numpy as np
from scipy.optimize import minimize
import itertools
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class ModelParams:
    """Stores numerical model parameters with defaults."""
    sig_V_AV: float = 1
    sig_A_AV: float = 2
    sig_A_A: float = 4
    sig_V_V: float = 1
    sig_P: float = 30
    a_A: float = 1
    b_A: float = 0
    mu_P: float = 0
    p_com: float = 0.5
    logbeta: float = 0.1
    logbeta_conf: float = 0.1
    epsilon: float = 1
    conf_bin_k1: float = 0.6
    conf_bin_k2: float = 0.8
    conf_bin_k3: float = 0.6
    conf_bin_k4: float = 0.7
    alp_shift_A: float = 0.1
    alp_shift_V: float = 0
    resp_loc: np.ndarray = field(default_factory=lambda: np.array([-12, -4, 4, 12]))
    resp_loc_len: int = 4
    laps: float = 0.01

class RecalModel:
    def __init__(self, mid:int, params: ModelParams = ModelParams()):
        self.params = params
        self.mid = mid
        self.conditions = self.generate_conditions()
        self.config = self.load_config()
        self.estParamsNames, self.bounds = self.est_params()

    def display_parameters(self):
        for param, value in self.params.__dict__.items():
            print(f"{param}: {value}")

    def generate_conditions(self):
        """Generate all possible conditions based on resp_loc.
        each row represent a trial condition = [A in AV, V in AV, A/V in uni, A(1) or V(0) block]
        """
        combs = np.array(list(itertools.product(self.params.resp_loc, repeat=3)))
        # Create condition arrays
        cond_A = np.column_stack((combs, np.ones(combs.shape[0])))  # A(1) block
        cond_V = np.column_stack((combs, np.zeros(combs.shape[0]))) # V(0) block
        # Concatenate both conditions
        return np.vstack((cond_A, cond_V)).astype(int)
    
    def load_config(self):
        """
        Load model configuration from an Excel file and filter by model_id.
        """
        config_df = pd.read_excel('M:/1confiProj/m_config.xlsx')  # Read full config file
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
        
        if self.config['sameVar2phase'] == 1:
            self.params.sig_A_A = self.params.sig_A_AV
            self.params.sig_V_V = self.params.sig_V_AV

    def est_params(self):
        params = ['sig_V_AV', 'sig_A_AV', 'sig_P', 'p_com', 'conf_bin_k1', 'conf_bin_k2', 'alp_shift_A']
        bounds = []#bounds = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]  
        
        if self.config['bA_flag'] == 1:
            params.append('b_A')
            #bounds.append((0, 1))  
        if self.config['alp_V_flag'] == 1:
            params.append('alp_shift_V')
            #bounds.append((0, 1))  
        if self.config['nConfK'] == 4:
            params.extend(['conf_bin_k3', 'conf_bin_k4'])
            #bounds.extend([(0, 1), (0, 1)])  
        if self.config['laps_flag'] == 1:
            params.append('laps')
            #bounds.append((0, 1)) 
        if self.config['C_readout'].lower() == 'p' and self.config['dec_noise_flag'] in (1, -1):
            params.append('logbeta')
            #bounds.append((0, 1))  
        if self.config['Conf_readout'].lower() == 'p' and self.config['dec_noise_flag'] in (2, -1):
            params.append('logbeta_conf')
            #bounds.append((0, 1))  
        if self.config['C_readout'].lower() != 'p' or self.config['Conf_readout'].lower() != 'p':
            params.append('epsilon')
            #bounds.append((0, 1))  
        if self.config['sameVar2phase'] == 0:
            params.extend(['sig_A_A', 'sig_V_V'])
            #bounds.extend([(0, 1), (0, 1)])  
        
        return params, bounds

    def getCondRep(self, nIntSamples:int =10000):
        condRep = np.repeat(self.conditions, repeats=nIntSamples, axis=0)
        return condRep

    def add_decision_noise(self, sim_post, conf):
        '''help fuction to add decision noise for simulation functions'''
        if conf==0:
            beta = 10**self.params.logbeta
        elif conf==1:
            beta = 10**self.params.logbeta_conf
        sim_post = np.random.gamma(sim_post * beta, 1) # add gamma noise
        sim_post /= sim_post.sum(axis=1, keepdims=True) # normalize
        return sim_post
    
    def run_simulation(self, inputs, trialConds=None, interimOutput = False):
        '''
        Parameters
        ----------
        inputs : the value of the tuning parameters 
        trialConds : numpy array
        each row represents the condition of a trial [A in AV, V in AV, A/V in uni, A(1) or V(0) block]
        dimention - ntrial * 4
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
        ## simple parameter transformation for easier and cleanercomputation
        ntrial = trialConds.shape[0]
        resp = np.zeros((ntrial, 3), dtype=int) #init a np array to save response data
        varV = self.params.sig_V_AV**2
        varA = self.params.sig_A_AV**2
        varP = self.params.sig_P**2
        varVA_hat = 1 / (1/varV + 1/varA + 1/varP)
        varV_hat = 1 / (1/varV + 1/varP)
        varA_hat = 1 / (1/varA + 1/varP)
        var_common = varV * varA + varV * varP + varA * varP
        varV_indep = varV + varP
        varA_indep = varA + varP

        ###########################################
        #############  AV Phase  ##################
        ###########################################
        # linear transformation of the physical stimuli 
        bi_sA_remap = trialConds[:, 0] * self.params.a_A + self.params.b_A
        bi_sV_remap = trialConds[:, 1]

        # sample A and V measurements from gaussion distribution
        bi_xA = bi_sA_remap + self.params.sig_A_AV * np.random.randn(ntrial)
        bi_xV = bi_sV_remap + self.params.sig_V_AV * np.random.randn(ntrial)

        # compute the posterior causal structure based on BCI model
        bi_s_hat_common = (bi_xV/varV + bi_xA/varA + self.params.mu_P/varP) * varVA_hat
        bi_sV_hat_indep = (bi_xV/varV + self.params.mu_P/varP) * varV_hat
        bi_sA_hat_indep = (bi_xA/varA + self.params.mu_P/varP) * varA_hat

        quad_common = (bi_xV - bi_xA)**2 * varP + (bi_xV - self.params.mu_P)**2 * varA + (bi_xA - self.params.mu_P)**2 * varV
        quadV_indep = (bi_xV - self.params.mu_P)**2
        quadA_indep = (bi_xA - self.params.mu_P)**2

        likelihood_common = np.exp(-quad_common / (2 * var_common)) / (2 * np.pi * np.sqrt(var_common))
        likelihoodV_indep = np.exp(-quadV_indep / (2 * varV_indep)) / np.sqrt(2 * np.pi * varV_indep)
        likelihoodA_indep = np.exp(-quadA_indep / (2 * varA_indep)) / np.sqrt(2 * np.pi * varA_indep)
        likelihood_indep = likelihoodV_indep * likelihoodA_indep

        post_common = likelihood_common * self.params.p_com
        post_indep = likelihood_indep * (1 - self.params.p_com)
        post_C1 = post_common / (post_common + post_indep)
        sim_post = np.column_stack((post_C1, 1 - post_C1))

        # final A, V spatial read-out for various strategies
        if self.config['AV_readout'] == 'MA':
            bi_sV_hat = post_C1 * bi_s_hat_common + (1 - post_C1) * bi_sV_hat_indep
            bi_sA_hat = post_C1 * bi_s_hat_common + (1 - post_C1) * bi_sA_hat_indep
        elif self.config['AV_readout'] == 'MS':
            bi_sV_hat = (post_C1 > 0.5) * bi_s_hat_common + (post_C1 <= 0.5) * bi_sV_hat_indep
            bi_sA_hat = (post_C1 > 0.5) * bi_s_hat_common + (post_C1 <= 0.5) * bi_sA_hat_indep
        elif self.config['AV_readout'] == 'PM':
            eta = np.random.rand(ntrial)
            bi_sV_hat = (post_C1 > eta) * bi_s_hat_common + (post_C1 <= eta) * bi_sV_hat_indep
            bi_sA_hat = (post_C1 > eta) * bi_s_hat_common + (post_C1 <= eta) * bi_sA_hat_indep
        elif self.config['AV_readout'] == 'na':
            bi_sA_hat = np.nan
            bi_sV_hat = np.nan

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

        # read-out the causal decision 
        if self.config['C_readout'].lower() == 'p':
            # Bayesian - sim_post
            # Output: 1 - common source, 2 - separate source
            resp[:, 0] = np.argmax(sim_post, axis=1) + 1
        else:
            # Non-Bayesian read-out
            if self.config['C_readout'].lower() == 'xdiff':
                # Measurement difference
                nb_c = np.abs(bi_xA - bi_xV)
            elif self.config['C_readout'].lower() == 'sdiff':
                # Spatial estimate difference
                nb_c = np.abs(bi_sA_hat - bi_sV_hat)
            # Output: 1 - common source, 2 - separate source
            resp[:, 0] = (nb_c >= self.params.epsilon) + 1

        # read-out the causal confidence - sim_post_conf
        if self.config['Conf_readout'].lower() == 'p':
            # Bayesian - sim_post_conf
            sim_c_map = np.max(sim_post_conf, axis=1) # max posterior prob from both category
            if self.config['nConfK'] == 2:
                # symmetrical boundary
                k = np.sort([0.49, self.params.conf_bin_k1, self.params.conf_bin_k2, 1])
                simRespConf = np.digitize(sim_c_map, k, right=True)
                resp[:, 1] = np.clip(simRespConf, 1, 3)
                
            elif self.config['nConfK'] == 4:
                # asymmetrical boundary
                resp_com = resp[:, 0] == 1
                k1 = np.sort([0.49, self.params.conf_bin_k1, self.params.conf_bin_k2, 1])
                simRespConfCom = np.digitize(sim_c_map[resp_com], k1, right=True)
                resp[resp_com, 1] = np.clip(simRespConfCom, 1, 3)
                
                k2 = np.sort([0.49, self.params.conf_bin_k3, self.params.conf_bin_k4, 1])
                simRespConfNoCom = np.digitize(sim_c_map[~resp_com], k2, right=True)
                resp[~resp_com, 1] = np.clip(simRespConfNoCom, 1, 3)
        else:
            # Non-Bayesian 
            if self.config['Conf_readout'].lower() == 'xdiff':
                nb_c = np.abs(bi_xA - bi_xV)
            elif self.config['Conf_readout'].lower() == 'sdiff':
                nb_c = np.abs(bi_sA_hat - bi_sV_hat)
            abs_dist = np.abs(nb_c - self.params.epsilon)

            if self.config['nConfK'] == 2:
                k = np.sort([0, self.params.conf_bin_k1, self.params.conf_bin_k2, self.params.epsilon])
                simRespConf = np.digitize(abs_dist, k, right=True)
                resp[:, 1] = np.clip(simRespConf, 1, 3)

            elif self.config['nConfK'] == 4:
                resp_com = resp[:, 0] == 1
                k1 = np.sort([0, self.params.conf_bin_k1, self.params.conf_bin_k2, self.params.epsilon])
                simRespConfCom = np.digitize(abs_dist[resp_com], k1, right=True)
                resp[resp_com, 1] = np.clip(simRespConfCom, 1, 3)

                k2 = np.sort([0, self.params.conf_bin_k3, self.params.conf_bin_k4, np.inf])
                simRespConfNoCom = np.digitize(abs_dist[~resp_com], k2, right=True)
                resp[~resp_com, 1] = np.clip(simRespConfNoCom, 1, 3)
                ##0<=k1<=k2<=epsilon and 0<=k3<=k4

        ###########################################
        ###########  recalibration  ###############
        ###########################################
        if self.config['shift_update'] == 'FR':
            updatedShift_A = self.params.alp_shift_A * (bi_xV - bi_xA)
            updatedShift_V = self.params.alp_shift_V * (bi_xA - bi_xV)

        elif self.config['shift_update'] == 'CI':
            updatedShift_A = self.params.alp_shift_A * (bi_sA_hat - bi_xA)
            updatedShift_V = self.params.alp_shift_V * (bi_sV_hat - bi_xV)

        else:
            updatedShift_A = np.zeros_like(bi_xA)
            updatedShift_V = np.zeros_like(bi_xV)

        ###########################################
        #############  uni phase  #################
        ###########################################
        ####auditory
        trialAud = trialConds[:, 3] == 1  
        uni_sA_remap = (trialConds[trialAud, 2] * self.params.a_A + self.params.b_A + updatedShift_A[trialAud])
        mu_sA_hat = (uni_sA_remap / (self.params.sig_A_A ** 2) + self.params.mu_P / varP) / (1 / varP + 1 / (self.params.sig_A_A ** 2))
        var_sA_hat = ((1 / varP + 1 / (self.params.sig_A_A ** 2)) ** 2) / (self.params.sig_A_A ** 2)
        uni_sA_hat = np.random.normal(mu_sA_hat, np.sqrt(var_sA_hat))

        # Find the response location closest to sV_hat and sA_hat (minimum deviation)    
        # Compute index of the closest response location - use broadcast in numpy
        tA = np.argmin(np.abs(uni_sA_hat[:, None] - self.params.resp_loc), axis=1)
        resp[trialAud, 2] = self.params.resp_loc[tA]

        ####visual
        uni_sV_remap = trialConds[~trialAud, 2] + updatedShift_V[~trialAud]
        mu_sV_hat = (uni_sV_remap / (self.params.sig_V_V ** 2) + self.params.mu_P / varP) / (1 / varP + 1 / (self.params.sig_V_V ** 2))
        var_sV_hat = ((1 / varP + 1 / (self.params.sig_V_V ** 2)) ** 2) / (self.params.sig_V_V ** 2)
        uni_sV_hat = np.random.normal(mu_sV_hat, np.sqrt(var_sV_hat))

        # find the closest discrete response
        tV = np.argmin(np.abs(uni_sV_hat[:, None] - self.params.resp_loc), axis=1)
        resp[~trialAud, 2] = self.params.resp_loc[tV]

        # concat the trial conditions and response
        predData = np.column_stack((trialConds, resp))
        #predData.dtype -'int32'
        if interimOutput:
            return predData, sim_post, sim_post_conf, updatedShift_A, bi_xA, bi_xV, bi_sA_hat, bi_sV_hat
        else:
            return predData
    
    def get_condition_data(self, data, condition):
        """Helper function to extract rows from data matching a condition."""
        cond_mask = np.all(data[:, :4] == condition, axis=1)
        return data[cond_mask, 4:]
    
    def count_data(self, data, ncat, nr, ns):
        # Map spatial locations into indices 1,2,3,4
        spatial_mapped = np.searchsorted(self.params.resp_loc, data[:, 2])
        multi_indices = np.ravel_multi_index((data[:, 0]-1, data[:, 1]-1, spatial_mapped), (ncat, nr, ns))
        # count instances
        CMat = np.bincount(multi_indices, minlength=ncat * nr * ns)
        return CMat.reshape((ncat, nr, ns))
    
    def computeNLL(self, behData, inputs_param, nIntSamples:int = 10000):
        # prediction data
        trialConds = model.getCondRep(nIntSamples)
        predData = self.run_simulation(inputs=inputs_param, trialConds=trialConds)

        # Levels of decisions
        ncat = 2  # Level of causal decisions
        nr = 3    # Level of confidence
        ns = 4    # Level of spatial locations
        NLL = 0  # Initialize NLL

        for cidx in range(self.conditions.shape[0]):
            # get data for this condition
            c_ = self.conditions[cidx, :]
            pred_resp = self.get_condition_data(predData, c_)
            beh_resp = self.get_condition_data(behData, c_)
            
            ##count predicted data
            temp_probs = self.count_data(pred_resp, ncat, nr, ns) / nIntSamples
            # Apply lapse rate correction
            pMat = (1 - self.params.laps) * temp_probs + self.params.laps / (ncat * nr * ns)
            # Ensure non-zero probabilities and normalize
            pMat += np.finfo(float).eps  # Add small value to avoid log(0)
            pMat /= pMat.sum()
            
            # Compute beh data counts
            cntMat = self.count_data(beh_resp, ncat, nr, ns)
            
            # Compute NLL
            NLL += -np.sum(cntMat * np.log(pMat))

        return NLL



  
# mat_rand_number = pd.read_csv('M:/1confiProj/codeCompare/rand_number.csv', header = None)
# trailcond = pd.read_csv('M:/1confiProj/codeCompare/trailcond.csv', header = None)
# sim_post_mat = pd.read_csv('M:/1confiProj/codeCompare/sim_post.csv', header = None)
# bi_xA_mat = pd.read_csv('M:/1confiProj/codeCompare/bi_xA.csv', header = None)
# bi_xV_mat = pd.read_csv('M:/1confiProj/codeCompare/bi_xV.csv', header = None)

model = RecalModel(mid = 70)
model.display_parameters()
model.estParamsNames
# Define test input for update_est_params
par = [2, 12, 100, 0.75, 0.55, 0.65, 0.55, 10, 0.15, 1]
x = [3, 12, 50, 0.35, 0.55, 0.65, 0.5, 10, 0.15, 1]

y = model.run_simulation(par)

#nll = model.computeNLL(y, x)
#print(nll)

# model.params.mu_P
# model.conditions
# model.config

# model.update_est_params(test_input)


#read mat data
# mat_sim = pd.read_csv('M:/1confiProj/codeCompare/mat_pred.csv', header = None)
# mat_sim2 = pd.read_csv('M:/1confiProj/codeCompare/mat_pred2.csv', header = None)
# model.computeNLL(mat_sim.values, mat_sim2.values)

# diff = mat_sim.values - predData
# abs(diff).sum()
# # np.array_equal(mat_sim, y)
# # mse = np.mean((mat_sim - y) ** 2)
# # print("MSE:", mse)
# np.allclose(mat_sim.values, predData, atol=1e-6)
# indices = np.where(np.abs(diff) >= 1e-4)
# print("Indices where difference is not 0:", indices)


# np.random.seed(42)
# py_rand = np.random.randn(100)
# np.allclose(mat_rand_number.values, py_rand, atol=1e-6)

