import numpy as np
from dataclasses import dataclass

cond = np.array([[12, 4, 2, 1], 
                 [3, 2, 5, 2], 
                 [5, 2, 56, 12], 
                 [12, 3, 2, 4], 
                 [5, 12, 3, 4]])
def getCondRep(cond, nIntSamples):
    condRep = np.repeat(cond, repeats=nIntSamples, axis=0)
    return condRep

def add_decision_noise(log_beta, sim_post):
    beta = 10**log_beta
    sim_post = np.random.gamma(sim_post * beta, 1) # add gamma noise
    sim_post /= sim_post.sum(axis=1, keepdims=True) # normalize
    return sim_post
    
trialConds = getCondRep(cond, 5)
params = {'sig_V_AV':1, 'sig_A_AV':2,'sig_P':30, 'a_A':1, 'b_A':0, 'mu_P':0, 'p_com':0.5, 'dec_noise':0.1, 'epsilon':1, 'conf_bin_k1': 0.6, 'conf_bin_k2': 0.8, \
          'conf_bin_k3': 0.6, 'conf_bin_k4': 0.7, 'alp_shift_A':0.1, 'alp_shift_V':0.01, 'sig_A_A':4, 'resp_loc':np.array([-12, -4, 4, 12]), \
              'resp_loc_len':4, 'sig_V_V':1}
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
    

    


def simAllConds(config, params, trialConds, interimOutput = False):
    '''
    Parameters
    ----------
    config : dict
        configuration for the model.
    params : dataclass
        the parameters values.
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
    ## simple parameter transformation for easier and cleanercomputation

    ntrial = trialConds.shape[0]
    resp = np.zeros((ntrial, 3), dtype=int) #init a np array to save response data
    varV = params.sig_V_AV**2
    varA = params.sig_A_AV**2
    varP = params.sig_P**2
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
    bi_sA_remap = trialConds[:, 0] * params.a_A + params.b_A
    bi_sV_remap = trialConds[:, 1]

    # sample A and V measurements from gaussion distribution
    bi_xA = bi_sA_remap + params.sig_A_AV * np.random.randn(ntrial)
    bi_xV = bi_sV_remap + params.sig_V_AV * np.random.randn(ntrial)

    # compute the posterior causal structure based on BCI model
    bi_s_hat_common = (bi_xV/varV + bi_xA/varA + params.mu_P/varP) * varVA_hat
    bi_sV_hat_indep = (bi_xV/varV + params.mu_P/varP) * varV_hat
    bi_sA_hat_indep = (bi_xA/varA + params.mu_P/varP) * varA_hat

    quad_common = (bi_xV - bi_xA)**2 * varP + (bi_xV - params.mu_P)**2 * varA + (bi_xA - params.mu_P)**2 * varV
    quadV_indep = (bi_xV - params.mu_P)**2
    quadA_indep = (bi_xA - params.mu_P)**2

    likelihood_common = np.exp(-quad_common / (2 * var_common)) / (2 * np.pi * np.sqrt(var_common))
    likelihoodV_indep = np.exp(-quadV_indep / (2 * varV_indep)) / np.sqrt(2 * np.pi * varV_indep)
    likelihoodA_indep = np.exp(-quadA_indep / (2 * varA_indep)) / np.sqrt(2 * np.pi * varA_indep)
    likelihood_indep = likelihoodV_indep * likelihoodA_indep

    post_common = likelihood_common * params.p_com
    post_indep = likelihood_indep * (1 - params.p_com)
    post_C1 = post_common / (post_common + post_indep)
    sim_post = np.column_stack((post_C1, 1 - post_C1))

    # final A, V spatial read-out for various strategies
    if config['AV_readout'] == 'MA':
        bi_sV_hat = post_C1 * bi_s_hat_common + (1 - post_C1) * bi_sV_hat_indep
        bi_sA_hat = post_C1 * bi_s_hat_common + (1 - post_C1) * bi_sA_hat_indep
    elif config['AV_readout'] == 'MS':
        bi_sV_hat = (post_C1 > 0.5) * bi_s_hat_common + (post_C1 <= 0.5) * bi_sV_hat_indep
        bi_sA_hat = (post_C1 > 0.5) * bi_s_hat_common + (post_C1 <= 0.5) * bi_sA_hat_indep
    elif config['AV_readout'] == 'PM':
        eta = np.random.rand(ntrial)
        bi_sV_hat = (post_C1 > eta) * bi_s_hat_common + (post_C1 <= eta) * bi_sV_hat_indep
        bi_sA_hat = (post_C1 > eta) * bi_s_hat_common + (post_C1 <= eta) * bi_sA_hat_indep
    elif config['AV_readout'] == 'na':
        bi_sA_hat = np.nan
        bi_sV_hat = np.nan

    # add decision noise for causal decision and/or confidence - 
    # output: sim_post used in causal decision read out, sim_post_conf for confidence read-out
    if config['dec_noise_flag'] == 1:
        # same noise for causal decision/confidence
        sim_post = add_decision_noise(params.logbeta, sim_post, 0)
        sim_post_conf = sim_post # same for confidence
        
    elif config['dec_noise_flag'] == 2:
        # only noise for causal confidence
        sim_post_conf = add_decision_noise(params.logbeta, sim_post, 1)
        
    elif config['dec_noise_flag'] == -1:
        # separate noises for confidence and decision
        sim_post_conf = add_decision_noise(params.logbeta_conf, sim_post, 1)
        sim_post = add_decision_noise(params.logbeta, sim_post, 0)
        
    else:
        sim_post_conf = sim_post

    # read-out the causal decision 
    if config['C_readout'].lower() == 'p':
        # Bayesian - sim_post
        # Output: 1 - common source, 2 - separate source
        resp[:, 0] = np.argmax(sim_post, axis=1) + 1
        
    else:
        # Non-Bayesian read-out
        if config['C_readout'].lower() == 'xdiff':
            # Measurement difference
            nb_c = np.abs(bi_xA - bi_xV)
        elif config['C_readout'].lower() == 'sdiff':
            # Spatial estimate difference
            nb_c = np.abs(bi_sA_hat - bi_sV_hat)
        # Output: 1 - common source, 2 - separate source
        resp[:, 0] = (nb_c >= params.epsilon) + 1


    # read-out the causal confidence - sim_post_conf
    if config['Conf_readout'].lower() == 'p':
        # Bayesian - sim_post_conf
        sim_c_map = np.max(sim_post_conf, axis=1) # max posterior prob from both category
        
        if config['conf_b_nr'] == 2:
            # symmetrical boundary
            k = np.sort([0.49, params.conf_bin_k1, params.conf_bin_k2, 1])
            simRespConf = np.digitize(sim_c_map, k, right=True)
            resp[:, 1] = np.clip(simRespConf, 1, 3)
            
        elif config['conf_b_nr'] == 4:
            # asymmetrical boundary
            resp_com = resp[:, 0] == 1
            k1 = np.sort([0.49, params.conf_bin_k1, params.conf_bin_k2, 1])
            simRespConfCom = np.digitize(sim_c_map[resp_com], k1, right=True)
            resp[resp_com, 1] = np.clip(simRespConfCom, 1, 3)
            
            k2 = np.sort([0.49, params.conf_bin_k3, params.conf_bin_k4, 1])
            simRespConfNoCom = np.digitize(sim_c_map[~resp_com], k2, right=True)
            resp[~resp_com, 1] = np.clip(simRespConfNoCom, 1, 3)

    else:
        # Non-Bayesian 
        if config['Conf_readout'].lower() == 'xdiff':
            nb_c = np.abs(bi_xA - bi_xV)
        elif config['Conf_readout'].lower() == 'sdiff':
            nb_c = np.abs(bi_sA_hat - bi_sV_hat)
        abs_dist = np.abs(nb_c - params.epsilon)

        if config['conf_b_nr'] == 2:
            k = np.sort([0, params.conf_bin_k1, params.conf_bin_k2, params.epsilon])
            simRespConf = np.digitize(abs_dist, k, right=True)
            resp[:, 1] = np.clip(simRespConf, 1, 3)

        elif config['conf_b_nr'] == 4:
            resp_com = resp[:, 0] == 1
            k1 = np.sort([0, params.conf_bin_k1, params.conf_bin_k2, params.epsilon])
            simRespConfCom = np.digitize(abs_dist[resp_com], k1, right=True)
            resp[resp_com, 1] = np.clip(simRespConfCom, 1, 3)

            k2 = np.sort([0, params.conf_bin_k3, params.conf_bin_k4, np.inf])
            simRespConfNoCom = np.digitize(abs_dist[~resp_com], k2, right=True)
            resp[~resp_com, 1] = np.clip(simRespConfNoCom, 1, 3)
            ##0<=k1<=k2<=epsilon and 0<=k3<=k4


    ###########################################
    ###########  recalibration  ###############
    ###########################################
    if config['shift_update'] == 'FR':
        updatedShift_A = params.alp_shift_A * (bi_xV - bi_xA)
        updatedShift_V = params.alp_shift_V * (bi_xA - bi_xV)

    elif config['shift_update'] == 'CI':
        updatedShift_A = params.alp_shift_A * (bi_sA_hat - bi_xA)
        updatedShift_V = params.alp_shift_V * (bi_sV_hat - bi_xV)

    else:
        updatedShift_A = np.zeros_like(bi_xA)
        updatedShift_V = np.zeros_like(bi_xV)


    ###########################################
    #############  uni phase  #################
    ###########################################

    ####auditory
    trialAud = trialConds[:, 3] == 1  

    uni_sA_remap = (trialConds[trialAud, 2] * params.a_A + params.b_A + updatedShift_A[trialAud])
    mu_sA_hat = (uni_sA_remap / (params.sig_A_A ** 2) + params.mu_P / varP) / (1 / varP + 1 / (params.sig_A_A ** 2))
    var_sA_hat = ((1 / varP + 1 / (params.sig_A_A ** 2)) ** 2) / (params.sig_A_A ** 2)
    uni_sA_hat = np.random.normal(mu_sA_hat, np.sqrt(var_sA_hat))

    # Find the response location closest to sV_hat and sA_hat (minimum deviation)    
    # Compute index of the closest response location - use broadcast in numpy
    tA = np.argmin(np.abs(uni_sA_hat[:, None] - params.resp_loc), axis=1)
    resp[trialAud, 2] = params.resp_loc[tA]

    ####visual
    uni_sV_remap = trialConds[~trialAud, 2] + updatedShift_V[~trialAud]
    mu_sV_hat = (uni_sV_remap / (params.sig_V_V ** 2) + params.mu_P / varP) / (1 / varP + 1 / (params.sig_V_V ** 2))
    var_sV_hat = ((1 / varP + 1 / (params.sig_V_V ** 2)) ** 2) / (params.sig_V_V ** 2)
    uni_sV_hat = np.random.normal(mu_sV_hat, np.sqrt(var_sV_hat))
    # find the closest discrete response
    tV = np.argmin(np.abs(uni_sV_hat[:, None] - params.resp_loc), axis=1)
    resp[~trialAud, 2] = params.resp_loc[tV]

    # concat the trial conditions and response
    predData = np.column_stack((trialConds, resp))
    #predData.dtype -'int32'
    
    if interimOutput:
        return predData, sim_post, sim_post_conf, updatedShift_A, bi_xA, bi_xV, bi_sA_hat, bi_sV_hat
    else:
        return predData


