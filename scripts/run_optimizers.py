import numpy as np
import os
import pandas as pd
from datetime import datetime
import cProfile
import warnings
import pstats
import time
from scipy.optimize import NonlinearConstraint
import functools
print = functools.partial(print, flush=True)


###to -do
## for discrete add the boundary restriction
## add DE constraints

def save_model_fit_result(
    model_id: int,
    sub_id: int,
    nfold:int,
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
        "nfold": nfold,
        "train_nll": nll,
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


# also random initialization with the constraints
def run_min(m_id, sub_id, nfold, save_file_name, objf,  x0s, **kwargs):
    from scipy.optimize import minimize
    options={'tol': 1e-6, 'maxiter': 10000, 'disp': True}

    '''Runs minimization routine with several starting points `x0s'.'''
    def make_callback(fun):
        def callback(xk):
            fx = fun(xk)
            x_str = np.array2string(xk, precision=4, suppress_small=True)
            print(f"Iter | x: {x_str} | f(x): {fx:.6f}")
        return callback
    
    for i, x0 in enumerate(x0s):
        print(f"\n Optimization Run {i + 1}/{len(x0s)}")
        print(f"Initial params: {np.round(x0, 4).tolist()}")
        start_time = time.time()
        res = minimize(objf, x0.tolist(), callback=make_callback(objf), options = options, **kwargs)
        end_time = time.time()
        elapsed_time = (end_time - start_time)/60

        # save the result
        save_model_fit_result(
            model_id=m_id,
            sub_id=sub_id,
            nfold=nfold,
            est_params=res.x.tolist(),
            init_params=x0.tolist(),
            nll=res.fun,
            runtime=elapsed_time,
            result_file=save_file_name,
            success=res.success
        )

        print(f"Final: training NLL = {np.round(res.fun,4)}, Params = {np.round(res.x.tolist(), 4)}, success = {res.success}")
        print(f"This run completed in {elapsed_time:.2f} minutes.")

def run_bads(m_id, sub_id, nfold, save_file_name, objf, x0s, **kwargs):
    from pybads.bads import BADS
    '''Runs minimization routine with several starting points `x0s'.'''
    # pybads options
    options = {
        "uncertainty_handling": True,
        "tol_mesh": 0.001,
        #"max_fun_evals": 1000,
        "noise_final_samples": 100,
        'poll_method': 'mads2n',  # More robust polling
            }

    for i, x0 in enumerate(x0s):
        print(f"\n Optimization Run {i + 1}/{len(x0s)}")
        print(f"Initial params: {np.round(x0, 4).tolist()}") # optionally print the val at x0

        bads = BADS(objf, x0.tolist(), options=options, **kwargs)
        optimize_result = bads.optimize()

        # save the result
        save_model_fit_result(
            model_id=m_id,
            sub_id=sub_id,
            nfold=nfold,
            est_params=optimize_result['x'],
            init_params=x0.tolist(),
            nll=optimize_result['fval'],
            runtime=optimize_result['total_time']/60,
            result_file=save_file_name,
            success=optimize_result['success']
        )

        print(f"BADS minimum at: x_min = {np.round(optimize_result['x'], 4)}, training NLL = {np.round(optimize_result['fval'],4)}")
        print(f"total f-count: {optimize_result['func_count']}, time: {round(optimize_result['total_time']/60, 2)} min")


def run_differential_evolution(m_id, sub_id, nfold, save_file_name, objf, x0s, **kwargs):
    from scipy.optimize import differential_evolution

    # def make_callback(fun):
    #     def callback(xk):
    #         fx = fun(xk)
    #         x_str = np.array2string(xk, precision=4, suppress_small=True)
    #         print(f"Iter | x: {x_str} | f(x): {fx:.6f}", flush=True)
    #     return callback
    def log_callback(xk, convergence):
        print(f"Current best: {xk}, convergence={convergence}", flush=True)

    for i, x0 in enumerate(x0s):
        print(f"\n Optimization Run {i + 1}/{len(x0s)}")
        print(f"Initial params: {np.round(x0, 4).tolist()}") # optionally print the val at x0
        start_time = time.time()
        res = differential_evolution(objf, disp=True, 
                                    #seed=42,
                                    strategy='best1bin',      # Good for multimodal problems
                                    maxiter=300,             # Increase for thorough search
                                    #popsize=15,              # 15 * 4D = 60 population size
                                    #mutation=(0.5, 1.0),     # Higher mutation for rough landscapes
                                    #recombination=0.7,       # Standard crossover rate
                                    # Convergence and robustness
                                    #seed=42 + run,           # Different seed each run
                                    atol=1e-6,
                                    tol=0.001,
                                    workers=1,
                                    x0=x0.tolist(),
                                    callback=log_callback,
                                    **kwargs)
        
        end_time = time.time()
        elapsed_time = (end_time - start_time)/60

        # save the result
        save_model_fit_result(
            model_id=m_id,
            sub_id=sub_id,
            nfold=nfold,
            est_params=res.x.tolist(),
            init_params=x0.tolist(),
            nll=res.fun,
            runtime=elapsed_time,
            result_file=save_file_name,
            success=res.success
        )

        print(f"Final: training NLL = {np.round(res.fun,4)}, Params = {np.round(res.x.tolist(), 4)}, success = {res.success}")
        print(f"This run completed in {elapsed_time:.2f} minutes.")



def run_vbmc(m_id, sub_id, nfold, save_file_name, objf, x0s, **kwargs):
    from pyvbmc import VBMC
    '''Runs minimization routine with several starting points `x0s'.'''
    options = {
        "uncertainty_handling": True,
        "tol_mesh": 0.001,
        #"max_fun_evals": 1000,
        "noise_final_samples": 100,
            }
    
    for i, x0 in enumerate(x0s):
        print(f"\n Optimization Run {i + 1}/{len(x0s)}")
        print(f"Initial params: {np.round(x0, 4).tolist()}")

        vc = VBMC(objf, x0.tolist(), options=options, **kwargs)
        optimize_result = vc.optimize()

        # save the result
        save_model_fit_result(
            model_id=m_id,
            sub_id=sub_id,
            nfold=nfold,
            est_params=optimize_result['x'],
            init_params=x0.tolist(),
            nll=optimize_result['fval'],
            runtime=optimize_result['total_time']/60,
            result_file=save_file_name,
            success=optimize_result['success']
        )

        print(f"VBMC minimum at: x_min = {np.round(optimize_result['x'], 4)}, training NLL = {np.round(optimize_result['fval'],4)}")
        print(f"total f-count: {optimize_result['func_count']}, time: {round(optimize_result['total_time']/60, 2)} min")


def gen_init_params(bounds, estParamsNames, bayes=True, split_data=0):
    ''' generate x0 for optimization'''
    x0s = np.array([np.random.uniform(low, high) for (low, high) in bounds])
    if split_data!=1:
        # number 1 : the visual high realibility variance smaller visual low realibility 
        sig_Vrh_idx = estParamsNames.index('sig_Vrh')
        sig_Vrl_idx = estParamsNames.index('sig_Vrl')
        x0s[sig_Vrl_idx] = np.random.uniform(max(x0s[sig_Vrh_idx], bounds[sig_Vrl_idx][0]), bounds[sig_Vrl_idx][1])

        # number 2 : spatial prior should have the biggest variance of all (only for bayes)
        if  bayes:
            sig_A_idx = estParamsNames.index('sig_A')
            if 'sig_P_h' in estParamsNames:
                sig_P_h_idx = estParamsNames.index('sig_P_h')
                sig_P_l_idx = estParamsNames.index('sig_P_l')
                x0s[sig_P_h_idx] = np.random.uniform(max(x0s[sig_Vrh_idx],x0s[sig_A_idx], bounds[sig_P_h_idx][0]), bounds[sig_P_h_idx][1])
                x0s[sig_P_l_idx] = np.random.uniform(max(x0s[sig_Vrl_idx],x0s[sig_A_idx], bounds[sig_P_h_idx][0]), bounds[sig_P_h_idx][1])
            else:
                sig_P_idx = estParamsNames.index('sig_P')
                x0s[sig_P_idx] = np.random.uniform(max(x0s[sig_Vrl_idx],x0s[sig_Vrh_idx],x0s[sig_A_idx], bounds[sig_P_idx][0]), bounds[sig_P_idx][1])

    return x0s

    
def define_constraints(estParamsNames, method, bayes=True):
     '''
     # constraints
     # number 1 : the visual high realibility variance smaller visual low realibility 
     # number 2 : spatial prior should have the biggest variance of all

     Parameters
     ----------
     estParamsNames : list of str
         from model class.

     Returns
     -------
     cons 

     '''

     sig_Vrh_idx = estParamsNames.index('sig_Vrh')
     sig_Vrl_idx = estParamsNames.index('sig_Vrl')

     if not bayes:
        if method.upper()=='BADS':
            # defines constraints violation
            def nonbox_constraints_bads(x):
                x_1 = np.atleast_2d(x)
                cond1 = x_1[:,sig_Vrl_idx] < x_1[:,sig_Vrh_idx] 
                return cond1 
            return nonbox_constraints_bads
        
        elif method.upper()=='COBYLA' or method.upper()=='DE':
            return [
                {'type': 'ineq', 'fun': lambda x: x[sig_Vrl_idx] - x[sig_Vrh_idx]},
                ]
        else:
            return None
        
     else:
        sig_A_idx = estParamsNames.index('sig_A')
        if 'sig_P_h' in estParamsNames:
                sig_P_h_idx = estParamsNames.index('sig_P_h')
                sig_P_l_idx = estParamsNames.index('sig_P_l')
                if method.upper()=='BADS':
                    # defines constraints violation
                    def nonbox_constraints_bads(x):
                        x_1 = np.atleast_2d(x)
                        cond1 = x_1[:,sig_Vrl_idx] < x_1[:,sig_Vrh_idx] 
                        cond2 = x_1[:,sig_P_h_idx] < x_1[:,sig_Vrh_idx]
                        cond3 = x_1[:,sig_P_l_idx] < x_1[:,sig_Vrl_idx]
                        cond4 = x_1[:,sig_P_h_idx] < x_1[:,sig_A_idx]
                        cond5 = x_1[:,sig_P_l_idx] < x_1[:,sig_A_idx]
                        return cond1 | cond2 | cond3 | cond4 | cond5
                    return nonbox_constraints_bads
                
                elif method.upper()=='COBYLA' or method.upper()=='DE':        
                    def constraints_fun(x, sig_Vrl_idx, sig_Vrh_idx, sig_P_h_idx, sig_P_l_idx, sig_A_idx):
                        return np.array([
                            x[sig_Vrl_idx] - x[sig_Vrh_idx],
                            x[sig_P_h_idx]   - x[sig_Vrh_idx],
                            x[sig_P_l_idx]   - x[sig_Vrl_idx],
                            x[sig_P_h_idx]   - x[sig_A_idx],
                            x[sig_P_l_idx]   - x[sig_A_idx],
                        ])
                    nlc = NonlinearConstraint(
                        lambda x: constraints_fun(x, sig_Vrl_idx, sig_Vrh_idx, sig_P_h_idx, sig_P_l_idx, sig_A_idx),
                        lb=0.0,        # all >= 0
                        ub=np.inf      # no upper bound
                    )
                    return nlc
                else:
                    return None
        else:
            sig_P_idx = estParamsNames.index('sig_P')
            if method.upper()=='BADS':
                # defines constraints violation
                def nonbox_constraints_bads(x):
                    x_1 = np.atleast_2d(x)
                    cond1 = x_1[:,sig_Vrl_idx] < x_1[:,sig_Vrh_idx] 
                    cond2 = x_1[:,sig_P_idx] < x_1[:,sig_Vrh_idx]
                    cond3 = x_1[:,sig_P_idx] < x_1[:,sig_Vrl_idx]
                    cond4 = x_1[:,sig_P_idx] < x_1[:,sig_A_idx]
                    return cond1 | cond2 | cond3 | cond4
                return nonbox_constraints_bads
            
            elif method.upper()=='COBYLA' or method.upper()=='DE':        
                def constraints_fun(x, sig_Vrl_idx, sig_Vrh_idx, sig_P_idx, sig_A_idx):
                    return np.array([
                        x[sig_Vrl_idx] - x[sig_Vrh_idx],
                        x[sig_P_idx]   - x[sig_Vrh_idx],
                        x[sig_P_idx]   - x[sig_Vrl_idx],
                        x[sig_P_idx]   - x[sig_A_idx],
                    ])
                nlc = NonlinearConstraint(
                    lambda x: constraints_fun(x, sig_Vrl_idx, sig_Vrh_idx, sig_P_idx, sig_A_idx),
                    lb=0.0,        # all >= 0
                    ub=np.inf      # no upper bound
                )
                return nlc
            else:
                return None



def timed_objf(x, objf):
    start = time.time()
    val = objf(x)
    end = time.time()
    print(f"Iteration took {end - start:.4f} seconds, x = {x}, f(x) = {val}")
    return val