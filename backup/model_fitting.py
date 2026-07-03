import os
wkdir = '/home/mpla/wenlou/1confiProj/'
#wkdir = 'M:/1confiProj/'
os.chdir(wkdir)
#from utils.help_function import sub_lst, fit_data_path, fit_res_path
from models.modelClass import ConfiModel
from scripts.computeQLL import func_compute_NLL, gen_init_params, define_constraints
import numpy as np
import json
import re
import pickle
import argparse
fit_data_path = '/project/3026008.02/data_for_fitting/'
fit_res_path = '/home/mpla/wenlou/1confiProj/fitting_results/output/'

import cProfile
import pstats

# make slurm compatible
parser = argparse.ArgumentParser()
parser.add_argument("--sub_id", type=int, required=True)
parser.add_argument("--m_id", type=int, default=1)
parser.add_argument("--optimizer", type=str, default='min')
parser.add_argument("--n_runs", type=int, default=1)
parser.add_argument("--n_sims", type=int, default=100000)

args = parser.parse_args()

sub_id = args.sub_id
model = ConfiModel(mid=args.m_id)
nruns = args.n_runs
num_sim = args.n_sims
#method = args.optimizer
#clean potential special chars
method = args.optimizer.strip() # Remove leading/trailing spaces/tabs/newlines
method = re.sub(r'[^a-zA-Z0-9]', '', method) # Keep only letters and numbers

# model = ConfiModel(mid = 2)
# nruns = 2 # number of random initialization
# sub_id = 2
#beh_data = np.loadtxt(os.path.join(fit_data_path, 'beh_data_sub_' + str(sub_id) + '.csv'), delimiter=',')
bin_data = pickle.load(open(os.path.join(fit_data_path, 'beh_bin_sub_' + str(sub_id) + '.pkl'), 'rb'))
    
#num_sim = 100
objf = func_compute_NLL(model, bin_data, num_sim)
x0s = [gen_init_params(model.p_bounds, model.estParamsNames) for _ in range(nruns)]
cons = define_constraints(model.estParamsNames, method)
#model.display_parameters()
#simulation = model.run_simulation(x0s[0],  model.getCondRep(nIntSamples=1000), interimOutput = False)
#sim_noise = model.add_decision_noise(simulation[1], 0)

# also random initialization with the constraints
def run_min(objf, x0s, v=True, wnlls=True, **kwargs):
    from scipy.optimize import minimize
    import time
    '''Runs minimization routine with several starting points `x0s'.'''
    bestFun = np.inf
    bestRes = None
    all_NLL = []
    all_xs = []

    def make_callback(fun):
        def callback(xk):
            fx = fun(xk)
            print(f"Iter | x: {xk[0]:.4f} | y: {xk[1]:.4f} | f(x): {fx:.6f}")
        return callback
    
    for i, x0 in enumerate(x0s):
        print(f"\n Optimization Run {i + 1}/{len(x0s)}")
        print(f"Initial params: {np.round(x0, 4).tolist()}")
        start_time = time.time()
        res = minimize(objf, x0.tolist(), callback=make_callback(objf), **kwargs)
        end_time = time.time()
        elapsed_time = (end_time - start_time)/60

        all_NLL.append(res.fun)
        all_xs.append(res.x)
        if res.success and res.fun < bestFun:
            bestFun = res.fun
            bestRes = res
        if v:
            #print(f'<{res.fun:.2f}>', x0, '->', res.x)
            print(f"Final: NLL = {np.round(all_NLL,4)}, Params = {np.round(all_xs, 4).tolist()}")
            print(f"This run completed in {elapsed_time:.2f} minutes.")
    # is it necessary to save res??
    rrr = res if bestRes is None else bestRes
    return (rrr, all_NLL, all_xs) if wnlls else rrr

def run_bads(objf, x0s, v=True, **kwargs):
    from pybads.bads import BADS
    '''Runs minimization routine with several starting points `x0s'.'''
    all_NLL = []
    all_xs = []

    # pybads options
    options = {
        "uncertainty_handling": True,
        "max_fun_evals": 300
            }
    
    for i, x0 in enumerate(x0s):
        print(f"\n Optimization Run {i + 1}/{len(x0s)}")
        print(f"Initial params: {np.round(x0, 4).tolist()}")

        bads = BADS(objf, x0.tolist(), options=options, **kwargs)
        optimize_result = bads.optimize()
        all_NLL.append(optimize_result['fval'])
        all_xs.append(optimize_result['x'])

        if v:
            print(f"BADS minimum at: x_min = {np.round(all_xs, 4).tolist()}, NLL = {np.round(all_NLL,4)}")
            print(f"total f-count: {optimize_result['func_count']}, time: {round(optimize_result['total_time']/60, 2)} min")

    return all_NLL, all_xs

pr = cProfile.Profile()
pr.enable()

# Fit model
print(f"\n ------- Start Optimization for Subject {sub_id} ---------- ")
if method.upper()=='COBYLA':
    print("\n -- Start Optimization: method scipy.minimize COBYLA --")
    bestRes, all_NLL, all_xs = run_min(objf, x0s, bounds=model.bounds, constraints=cons, method='COBYLA')
elif method.upper()=='COBYQA':
    print("\n -- Start Optimization: method scipy.minimize COBYQA --")
    bestRes, all_NLL, all_xs = run_min(objf, x0s, bounds=model.bounds, method='COBYQA')
elif method.upper()=='POWELL':
    print("\n -- Start Optimization: method scipy.minimize Powell --")
    bestRes, all_NLL, all_xs = run_min(objf, x0s, bounds=model.bounds, method='Powell')
elif method.upper()=='BADS':
    print("\n -- Start Optimization: method BADS --")
    lower_bounds = [b[0] for b in model.bounds]
    upper_bounds = [b[1] for b in model.bounds]
    plausible_lower_bounds= [b[0] for b in model.p_bounds] 
    plausible_upper_bounds= [b[1] for b in model.p_bounds]
    all_NLL, all_xs = run_bads(objf, x0s,lower_bounds=lower_bounds, 
                                upper_bounds=upper_bounds, 
                                plausible_lower_bounds=plausible_lower_bounds, 
                                plausible_upper_bounds=plausible_upper_bounds,
                                non_box_cons=cons)

pr.disable()
p = pstats.Stats(pr)
p.strip_dirs().sort_stats(-1).print_stats()

print(f"\n ------- Finish Optimization for Subject {sub_id} ---------- ")
print("Best-fit parameters:", all_xs[np.argmin(all_NLL)])
print("Negative log-likelihood:", min(all_NLL))

# Save results
filename = f"fitted_sub_{sub_id}_m{args.m_id}_{method}.json"
filepath = os.path.join(fit_res_path, filename)

# Save results as JSON
with open(filepath, "w") as f:
    json.dump({
        "sub_id": sub_id,
        "m_id": args.m_id,
        "all_NLL": all_NLL,
        "all_xs": [x.tolist() for x in all_xs], 
        "x0s": [x.tolist() for x in x0s],
        "method":method
    }, f)

print(f"Saved results to {filepath}")
