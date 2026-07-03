import os
from scripts.modelClassGPU import ConfiModel
from scripts.experiments import Experiment
from scripts.run_optimizers import define_constraints, gen_init_params, run_differential_evolution, run_bads, run_min
from scripts.computeQLL import func_compute_NLL
#import numpy as np
import sys
import random
import string 
import functools
# for debugging
# import cProfile
# import pstats
wkdir = '/home/mpla/wenlou/1confiProj/'
#wkdir = 'M:/1confiProj/'
os.chdir(wkdir)

arguments = sys.argv[1:]
print(arguments)
# load info from the experiment
experiment_name = arguments[0] 
m_id = int(arguments[1])
subj_setup = arguments[2]
optimizer = arguments[3]
nruns = int(arguments[4])

print = functools.partial(print, flush=True)

if subj_setup == 'all': 
    sub_lst = [2, 3, 5, 6 ,11 ,15 ,20, 25]
else: 
    sub_lst = [int(x) for x in subj_setup.split(',')]

experiment = Experiment(experiment_name)
experiment.make_exp_dir(m_id = m_id, sub_lst=sub_lst)

# model = ConfiModel(m_id, experiment.exp_config)
# #num_sim = experiment.exp_config['num_sim']
# bayes = model.model_config['Causal_readout']=='Bay'
n_fold = experiment.exp_config['nfold_CV']

if experiment.exp_config['split_data_f'] ==1:
    vrel_lst = [1,2]
else:
    vrel_lst = [0]  # dummy value for no split

for sub_id in sub_lst:
    print(f"\n ------- Start Optimization for Subject {sub_id} ---------- ")
    for i_fold in range(n_fold):
        for vrel in vrel_lst:
            print(f"\n Cross Validation fold {i_fold}/{n_fold}")
            model = ConfiModel(m_id = m_id, exp_config = experiment.exp_config, vrel=vrel)
            #num_sim = experiment.exp_config['num_sim']
            bayes = model.model_config['Causal_readout']=='Bay'

            ## load data for this experiment and subject
            train_data, _ = experiment.load_data_for_fitting(sub_id=sub_id, i_fold=i_fold,nll_method=experiment.exp_config['NLL_method'], vrel=vrel)

            #define objective function
            objf = experiment.get_objf(train_data, model, experiment.exp_config['NLL_method'])
            x0s = [gen_init_params(model.p_bounds, model.estParamsNames, bayes, experiment.exp_config['split_data_f']) for _ in range(nruns)]
            if experiment.exp_config['split_data_f'] ==1:
                cons=None
            else:
                cons = define_constraints(model.estParamsNames, optimizer, bayes)

            # define the save file name
            sub_path = os.path.join(experiment.data_handler.fit_res_path, experiment.exp_name, 'partial', 'model_' + str(m_id), 'subject_' + str(sub_id))
            random_ending = ''.join(random.sample(string.ascii_letters,6))
            save_file_name = os.path.join(sub_path, f"fitted_sub_{sub_id}_m{m_id}_{optimizer}_f{i_fold}_vrel{vrel}_{random_ending}.pkl")

            # save the estimated parameters name
            with open(os.path.join(experiment.data_handler.fit_res_path, experiment.exp_name, 'partial', 'model_' + str(m_id), 'est_param_names.txt'), 'w') as f:
                for item in model.estParamsNames:
                    f.write("%s\n" % item)

            # pr = cProfile.Profile()
            # pr.enable()
            if optimizer.upper()=='COBYLA':
                print("\n -- Start Optimization: method scipy.minimize COBYLA --")
                run_min(m_id, sub_id, i_fold, save_file_name, objf, x0s, bounds=model.bounds, constraints=cons, method='COBYLA')
            elif optimizer.upper()=='DE':
                print("\n -- Start Optimization: method scipy.differential_evolution --")
                run_differential_evolution(m_id, sub_id, i_fold, save_file_name, objf, x0s, constraints=cons, bounds=model.bounds)
            elif optimizer.upper()=='BADS':
                print("\n -- Start Optimization: method BADS --")
                lower_bounds = [b[0] for b in model.bounds]
                upper_bounds = [b[1] for b in model.bounds]
                plausible_lower_bounds= [b[0] for b in model.p_bounds] 
                plausible_upper_bounds= [b[1] for b in model.p_bounds]

                run_bads(m_id, sub_id, i_fold, save_file_name, objf, x0s,lower_bounds=lower_bounds, 
                                            upper_bounds=upper_bounds, 
                                            plausible_lower_bounds=plausible_lower_bounds, 
                                            plausible_upper_bounds=plausible_upper_bounds,
                                            non_box_cons=cons)
            elif optimizer.upper()=='VBMC':
                objf = func_compute_NLL(model, train_data, experiment.exp_config['num_sim'])
                
            # pr.disable()
            # p = pstats.Stats(pr)
            # p.strip_dirs().sort_stats(-1).print_stats()
    print(f"\n ------- Finish Optimization for Subject {sub_id} ---------- ")




