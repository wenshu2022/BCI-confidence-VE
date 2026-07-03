# general imports
import numpy as np
from sbi.utils import MultipleIndependent
import torch
from torch.distributions import Uniform
import time
import os
import pickle
import random
import string 
import sys
print('---------import lib---------')
os.chdir("/home/mpla/wenlou/1confiProj")
from scripts.experiments import Experiment
from scripts.modelClassGPU import ConfiModel

exp_name = sys.argv[1]
m_id     = int(sys.argv[2])
sim_id   = int(sys.argv[3])   # unique for each Slurm array task
print(f"Experiment: {exp_name}, Model ID: {m_id}, Simulation ID: {sim_id}")

exp = Experiment(exp_name)
model = ConfiModel(m_id, exp.exp_config)

# setting manual seeds
my_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 1. Parameters' prior distro definition (bounds must be float!)
priors = []
for name, (low, high) in zip(model.estParamsNames, model.bounds):
    priors.append(Uniform(
        torch.tensor([low], dtype=torch.float32, device=my_device),
        torch.tensor([high], dtype=torch.float32, device=my_device)
    ))
    print(f"Prior for {name}: Uniform({low}, {high})")
prior = MultipleIndependent(priors, validate_args=False)
# prior = MultipleIndependent([Uniform(torch.tensor([1e-3], device=my_device),
#                                     torch.tensor([5.], device=my_device)),  # sig_Vrh
#                             Uniform(torch.tensor([1e-3], device=my_device),
#                                     torch.tensor([20.], device=my_device)),  # sig_Vrl 
#                             Uniform(torch.tensor([1e-2], device=my_device),
#                                     torch.tensor([20.], device=my_device)),  # sig_A
#                             Uniform(torch.tensor([1e-8], device=my_device),
#                                     torch.tensor([30.], device=my_device)),  # kC1
#                             Uniform(torch.tensor([-10.], device=my_device),
#                                     torch.tensor([10.], device=my_device)),  # mC1
#                             Uniform(torch.tensor([1.], device=my_device),
#                                     torch.tensor([50.], device=my_device)),  # sig_P
#                             Uniform(torch.tensor([1e-4], device=my_device),
#                                     torch.tensor([1-1e-4], device=my_device)),  # p_com
#                             Uniform(torch.tensor([1e-8], device=my_device),
#                                     torch.tensor([0.8], device=my_device)),    # rng
#                             Uniform(torch.tensor([1e-8], device=my_device),   #sig_rs
#                                     torch.tensor([0.8], device=my_device)),        
#                             Uniform(torch.tensor([1e-8], device=my_device),   #sig_rconf
#                                     torch.tensor([0.8], device=my_device)),   
#                             Uniform(torch.tensor([1e-8], device=my_device),   #sig_rc
#                                     torch.tensor([0.8], device=my_device)),                                       
#                                     ],  
#                             validate_args=False)

print('---------prior defined---------')
n_rep = 100

print('---------model defined---------')
trialcond = model.getCondRep(n_rep)
num_simulations = trialcond.shape[0]
theta_all = prior.sample((num_simulations,))
print('---------theta defined---------')
# Prepare data:
# rub simulation for all the parameters and get three columns for tensor
# emppty tensor to store all the simulations
all_sim_dat = np.zeros((num_simulations, 8))

for i in range(num_simulations):
    cond = trialcond[i, :].unsqueeze(0)
    sim_dat = model.run_simulation(theta_all[i, :], cond)
    all_sim_dat[i, :] = sim_dat
    if (i+1) % 1000 == 0 or (i+1) == num_simulations:
        print(f"[{time.strftime('%H:%M:%S')}] Completed {i}/{num_simulations} simulations", flush=True)

all_sim_dat = torch.tensor(all_sim_dat, dtype=torch.float32)

all_sim_dat = all_sim_dat[:, list(range(4)) + [4, 5, 7, 6]]
all_sim_dat[:, -1] = 2-all_sim_dat[:, -1].to(torch.long)
all_sim_dat[:, -2] = all_sim_dat[:, -2]/100.0

#reorg_data
# two tips for using the sbi. the choice columns must be the last column and must be long type (discrete values takes from 0).
random_ending = ''.join(random.sample(string.ascii_letters,10))
sim_data_path = os.path.join('/project/3026008.02/MNLE/', exp_name, 'model_' + str(m_id), 'sim_data')
os.makedirs(sim_data_path, exist_ok=True)
save_file_name = os.path.join(sim_data_path, f"{exp_name}_model_{m_id}_{random_ending}.pkl")

with open(save_file_name, "wb") as handle:
    pickle.dump((all_sim_dat, theta_all, prior), handle)

