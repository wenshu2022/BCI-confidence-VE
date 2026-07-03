# general imports
import numpy as np
import sys
from sbi.inference import MNLE
from sbi.utils import MultipleIndependent
import torch
from torch.distributions import Uniform
import time
import os
import pickle
os.chdir("/home/mpla/wenlou/1confiProj")
from scripts.experiments import Experiment
from scripts.modelClassGPU import ConfiModel

print('---------import lib---------')
#os.chdir("/home/mpla/wenlou/1confiProj")
data_path = '/project/3026008.02/MNLE' # path to the data
# load data 
exp_name = 'MNLE-base'
m_id = 3

exp = Experiment(exp_name)
model = ConfiModel(m_id, exp.exp_config)

targetname = exp_name + '_model_' + str(m_id)
sim_agg_data_path = os.path.join(data_path, exp_name, 'model_' + str(m_id), 'agg_data')
with open(os.path.join(sim_agg_data_path, targetname + '_agg.pickle'), 'rb') as handle:
    all_sim_dat, all_theta, prior = pickle.load(handle)
num_simulations = all_sim_dat.shape[0]
print('---------data loaded---------')

# setting manual seeds
np.random.seed(0)
torch.manual_seed(0)
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

print('---------prior defined---------')

#reorg_data
# two tips for using the sbi. the choice columns must be the last column and must be long type (discrete values takes from 0).
# all_sim_dat[:,-2] = all_sim_dat[:,-2]/100.0
# all_sim_dat[:, -1] = all_sim_dat[:, -1].to(torch.long)
# input = torch.cat((all_theta, ))
time_start = time.time()
#all_sim_dat = (all_sim_dat - all_sim_dat.mean(0)) / all_sim_dat.std(0)
nan_mask = torch.sum(torch.isnan(all_sim_dat), axis=1).to(torch.bool)
trainer = MNLE(prior=prior, device="cuda")
estimator = trainer.append_simulations(all_theta[~nan_mask, :],
                                       all_sim_dat[~nan_mask, :], data_device="cpu").train(show_train_summary=True)
print('For a batch of ' + str(num_simulations) + ' simulations, it took ' + str(round(int(time.time() - time_start)/60, 3)) + ' mins')

estimator_path = os.path.join(data_path, exp_name, 'estimators')
os.makedirs(estimator_path, exist_ok=True)

with open(os.path.join(estimator_path, 'model_' + str(m_id) + '_estimator.pkl'), "wb") as handle:
    pickle.dump(estimator, handle)

