import os
wkdir = 'M:/1confiProj/'
os.chdir(wkdir)
import numpy as np
import torch
import random
# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from models.modelClass import ConfiModel
from models.modelClassGPU import ConfiModelGPU
import seaborn as sns
import matplotlib.pyplot as plt

# mid = 1
# model_cpu = ConfiModel(mid = mid)
# model_gpu = ConfiModelGPU(mid = mid)

# model_cpu.display_parameters()
# par_input = [getattr(model_cpu.params, name) for name in model_cpu.estParamsNames]
# trialConds = model_cpu.getCondRep(nIntSamples=500)

def compare_cpu_gpu(m_id):
    test = {'flag': None, 'xA': None, 'xV': None}
    model_cpu = ConfiModel(mid = m_id)
    model_gpu = ConfiModelGPU(mid = m_id)

    #model_cpu.display_parameters()
    par_input = [getattr(model_cpu.params, name) for name in model_cpu.estParamsNames]
    trialConds = model_cpu.getCondRep(nIntSamples=500)

    predDatac, _, _, xAc, xVc, _, _, _, _, _, _ = model_cpu.run_simulation(par_input,trialConds,interimOutput=True)
    test['flag'] = 1
    test['xA'] = xAc  # NumPy array
    test['xV'] = xVc  # NumPy array
    predDatag, _, _, xAg, xVg, _, _, _, _, _, _ = model_gpu.run_simulation(par_input, test, trialConds,interimOutput=True)
    
    xAg_np = xAg.numpy()
    xVg_np = xVg.numpy()
    predDatag_np = predDatag.numpy()
    print(np.allclose(xAg_np, xAc, rtol=1e-5, atol=1e-8))
    print((xAg_np == xAc).all())  # Should be True if exactly equal
    print((xVg_np == xVc).all())  # Should be True if exactly equal
    return predDatac, predDatag_np, xAc, xVc, xAg_np, xVg_np


predDatac, predDatag, xAc, xVc, xAg, xVg = compare_cpu_gpu(3)
sns.kdeplot(predDatac[:, 7], label='NumPy')
sns.kdeplot(predDatag[:, 7], label='PyTorch')
plt.legend()
plt.show()






