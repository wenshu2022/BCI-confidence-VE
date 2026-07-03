import os
wkdir = 'M:/1confiProj/'
os.chdir(wkdir)
from modelClass import *
from utils.help_function import *
from utils.plot_function import *

model = ConfiModel(mid = 1)
model.display_parameters()

par_input = [getattr(model.params, name) for name in model.estParamsNames]
trialConds = model.getCondRep(nIntSamples=500)
start_time = time.time()
dat_sim = model.run_simulation(par_input,trialConds)
end_time = time.time()
print(f"Execution time: {end_time - start_time:.6f} seconds")  

#organize into a dataframe as the behaviour
df_sim = organize_sim_data(dat_sim)

####################################################################
### ventriloquism effect 
plot_VE(df_sim,  1)

####confidence interval as a function of spatial disparity.
plot_group_pattern(df_sim, 'CISize', 'Common')
plot_group_pattern(df_sim, 'CISize', 'Separate')
plot_group_pattern(df_sim, 'CISize', 'Total')

plot_group_pattern(df_sim, 'ConfLevel', 'Common')
plot_group_pattern(df_sim, 'ConfLevel', 'Separate')
plot_group_pattern(df_sim, 'ConfLevel', 'Total')

plot_group_pattern(df_sim, 'bias', 'Common')
plot_group_pattern(df_sim, 'bias', 'Separate')
plot_group_pattern(df_sim, 'bias', 'Total')

plotDavidFancy(df_sim, 'RespLoc', 'High')
plotDavidFancy(df_sim, 'CISize', 'High')
