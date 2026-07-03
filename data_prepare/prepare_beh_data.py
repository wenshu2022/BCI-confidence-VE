import os
work_path = "M:/1confiProj/"
os.chdir(work_path)
#the help fucntion (for plotting) import all the necessary librarys and define plot parameters. 
from utils.help_function import *

##part I
### compute spatial bias for plotting and anaylsising
df = pd.read_csv(os.path.join(all_data_path, 'bimodalAllSub.csv'))
trialType='A' # 'V'

df['delta_VA'] = df['VisLoc'] - df['AudLoc']
df.loc[ df.VisRel==1, 'VisRelLabel'] = 'High visual reliability'
df.loc[ df.VisRel==2, 'VisRelLabel'] = 'Low visual reliability'

if trialType == 'A':
    df = df[df.blockType==1] # 1 for auditory, 0 - for visual.
    df['bias'] = df['RespLoc'] - df['AudLoc']
    title = 'Auditory reports'
else:
    df = df[df.blockType==0]
    df['bias'] = df['RespLoc'] - df['VisLoc']
    title = 'Visual reports'

#flip the bias
df['flipBias'] = df['bias'].copy()
df.loc[df.delta_VA<0, 'flipBias'] = -df.loc[df.delta_VA<0, 'flipBias']
df['abs_delta_VA'] = abs(df.delta_VA)

df.to_csv(os.path.join(all_data_path, 'Block' + trialType + 'CMB.csv'), index = False)

##part II prepare data for fitting
### prepare behaviour data for fitting - separate file for each subject and only keeps the necessary informations
# Load behavioral data CSV
df_beh_allsub = pd.read_csv(os.path.join(all_data_path,"bimodalAllSub.csv"))
# make to csv 
#[A , V, V reliability 1 - high, 2 - low, A(1) or V(0) block, resp_loc, resp_ci, causal decision, causal confidence]
for sid in sub_lst:
    df_beh_sub = df_beh_allsub.loc[df_beh_allsub.sub_id==sid, ['AudLoc', 'VisLoc', 'VisRel', 'blockType', 'RespLoc', 'CISize', \
           'Sources', 'ConfLevel']]
    df_beh_sub.to_csv(os.path.join(fit_data_path, 'beh_data_sub_' + str(sid) + '.csv'), header=False, index=False)
