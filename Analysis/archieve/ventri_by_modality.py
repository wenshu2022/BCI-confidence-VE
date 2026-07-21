import os
work_path = "M:/1confiProj/Analysis"
os.chdir(work_path)
#the help fucntion (for plotting) import all the necessary librarys and define plot parameters. 
from help_function import *

#df = pd.read_csv(os.path.join(data_path, 'bimodalAllSub.csv'))

#plot the ventriloquism effects
def plotVentri(df, trialType, plot_f, name_label, save = 0):
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
    if save:
        df.to_csv(os.path.join(data_path, 'Block' + trialType + 'CMB.csv'), index = False)
    
    fig, ax = plt.subplots()
    if plot_f==1:
        # first avergae within the subjects
        df_plt = df.groupby(['sub_id', 'delta_VA', 'VisRelLabel'])['bias'].mean().reset_index()
        sns.lineplot(df_plt, x = 'delta_VA', y = 'bias', hue = 'VisRelLabel', ax = ax, errorbar = 'se', err_style='band')
        label = 'Raw CMB'    
    elif plot_f==2:
        df_plt = df.groupby(['sub_id', 'abs_delta_VA', 'VisRelLabel'])['flipBias'].mean().reset_index()
        sns.lineplot(df_plt, x = 'abs_delta_VA', y = 'flipBias', hue = 'VisRelLabel', ax = ax, errorbar = 'se', err_style='band')
        label = 'Flipped CMB'    

    clean_axs(ax,fontsize=16)
    ax.set_xlabel("Spatial disparity (V - A, visual angle \u00B0)", fontsize=fontsize)
    ax.set_ylabel(label, fontsize=fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.legend(title = '', frameon= False)
    ax.set_title(title)
    plt.grid(color='b', linestyle=':', linewidth=0.4)
    #plt.savefig(os.path.join('./output', 'VentriBlock' + trialType + label.replace(' ', '') + name_label + '.png'), dpi = 300)
    
    plt.show()    
    
    
    
#plotVentri(df, 'V', 2,'beh')
