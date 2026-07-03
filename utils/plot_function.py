import os
wkdir = 'M:/1confiProj/'
os.chdir(wkdir)
from utils.help_function import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter1d
#import matplotlib.lines as mlines
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, PercentFormatter)
import matplotlib.gridspec as gridspec

sns.set_theme(style="white")
plot_path = 'M:\\1confiProj\\plots\\'
fontsize = 16
#plot the ventriloquism effects
def plot_VE(df, ax, label, bias_var = 'bias', trial_type='A'):
    """
    The function will plot across-subjects mean bias with sem error band

    Parameters
    ----------
    df : pd.DataFrame
        processed data frame with CMB bias and lables.
    trial_type : str
        'A' - auditory trials, 'V' - visual trials. Default 'A'
    raw_CMB : bolen
        1 - plot raw CMB, 0 - plot flipped CMB.
    plot_label : addition lables for the saved plots
        for exmaple, if simulated trial, add 'sim'.
    save_fig : bolen, optional
       The default is 0.
    Returns
    -------
    None.

    """
    if trial_type == 'A':
        df = df[df.blockType==1] # 1 for auditory, 0 - for visual.
        #title = 'Auditory reports'
    else:
        df = df[df.blockType==0]
        #title = 'Visual reports'

    # first average within the subjects
    # to deal with the VE bias, we need to remove the inf values, which are caused by the 0/0 division.
    df_clean = df.replace([np.inf, -np.inf], np.nan)

    df_plt = (
        df_clean
        .groupby(['sub_id', 'delta_VA', 'VisRelLabel'])[bias_var]
        .mean()
        .reset_index()
        )

    #df_plt = df.groupby(['sub_id', 'delta_VA', 'VisRelLabel'])[bias_var].mean().reset_index()
    sns.lineplot(df_plt, x = 'delta_VA', y = bias_var, linewidth=2.5, hue = 'VisRelLabel', ax = ax, errorbar = 'se', err_style='band') 

    clean_axs(ax)
    ax.set_xlabel("Spatial disparity (V - A, visual angle \u00B0)", fontsize=fontsize)
    ax.set_ylabel(label, fontsize=fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.legend(title = '', frameon= False, fontsize = fontsize-2)
    #ax.set_title(title)
    #plt.grid(color='b', linestyle=':', linewidth=0.4)
  

# def plot_VE(df, ax, palette, label, bias_var, title=None, trial_type='A'):
#     """
#     The function will plot across-subjects mean bias with sem error band

#     Parameters
#     ----------
#     df : pd.DataFrame
#         processed data frame with CMB bias and lables.
#     trial_type : str
#         'A' - auditory trials, 'V' - visual trials. Default 'A'
#     ax : matplotlib.axes.Axes
#         The axes object to plot on.
#     label : str
#         The y label for the plot.
#     bias_var : str
#         The variable name representing the bias.
#     title : str, optional
#         The title for the plot.
#     Returns
#     -------
#     None.

#     """
#     if trial_type == 'A' and title is None:
#         df = df[df.blockType==1] # 1 for auditory, 0 - for visual.
#         title = 'Auditory reports'
#     elif trial_type == 'V' and title is None:
#         df = df[df.blockType==0]
#         title = 'Visual reports'
    
#     # first avergae within the subjects
#     df_plt = df.groupby(['sub_id', 'delta_VA', 'VisRelLabel'])[bias_var].mean().reset_index()
#     sns.lineplot(df_plt, 
#                     x = 'delta_VA', 
#                     y = bias_var, 
#                     hue = 'VisRelLabel', 
#                     hue_order=['High','Low'],  
#                     palette=palette ,
#                     ax = ax, 
#                     errorbar = 'se',
#                     err_style='band')


#     clean_axs(ax)
#     ax.set_xlabel("Spatial disparity (V - A, visual angle \u00B0)", fontsize=fontsize)
#     ax.set_ylabel(label, fontsize=fontsize)
#     ax.xaxis.set_major_locator(MultipleLocator(5))
#     ax.legend(title = '', frameon= False, fontsize = fontsize-2)
#     ax.set_title(title)
#     # plt.grid(color='b', linestyle=':', linewidth=0.4)

def plot_casual_conf(df, com_label, palette, ax, trial_type=None, leg=None):
    """
    Plot Confidence interval size/ causal confidence level / spatial report bias as a function of spatial disparities

    Parameters
    ----------
    df : pd.DataFrame
        processed data frame with CMB bias and lables.
    yvar : str
        the variable for examination. CISize/ConfLevel/bias.
    com_label : str
        Common/Separate/Total.
    trial_type : str
        'A' - auditory trials, 'V' - visual trials. Default None, which means both trial types are included.
    save_fig : bolen, optional
       The default is 0.
    Returns
    -------
    None.

    """
    trial_type_map = {'A':1, 'V':0}# 1 for auditory, 0 - for visual.
    if trial_type is not None:
        df = df[df.blockType==trial_type_map[trial_type]]

    com_label_map = {'Common':1, 'Separate':2, 'Total':0}
    
    source_idx = com_label_map[com_label] # 1 - common, 2 - separate , 0 - two combined.
    # two-step avergae
    if source_idx>0:
        dfBySub = df.groupby(['sub_id', 'delta_VA', 'VisRelLabel', 'Sources'])['ConfLevel'].mean().reset_index()
        dfBySub = dfBySub.loc[dfBySub.Sources==source_idx]
    else:
        dfBySub = df.groupby(['sub_id', 'delta_VA', 'VisRelLabel'])['ConfLevel'].mean().reset_index()
        
    #_, ax = plt.subplots(figsize = (6, 5))
    sns.lineplot(data = dfBySub,
                x = 'delta_VA',
                y = 'ConfLevel',
                linewidth=2.5,
                hue = 'VisRelLabel',
                estimator = 'mean',
                errorbar='se',
                err_style='bars',
                ax = ax,
                palette=palette  
                ,legend=leg
        ) 
    clean_axs(ax)  
    ax.set_xlabel('Spatial disparity, V - A ($^\circ$)', fontsize = fontsize)
    #ax.set_ylabel('Causal confidence (0-100)', fontsize = fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    if leg is not None:
        ax.legend(title = 'Visual reliability', 
                frameon=False, 
                ncol = 2, 
                loc = 'upper right', 
                bbox_to_anchor=(0, -0.2), 
                title_fontsize = fontsize-2, 
                fontsize = fontsize-2)


###plot: CI size/conf/bias as a function of spatial disparity
def plot_group_pattern(df, yvar, com_label, palette, ax, trial_type='A', leg=None, save_fig = 0, plot_name_suffix=''):
    """
    Plot Confidence interval size/ causal confidence level / spatial report bias as a function of spatial disparities

    Parameters
    ----------
    df : pd.DataFrame
        processed data frame with CMB bias and lables.
    yvar : str
        the variable for examination. CISize/ConfLevel/bias.
    com_label : str
        Common/Separate/Total.
    trial_type : str
        'A' - auditory trials, 'V' - visual trials. Default 'A'
    save_fig : bolen, optional
       The default is 0.
    Returns
    -------
    None.

    """
    trial_type_map = {'A':1, 'V':0}# 1 for auditory, 0 - for visual.
    df = df[df.blockType==trial_type_map[trial_type]]
    # if trial_type == 'A':
    #     df = df[df.blockType==1] # 1 for auditory, 0 - for visual.
    # else:
    #     df = df[df.blockType==0]

    com_label_map = {'Common':1, 'Separate':2, 'Total':0}
    ylabel_map = {'CISize': 'Confidence interval', 'ConfLevel' : 'Causal confidence (0-100)', 'bias' : 'Auditory bias'}
    #yvar = 'bias'
    #palette = {"High": "blue", "Low": "orange"}
    #palette = {"High": "#4daf4a", "Low": "#e41a1c"}  
    #palette = {"High": "#1b9e77", "Low": "#d95f02"} 
    source_idx = com_label_map[com_label] # 1 - common, 2 - separate , 0 - two combined.
    # two-step avergae
    if source_idx>0:
        dfBySub = df.groupby(['sub_id', 'delta_VA', 'VisRelLabel', 'Sources'])[['CISize', 'ConfLevel', 'bias']].mean().reset_index()
        dfBySub = dfBySub.loc[dfBySub.Sources==source_idx]
    else:
        dfBySub = df.groupby(['sub_id', 'delta_VA', 'VisRelLabel'])[['CISize', 'ConfLevel', 'bias']].mean().reset_index()
        
    #_, ax = plt.subplots(figsize = (6, 5))
    sns.lineplot(data = dfBySub,
                x = 'delta_VA',
                y = yvar,
                linewidth=2.5,
                hue = 'VisRelLabel',
                estimator = 'mean',
                errorbar='se',
                err_style='bars',
                ax = ax,
                palette=palette  
                ,legend=leg
        ) 
    clean_axs(ax)  
    ax.set_xlabel('Spatial disparity, V - A ($^\circ$)', fontsize = fontsize)
    ax.set_ylabel(ylabel_map[yvar], fontsize = fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    if leg is not None:
        ax.legend(title = 'Visual reliability', 
                frameon=False, 
                ncol = 2, 
                loc = 'upper right', 
                bbox_to_anchor=(0, -0.2), 
                title_fontsize = fontsize-2, 
                fontsize = fontsize-2)
    #ax.set_title(com_label, loc='center', fontsize=fontsize, pad=20)
    if save_fig:
        plot_name = os.path.join(plot_path, plot_name_suffix + yvar + '_' + com_label + '_trial' + trial_type + '.png')
        print(f"Plot is saved @ {plot_name}")
        plt.savefig(plot_name, dpi = 300)
    #plt.show()

def plotDavidFancy(df, sorted_by, vis_rel_label, trial_type='A', save_fig = 0, plot_label=''):
    """
    Create the same plot as David did, i.e., all four reports in one plot

    Parameters
    ----------
    df : pd.DataFrame
        for beh data, first filter the dataset and make sure the ntrial is consitent across the conditions.
        Data frame is processed with CMB bias and lables.
    sorted_by : str
        RespLoc/CISize.
    vis_rel_label : str
        visual reliability label. High/Low
    trial_type : TYPE, optional
        DESCRIPTION. The default is 'A'.
    save_fig : TYPE, optional
        DESCRIPTION. The default is 0.
    plot_label : addition lables for the saved plots
        for exmaple, if simulated trial, add 'sim'.
    Returns
    -------
    None.

    """
    trial_type_map = {'A':1, 'V':0}# 1 for auditory, 0 - for visual.
    df = df[df.blockType==trial_type_map[trial_type]]
    
    ntrial = 34
    #df = df.groupby(['sub_id', 'AudLoc', 'VisLoc', 'VisRel']).head(ntrial).reset_index(drop=True)
    df = (
        df.groupby(['sub_id', 'AudLoc', 'VisLoc', 'VisRel'], group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), ntrial), random_state=42))
        .reset_index(drop=True)
    )

 
    # first sort the data according to either auditory spatial bias or CI size
    if sorted_by == 'bias':
        df = df.sort_values(by = ['sub_id', 'AudLoc', 'VisLoc', 'VisRel', 'RespLoc'], ascending = True).reset_index(drop = True)
    elif sorted_by == 'ci':
        df = df.sort_values(by = ['sub_id', 'AudLoc', 'VisLoc', 'VisRel', 'CISize'], ascending = True).reset_index(drop = True)
    
    df['TrialNrOrdered'] = df.groupby(['sub_id', 'AudLoc', 'VisLoc', 'VisRel']).cumcount() + 1
    #df['TrialNrOrdered'] = np.tile(np.arange(1, ntrial+1), df.shape[0]//ntrial)

    #according to David's logic, first make the confLevel from common and separate sources with different dign (but is it necessary?)
    # To turn Same source confidences  negative (-100 [certain] to 0 [guess])
    df.loc[df.Sources==1, 'ConfLevel'] = -1 * df.loc[df.Sources==1, 'ConfLevel']

    #avergage across subjects
    df = df.groupby(['AudLoc', 'VisLoc', 'VisRel', 'TrialNrOrdered'])[['RespLoc', 'CISize', 'Sources', 'ConfLevel']].mean().reset_index()
    df['ConfLevel'] = df['ConfLevel'].abs().round()
    # according to David's logic, reassign the common source judgement based on mean Sources
    df.loc[df['Sources']-1<=0.5, 'Sources_new'] = 1
    df.loc[df['Sources']-1>0.5, 'Sources_new'] = 2

    ### Plot 
    # Define two colormaps - shared
    cmap_com = LinearSegmentedColormap.from_list('Common', [[0.50 ,0.50 ,1.00], [0.00 ,1.00 ,1.00]])  
    cmap_sep = LinearSegmentedColormap.from_list('Separate', [[1.00 ,0.50 ,0.50], [1.00 ,1.00 ,0.00]])  
    norm = plt.Normalize(0, 100)
    # the audiovisual grid 
    # Alocs = [-10, -5, 0, 5, 10]
    # Vlocs = [10, 5, 0, -5, -10]
    vis_rel_map = {"High":1, "Low":2} #Visual reliability level ('1' = high reliability [2 SD], '2' = low reliability [14 SD])
    rel = vis_rel_map[vis_rel_label] 

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(5, 6, width_ratios=[1, 1, 1, 1, 1, 0.2], figure=fig)

    # Create subplots for data
    axs = [[fig.add_subplot(gs[row, col]) for col in range(5)] for row in range(5)]
    axs = np.array(axs)

    for i in np.arange(5):
        for j in np.arange(5):
            plot1AVpair(df, axs[i,j], i, j, rel, cmap_com, cmap_sep, norm, ntrial)

    # # Add two colorbars
    cbar_ax_com = fig.add_subplot(gs[:2, 5]) 
    cbar_ax_sep = fig.add_subplot(gs[3:, 5]) 
    sm_com = plt.cm.ScalarMappable(cmap=cmap_com, norm=norm)
    cbar_com = plt.colorbar(sm_com, cax=cbar_ax_com)
    cbar_com.set_label("Confidence(%)")
    cbar_com.ax.set_title("Common")
    sm_sep = plt.cm.ScalarMappable(cmap=cmap_sep, norm=norm)
    cbar_sep = plt.colorbar(sm_sep, cax=cbar_ax_sep)
    cbar_sep.set_label("Separate: Confidence(%)")
    cbar_sep.ax.set_title("Separate")
    fig.supxlabel('Auditory Locations')
    fig.supylabel('Visual Locations')
    fig.suptitle(f"{trial_type} Report, Sorted by: {sorted_by}, Visual reliability: {vis_rel_label}, {plot_label}", fontsize=20)
    #plt.title(f"Visual reliability level: {visrel_map[rel]}")
    if save_fig:
        plot_name = os.path.join('./plots/david', plot_label + '_sortby_' + sorted_by + '_Vreli_' + str(rel) + '.png')
        print(f"Plot is saved @ {plot_name}")
        plt.savefig(plot_name, dpi = 300)
    plt.show()


#help plot function for plotDavidFancy
def plot1AVpair(df, ax, i, j, rel, cmap_com, cmap_sep, norm, ntrial, Alocs = [-10, -5, 0, 5, 10],  Vlocs = [10, 5, 0, -5, -10], smooth=0):
    data = df[(df.AudLoc== Alocs[j]) & (df.VisLoc== Vlocs[i]) & (df.VisRel== rel)].reset_index(drop = True)

    if smooth == 0:
        # Create color mappings
        com_colors = cmap_com(norm(data.loc[data.Sources_new==1, 'ConfLevel']))
        sep_colors = cmap_sep(norm(data.loc[data.Sources_new==2, 'ConfLevel']))
    else:
        # add smoothing colors
        sigma = 0.5  #larger = more smoothing
        smoothed_com_conf = gaussian_filter1d(data.loc[data.Sources_new == 1, 'ConfLevel'], sigma)
        smoothed_sep_conf = gaussian_filter1d(data.loc[data.Sources_new == 2, 'ConfLevel'], sigma)
        # Create color mappings with smoothed confidence levels
        com_colors = cmap_com(norm(smoothed_com_conf))
        sep_colors = cmap_sep(norm(smoothed_sep_conf))
    
    # Prepare CI bar 
    com_segments = [
        [[x, y - err / 2], [x, y + err / 2]]  # Bottom to top segment
        for x, y, err in data.loc[data['Sources_new'] == 1, ['TrialNrOrdered', 'RespLoc', 'CISize']].values
    ]
    sep_segments = [
        [[x, y - err / 2], [x, y + err / 2]]  # Bottom to top segment
        for x, y, err in data.loc[data['Sources_new'] == 2, ['TrialNrOrdered', 'RespLoc', 'CISize']].values
    ]
    
    # Create two LineCollections for different probability groups
    com_collection = LineCollection(com_segments, colors=com_colors, linewidths=4)
    sep_collection = LineCollection(sep_segments, colors=sep_colors, linewidths=4)
   
    # Add two sets of CI bars
    ax.add_collection(com_collection)
    ax.add_collection(sep_collection)
    
    # add three lines: the response location, the top and bottom CI
    ax.plot(data['TrialNrOrdered'], data['RespLoc'], color = 'b', lw = 0.5)
    ax.plot(data['TrialNrOrdered'], data['RespLoc']+data['CISize']/2, color = 'b', lw = 0.5)
    ax.plot(data['TrialNrOrdered'], data['RespLoc']-data['CISize']/2, color = 'b', lw = 0.5)

    #clean axes
    clean_axs(ax, 10)
    ax.set_xticks(np.arange(0, ntrial + 1, step=10))  
    ax.set_xlim(0, ntrial + 1)  
    ax.set_ylim(-20, 20)
    
    if i==4:
        ax.set_xlabel(f"{Alocs[j]}\u00b0", fontweight = 'bold')     
    if j==0:
        ax.set_ylabel(f"{Vlocs[i]}\u00b0", fontweight = 'bold')


    