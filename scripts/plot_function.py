import os
wkdir = 'M:/1confiProj/'
os.chdir(wkdir)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter1d
#import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator
# import matplotlib.gridspec as gridspec
# from matplotlib.lines import Line2D

sns.set_theme(style="white")
plot_path = 'M:\\1confiProj\\plots\\'
all_data_path = "P:/3026008.02/AllData/"
plot_path = "M:/1confiProj/plots/"
fit_data_path = "P:/3026008.02/data_for_fitting/"
fit_res_path = "M:\\1confiProj\\fit_res\\output\\"
fontsize = 16

### help functions
#define color scheme.
palette_beh_vrel= {'High': '#4C72B0', 'Low': '#DD8452'}
palette_mod_vrel= {'High': '#9FB3C8', 'Low': '#EBD38A'}
palette_vloc = {
    -10: "#08519c",  # dark blue
    -5:  "#6baed6",  # light blue
     0:  "#7f7f7f",  # gray
     5:  "#74c476",  # light green
    10: "#006d2c",   # dark green
}
palette_d = sns.color_palette("viridis", as_cmap=False)[::-1]
palette_ci = {1: "tab:purple", 2:"tab:olive"}
              
def clean_axs(ax,fontsize=16):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', direction='in', left=True, bottom=True, labelsize=fontsize)

def ax_noXlabel(ax):
    ax.set_xlabel('')
    ax.xaxis.set_ticklabels([])

def ax_noYlabel(ax):
    ax.set_ylabel('')
    ax.yaxis.set_ticklabels([])

def scale_marker_size(n, global_min, global_max,
                      power=2, smin=50, smax=300):
    # first power then scale to make the trial counts difference more visible
    n = np.power(n, power)
    gmin_pw = np.power(global_min, power)
    gmax_pw = np.power(global_max, power)
    return (
        smin +
        (n - gmin_pw) /
        (gmax_pw - gmin_pw) *
        (smax - smin)
    )

def get_global_min_max(df, groupvar, modality='A'):
    df_plot = df[df['Modality'] == modality]

    if isinstance(groupvar, str):
        groupvar = [groupvar]

    g_lst = ['VisRelLabel', 'Sources'] + groupvar

    all_summary = (
        df_plot
        .groupby(g_lst)
        .agg(n=('RespLoc', 'size'))
        .reset_index()
    )

    return all_summary["n"].min(), all_summary["n"].max()

###################################################################################################
### plot the raw localization data
def plot_raw_rep_all(df, rel, ax, visloc, palette=palette_vloc,global_min=None, global_max=None,
                     power=2,
                     resp_var='RespLoc',
                     modality='A',
                     legend=True,
                     scale = True):

    df_plot = df[(df['Modality'] == modality) &
                 (df['VisRelLabel'] == rel)]
    # ------------------------------------------------------------
    # Helper function
    # ------------------------------------------------------------
    def plot_condition(df_cond, color, label):

        # Mean, SEM and total number of trials - two steps averaging
        summary_sub = (
            df_cond
            .groupby(["AudLoc", "sub_id"])
            .agg(
                mean=(resp_var, "mean"),
                n=(resp_var, "size")
            )
            .reset_index()
        )
        summary = (
            summary_sub
            .groupby(["AudLoc"])
            .agg(
                mean=('mean', "mean"),
                sem=('mean', "sem"),
                n=('n', "sum")
            )
            .reset_index()
            .sort_values("AudLoc")
        )

        # use scaling to exaggerate the difference between the size of marker - not really work..
        if scale:
            summary["marker_size"] = scale_marker_size(n=summary["n"], global_min=global_min, global_max=global_max,
                      power=power, smin=50, smax=300)

        else:
            summary["marker_size"] = summary["n"]

        ax.scatter(
            summary["AudLoc"],
            summary["mean"],
            s=summary["marker_size"],
            color=color,
            # edgecolors="black",
            # linewidth=0.5,
            zorder=3,
            alpha=0.7,
        )
        ax.errorbar(
            summary["AudLoc"],
            summary["mean"],
            yerr=summary["sem"],
            color=color,
            linewidth=2,
            capsize=2,
            label=label,
            zorder=2,
            #alpha=0.7,
            marker='o',
            markersize=5,
        )


    # ------------------------------------------------------------
    # Congruent condition
    # ------------------------------------------------------------
    plot_condition(
        df_plot[df_plot.AudLoc == df_plot.VisLoc],
        color="red",
        label="Congruent"
    )
    if resp_var=='RespLoc':
        ax.plot(
            [-10, 10],
            [-10, 10],
            "--",
            color="red",
            linewidth=1,
            alpha=0.6
        )

    # ------------------------------------------------------------
    # Fixed positive visual location
    # ------------------------------------------------------------
    plot_condition(
        df_plot[df_plot.VisLoc == visloc],
        color=palette[visloc],
        label=f"Visual Stimulus Location = {visloc}"
    )
    if resp_var=='RespLoc':
        ax.hlines(
            y=visloc,
            xmin=-10,
            xmax=10,
            color=palette[visloc],
            linestyle="--",
            linewidth=1
        )

    # ------------------------------------------------------------
    # Fixed negative visual location
    # ------------------------------------------------------------
    if visloc != 0:

        plot_condition(
            df_plot[df_plot.VisLoc == -visloc],
            color=palette[-visloc],
            label=f"Visual Stimulus Location = {-visloc}"
        )
        if resp_var=='RespLoc':
            ax.hlines(
                y=-visloc,
                xmin=-10,
                xmax=10,
                color=palette[-visloc],
                linestyle="--",
                linewidth=1
            )

    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------
    if legend:
        ax.legend(
            frameon=False,
            ncol=2,
            fontsize=fontsize-2,
            loc="lower center",
            bbox_to_anchor=(0.9, -0.5)
        )

    clean_axs(ax)
    if resp_var=='RespLoc':
        ax.set_aspect('equal', adjustable='box')
        ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel('Sound Location ($^\\circ$)', fontsize=fontsize)


###################################################################################################
## plot localization uncertainty by spatial disparity
def plot_CI_size_by_disp_vrel(df, ax, m_id=None, palette = palette_beh_vrel, Modality='A', legend=False, scale=False, k=1):
    
    if m_id is not None:
        #model simulation data
        df = df.loc[(df.m_id == m_id) & (df.Modality == Modality)].copy()
    else:
        #beh data
        df = df.loc[df.Modality == Modality].copy()
    #df_plot = df.groupby(['sub_id', 'VisRelLabel', 'abs_delta_VA']).CISize.mean().reset_index()

    # ------------------------------------------------------------
    # Helper function
    # ------------------------------------------------------------
    def plot_condition(df, vrel):
        df_cond = df.loc[df.VisRelLabel==vrel]
        # Mean, SEM and total number of trials - two steps averaging
        summary_sub = (
            df_cond
            .groupby(['abs_delta_VA', "sub_id"])
            .agg(
                mean=('CISize', "mean"),
                n=('CISize', "size")
            )
            .reset_index()
        )
        summary = (
            summary_sub
            .groupby(["abs_delta_VA"])
            .agg(
                mean=('mean', "mean"),
                sem=('mean', "sem"),
                n=('n', "sum")
            )
            .reset_index()
            .sort_values("abs_delta_VA")
        )

        if scale:
            # for model simulation data (many simulated trials), linearly decrease to match the behavioural trial number in each condition
            # summary["marker_size"] = scale_marker_size(n=summary["n"], global_min=global_min, global_max=global_max,
            #           power=power, smin=50, smax=300)
            summary["marker_size"] = summary["n"]/k
        else:
            summary["marker_size"] = summary["n"]

        ax.scatter(
            summary["abs_delta_VA"],
            summary["mean"],
            s=summary["marker_size"],
            color=palette[vrel],
            # edgecolors="black",
            # linewidth=0.5,
            zorder=3,
            alpha=0.7,
        )
        ax.errorbar(
            summary["abs_delta_VA"],
            summary["mean"],
            yerr=summary["sem"],
            color=palette[vrel],
            linewidth=2,
            capsize=2,
            label=f'{vrel} visual reliability',
            zorder=2,
            #alpha=0.7,
            marker='o',
            markersize=5,
        )

    plot_condition(df, 'High')
    plot_condition(df, 'Low')
    if legend:
        ax.legend(
            frameon=False,
            ncol=2,
            fontsize=fontsize-2,
            loc='lower center',
            bbox_to_anchor=(0.4, -0.4),
        )

    clean_axs(ax)

def plot_CI_size_by_disp(df, ax, m_id=None, palette = palette_ci, Modality='A', legend=False, scale=False, k=1):
    
    if m_id is not None:
        #model simulation data
        df = df.loc[(df.m_id == m_id) & (df.Modality == Modality)].copy()
    else:
        #beh data
        df = df.loc[df.Modality == Modality].copy()
    #df_plot = df.groupby(['sub_id', 'VisRelLabel', 'abs_delta_VA']).CISize.mean().reset_index()
    sources_label = {1:'Report same cause', 2:'Report different causes'}
    # ------------------------------------------------------------
    # Helper function
    # ------------------------------------------------------------
    #def plot_condition(df, vrel):
        #df_cond = df.loc[df.VisRelLabel==vrel]
        # Mean, SEM and total number of trials - two steps averaging
    summary_sub = (
        df
        .groupby(['abs_delta_VA', 'Sources', "sub_id"])
        .agg(
            mean=('CISize', "mean"),
            n=('CISize', "size")
        )
        .reset_index()
    )
    summary = (
        summary_sub
        .groupby(["abs_delta_VA", 'Sources'])
        .agg(
            mean=('mean', "mean"),
            sem=('mean', "sem"),
            n=('n', "sum")
        )
        .reset_index()
        .sort_values(["abs_delta_VA", 'Sources'])
    )

    if scale:
        # for model simulation data (many simulated trials), linearly decrease to match the behavioural trial number in each condition
        # summary["marker_size"] = scale_marker_size(n=summary["n"], global_min=global_min, global_max=global_max,
        #           power=power, smin=50, smax=300)
        summary["marker_size"] = summary["n"]/k
    else:
        summary["marker_size"] = summary["n"]

    for sour in [1, 2]:
        ax.scatter(
            summary.loc[summary.Sources==sour, "abs_delta_VA"],
            summary.loc[summary.Sources==sour,"mean"],
            s=summary.loc[summary.Sources==sour,"marker_size"],
            color=palette[sour],
            # edgecolors="black",
            # linewidth=0.5,
            zorder=3,
            alpha=0.7,
        )
        ax.errorbar(
            summary.loc[summary.Sources==sour, "abs_delta_VA"],
            summary.loc[summary.Sources==sour, "mean"],
            yerr=summary.loc[summary.Sources==sour, "sem"],
            color=palette[sour],
            linewidth=2,
            capsize=2,
            label=sources_label[sour],
            zorder=2,
            #alpha=0.7,
            marker='o',
            markersize=5,
        )

    if legend:
        ax.legend(
            #title = 'Sources',
            frameon=False,
            ncol=2,
            fontsize=fontsize-2,
            loc='lower center',
            bbox_to_anchor=(0.4, -0.4),
        )

    clean_axs(ax)

###################################################################################################
### plot localization uncertainty with locations and disparities
def plot_CI_size_by_condition(df, m_id=None, axs=None, Modality='A', loc_var = 'AudLoc', palette=palette_d):
    
    if m_id is not None:
        #model simulation data
        df = df.loc[(df.m_id == m_id) & (df.Modality == Modality)].copy()
    else:
        #beh data
        df = df.loc[df.Modality == Modality].copy()
    df_plot = df.groupby(['sub_id', 'VisRelLabel', 'abs_delta_VA', loc_var]).CISize.mean().reset_index()

    sns.lineplot(
        data=df_plot[df_plot.VisRelLabel == 'High'],
        x=loc_var,
        y='CISize',
        hue='abs_delta_VA',
        palette=palette,
        linewidth=2.5,
        ax = axs[0],
        legend = None,
        estimator = 'mean',
        errorbar='se',
        err_style='bars',
    )
    sns.lineplot(
        data=df_plot[df_plot.VisRelLabel == 'Low'],
        x=loc_var,
        y='CISize',
        hue='abs_delta_VA',
        palette=palette,
        linewidth=2.5,
        ax = axs[1],
        estimator = 'mean',
        errorbar='se',
        err_style='bars',
        #legend = None,
    )
    clean_axs(axs[0])
    clean_axs(axs[1])
    axs[0].set_xlabel('')
    axs[1].set_xlabel('')
    axs[1].legend(
        title='Absolute spatial disparity ($^\\circ$)',
        frameon=False,
        loc='lower center',
        ncol=5,
        bbox_to_anchor=(0, -0.5),
        fontsize=fontsize-2,
        title_fontsize=fontsize-2
    )

    axs[0].set_ylabel('Localization uncertainty \n response ($^\\circ$)', fontsize=fontsize)
    axs[0].set_xlabel('Sound location ($^\\circ$)', fontsize=fontsize)
    axs[1].set_xlabel('Sound location ($^\\circ$)', fontsize=fontsize)

###################################################################################################
### plot causal confidence
def plot_casual_conf(df,ax, palette=palette_beh_vrel, trial_type=None, leg=False, scale=False, k=1):

    trial_type_map = {'A':1, 'V':0}# 1 for auditory, 0 - for visual.
    if trial_type is not None:
        df = df[df.blockType==trial_type_map[trial_type]]

    # helper fuction
    def plot_condition(df_cond, color,label):
         # Mean, SEM and total number of trials - two steps averaging
        summary_sub = (
            df_cond
            .groupby(["delta_VA", "sub_id"])
            .agg(
                mean=('ConfLevel', "mean"),
                n=('ConfLevel', "size")
            )
            .reset_index()
        )
        summary = (
            summary_sub
            .groupby(["delta_VA"])
            .agg(
                mean=('mean', "mean"),
                sem=('mean', "sem"),
                n=('n', "sum")
            )
            .reset_index()
            .sort_values("delta_VA")
        )

        if scale:
            # for model simulation data (many simulated trials), linearly decrease to match the behavioural trial number in each condition
            #summary["marker_size"] = scale_marker_size(n=summary["n"], global_min=global_min, global_max=global_max, power=power)
            summary["marker_size"] = summary["n"]/k 
        else:
            summary["marker_size"] = summary["n"]
            
        ax.errorbar(
            summary["delta_VA"],
            summary["mean"],
            yerr=summary["sem"],
            color=color,
            linewidth=2,
            capsize=0,
            label=label,
            zorder=2,
            #alpha=0.7,
            marker='o',
            markersize=5,
        )
        ax.scatter(
                summary["delta_VA"],
                summary["mean"],
                s=summary["marker_size"], 
                color=color,
                # edgecolors="black",
                # linewidth=0.5,
                zorder=3,
                alpha=0.7,
            )
    plot_condition(
        df.loc[df.VisRelLabel == 'High'],
        color=palette['High'],
        label="High visual reliability"
    )
    plot_condition(
        df.loc[df.VisRelLabel == 'Low'],
        color=palette['Low'],
        label="Low visual reliability"
    )

    clean_axs(ax, fontsize=fontsize-2)  
    ax.set_xlabel('Spatial disparity, V - A ($^\circ$)', fontsize = fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    if leg:
        ax.legend(
                frameon=False, 
                ncol = 2, 
                loc = 'upper right', 
                bbox_to_anchor=(0.8, -0.2), 
                fontsize = fontsize-2)

###################################################################################################

###################################################################################################

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


    