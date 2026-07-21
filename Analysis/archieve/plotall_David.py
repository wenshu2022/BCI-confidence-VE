import os
work_path = "M:/1confiProj/Analysis"
os.chdir(work_path)
#the help fucntion (for plotting) import all the necessary librarys and define plot parameters. 
from help_function import *

###################
## Prepare data ###
###################
#read data
dfAllSub = pd.read_csv(os.path.join(data_path, 'bimodalAllSub.csv'))
dfAllSubAud = dfAllSub.loc[dfAllSub.blockType ==1]# select the auditory blocks
dfAllSubAud.columns
#first a quick analysis, distribution of number of avaialble trials for each combinitations
cntTrial = dfAllSubAud.groupby(['sub_id', 'AudLoc', 'VisLoc', 'VisRel'])['BlockNr'].count()
#cntTrial.value_counts() # mostly the trial recur 36 times
cntTrial.describe() # for each sub and each AV location/reliability combos, min 34 occurence.
nPnts = 34
# note: filter is only for beh data as not all subjects finish every trial
dfAllSubAud_filterd = dfAllSubAud.groupby(['sub_id', 'AudLoc', 'VisLoc', 'VisRel']).head(nPnts).reset_index(drop=True)

sorted_by = 'ci'
# first sort the data according to either auditory spatial bias or CI size
if sorted_by == 'audbias':
    dfAllSubAud_filterd = dfAllSubAud_filterd.sort_values(by = ['sub_id', 'AudLoc', 'VisLoc', 'VisRel', 'RespLoc'], ascending = True).reset_index(drop = True)
elif sorted_by == 'ci':
    dfAllSubAud_filterd = dfAllSubAud_filterd.sort_values(by = ['sub_id', 'AudLoc', 'VisLoc', 'VisRel', 'CISize'], ascending = True).reset_index(drop = True)

dfAllSubAud_filterd['TrialNrOrdered'] = np.tile(np.arange(1, nPnts+1), dfAllSubAud_filterd.shape[0]//nPnts)

#according to David's logic, first make the confLevel from common and separate sources with different dign (but is it necessary?)
dfAllSubAud_filterd.loc[dfAllSubAud_filterd.Sources==1, ['ConfLevel', 'ConfLevelNorm']] = -1 * dfAllSubAud_filterd.loc[dfAllSubAud_filterd.Sources==1, ['ConfLevel', 'ConfLevelNorm']]

#avergage across subjects
dfAggAud = dfAllSubAud_filterd.groupby(['AudLoc', 'VisLoc', 'VisRel', 'TrialNrOrdered'])[['RespLoc', 'CISize', 'Sources', 'ConfLevel', 'ConfLevelNorm']].mean().reset_index()
dfAggAud[['ConfLevel', 'ConfLevelNorm']] = abs(dfAggAud[['ConfLevel', 'ConfLevelNorm']])
dfAggAud['ConfLevel'] = round(dfAggAud['ConfLevel'])

# according to David's logic, reassign the common source judgement based on mean Sources
dfAggAud.loc[dfAggAud['Sources']-1<=0.5, 'Sources_new'] = 1
dfAggAud.loc[dfAggAud['Sources']-1>0.5, 'Sources_new'] = 2


###################
####### Plot ######
###################
# OneSource_LowConf   = [0.50 0.50 1.00];     %halfway magenta and cyan
# OneSource_HighConf  = [0.00 1.00 1.00];     %cyan
# TwoSource_LowConf   = [1.00 0.50 0.50];     %halfway magenta and yellow
# TwoSource_HighConf  = [1.00 1.00 0.00];     %yellow
# Define two colormaps - shared
cmap_com = LinearSegmentedColormap.from_list('common', [[0.50 ,0.50 ,1.00], [0.00 ,1.00 ,1.00]])  
cmap_sep = LinearSegmentedColormap.from_list('separate', [[1.00 ,0.50 ,0.50], [1.00 ,1.00 ,0.00]])  
norm = plt.Normalize(0, 100)
# the audiovisual grid 
Alocs = [-10, -5, 0, 5, 10]
Vlocs = [10, 5, 0, -5, -10]
visrel_map = {1:"High", 2:"Low"}

rel = 2  #Visual reliability level ('1' = high reliability [2 SD], '2' = low reliability [14 SD])

#plot function
def plot1AVpair(dfAggAud, axs, i, j, smooth=0):
    
    data = dfAggAud[(dfAggAud.AudLoc== Alocs[j]) & (dfAggAud.VisLoc== Vlocs[i]) & (dfAggAud.VisRel== rel)].reset_index(drop = True)
    ax = axs[i,j]
    
    if smooth == 0:
        # Create color mappings
        com_colors = cmap_com(norm(data.loc[data.Sources_new==1, 'ConfLevelNorm']))
        sep_colors = cmap_sep(norm(data.loc[data.Sources_new==2, 'ConfLevelNorm']))
    else:
        # add smoothing colors
        sigma = 0.5  #larger = more smoothing
        smoothed_com_conf = gaussian_filter1d(data.loc[data.Sources_new == 1, 'ConfLevelNorm'], sigma)
        smoothed_sep_conf = gaussian_filter1d(data.loc[data.Sources_new == 2, 'ConfLevelNorm'], sigma)
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
    ax.set_xticks(np.arange(0, nPnts + 1, step=10))  
    ax.set_xlim(0, nPnts + 1)  
    ax.set_ylim(-20, 20)
    
    if i==4:
        ax.set_xlabel(f"{Alocs[j]}\u00b0", fontweight = 'bold')
        
    if j==0:
        ax.set_ylabel(f"{Vlocs[i]}\u00b0", fontweight = 'bold')


# Plot
#fig, axs = plt.subplots(figsize=(12, 12), ncols = 6, nrows = 5, sharey=True, width_ratios=[1, 1, 1, 1, 1, 0.2])
fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(5, 6, width_ratios=[1, 1, 1, 1, 1, 0.2], figure=fig)

# Create subplots for data
axs = [[fig.add_subplot(gs[row, col]) for col in range(5)] for row in range(5)]
axs = np.array(axs)

for i in np.arange(5):
    for j in np.arange(5):
        plot1AVpair(dfAggAud, axs, i, j)

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
#plt.title(f"Visual reliability level: {visrel_map[rel]}")
plt.savefig(os.path.join('./output', 'beh_sortby_' + sorted_by + '_Vreli_' + str(rel) + '.png'), dpi = 300)
plt.show()
