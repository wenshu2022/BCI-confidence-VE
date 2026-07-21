import os
work_path = "M:/1confiProj/Analysis"
os.chdir(work_path)
#the help fucntion (for plotting) import all the necessary librarys and define plot parameters. 
from help_function import *

###################
## Prepare data ###
###################
#read data
dfAllSubAud = pd.read_csv(os.path.join(data_path, 'BlockACMB.csv'))
#dfAllSubAud = dfAllSub.loc[dfAllSub.blockType ==1]# select the auditory blocks
dfAllSubAud.columns

####################################################################
###plot 1: CI size/conf/bias as a function of spatial disparity

#dfAllSubAud.loc[:,'delta_VA'] =  dfAllSubAud.VisLoc- dfAllSubAud.AudLoc

com_label = {1:'Common', 2: 'Separate', 0: 'Total'}
visrel_label = {1: 'High', 2: 'Low'}
dfAllSubAud['VisRelLabel'] = dfAllSubAud['VisRel'].map(visrel_label)
ylabel = {'CISize': 'Confidence interval', 'ConfLevel' : 'Causal confidence (0-100)', 'bias' : 'Auditory bias'}
yvar = 'bias'

source_label = 0 # 1- common, 2 - separate , 0 - two combined.
# two-step avergae
if source_label>0:
    dfBySubAud = dfAllSubAud.groupby(['sub_id', 'delta_VA', 'VisRelLabel', 'Sources'])[['CISize', 'ConfLevel', 'bias']].mean().reset_index()
    dfBySubAud = dfBySubAud.loc[dfBySubAud.Sources==source_label]
else:
    dfBySubAud = dfAllSubAud.groupby(['sub_id', 'delta_VA', 'VisRelLabel'])[['CISize', 'ConfLevel', 'bias']].mean().reset_index()
    
fig, ax = plt.subplots(figsize = (6, 5))
sns.lineplot(data = dfBySubAud,
            x = 'delta_VA',
            y = yvar,
            hue = 'VisRelLabel',
            estimator = 'mean',
            errorbar='se',
            err_style='bars',
            ax = ax
    ) 
clean_axs(ax)  
ax.set_xlabel('Spatial disparity, V - A ($^\circ$)', fontsize = fontsize)
ax.set_ylabel(ylabel[yvar], fontsize = fontsize)
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.legend(title = 'Visual reliability', 
        frameon=False, 
        ncol = 2, 
        loc = 'upper right', 
        bbox_to_anchor=(1, 1.05), 
        title_fontsize = fontsize-2, 
        fontsize = fontsize-2)
ax.set_title(com_label[source_label], loc='center', fontsize=fontsize, pad=20)
plt.show()

####################################################################
###plot 2: David style but on spatial disparity x

dfAllSub = pd.read_csv(os.path.join(data_path, 'BlockACMB.csv'))
dfAllSubAud = dfAllSub.loc[dfAllSub.blockType ==1]# select the auditory blocks
#dfAllSubAud.loc[:,'delta_AV'] = dfAllSubAud.loc[:, 'AudLoc'] - dfAllSubAud.loc[:, 'VisLoc']
dfAllSubAud.columns
#first a quick analysis, distribution of number of avaialble trials for each combinitations
cntTrial = dfAllSubAud.groupby(['sub_id', 'AudLoc', 'VisLoc', 'VisRel'])['BlockNr'].count()
#cntTrial.value_counts() # mostly the trial recur 36 times
cntTrial.describe() # for each sub and each AV location/reliability combos, min 34 occurence.
nPnts = 34
dfAllSubAud_filterd = dfAllSubAud.groupby(['sub_id', 'AudLoc', 'VisLoc', 'VisRel']).head(nPnts).reset_index(drop=True)

sorted_by = 'audbias'
# first sort the data according to either auditory spatial bias or CI size
if sorted_by == 'audbias':
    dfAllSubAud_filterd = dfAllSubAud_filterd.sort_values(by = ['sub_id', 'AudLoc', 'VisLoc', 'VisRel', 'flipBias'], ascending = True).reset_index(drop = True)
elif sorted_by == 'ci':
    dfAllSubAud_filterd = dfAllSubAud_filterd.sort_values(by = ['sub_id', 'AudLoc', 'VisLoc', 'VisRel', 'CISize'], ascending = True).reset_index(drop = True)

dfAllSubAud_filterd['TrialNrOrdered'] = np.tile(np.arange(1, nPnts+1), dfAllSubAud_filterd.shape[0]//nPnts)

#according to David's logic, first make the confLevel from common and separate sources with different dign (but is it necessary?)
dfAllSubAud_filterd.loc[dfAllSubAud_filterd.Sources==1, ['ConfLevel', 'ConfLevelNorm']] = -1 * dfAllSubAud_filterd.loc[dfAllSubAud_filterd.Sources==1, ['ConfLevel', 'ConfLevelNorm']]

#avergage across subjects
dfAggAud = dfAllSubAud_filterd.groupby(['abs_delta_VA', 'VisRel', 'TrialNrOrdered'])[['flipBias', 'CISize', 'Sources', 'ConfLevel', 'ConfLevelNorm']].mean().reset_index()
dfAggAud[['ConfLevel', 'ConfLevelNorm']] = abs(dfAggAud[['ConfLevel', 'ConfLevelNorm']])
dfAggAud['ConfLevel'] = round(dfAggAud['ConfLevel'])

# according to David's logic, reassign the common source judgement based on mean Sources
dfAggAud.loc[dfAggAud['Sources']-1<=0.5, 'Sources_new'] = 1
dfAggAud.loc[dfAggAud['Sources']-1>0.5, 'Sources_new'] = 2

###################
####### Plot ######
###################
# Define two colormaps - shared
cmap_com = LinearSegmentedColormap.from_list('common', [[0.50 ,0.50 ,1.00], [0.00 ,1.00 ,1.00]])  
cmap_sep = LinearSegmentedColormap.from_list('separate', [[1.00 ,0.50 ,0.50], [1.00 ,1.00 ,0.00]])  
norm = plt.Normalize(0, 100)
# the audiovisual grid 
del_lst = dfAggAud.abs_delta_VA.unique()
Alocs = [-10, -5, 0, 5, 10]
Vlocs = [10, 5, 0, -5, -10]
visrel_map = {1:"High", 2:"Low"}

rel = 1  #Visual reliability level ('1' = high reliability [2 SD], '2' = low reliability [14 SD])

#plot function
def plot1AVpair(dfAggAud, axs, i, smooth=0):
    
    data = dfAggAud[(dfAggAud.abs_delta_VA== del_lst[i]) & (dfAggAud.VisRel== rel)].reset_index(drop = True)
    ax = axs[i]
    
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
        for x, y, err in data.loc[data['Sources_new'] == 1, ['TrialNrOrdered', 'flipBias', 'CISize']].values
    ]
    sep_segments = [
        [[x, y - err / 2], [x, y + err / 2]]  # Bottom to top segment
        for x, y, err in data.loc[data['Sources_new'] == 2, ['TrialNrOrdered', 'flipBias', 'CISize']].values
    ]
    
    # Create two LineCollections for different probability groups
    com_collection = LineCollection(com_segments, colors=com_colors, linewidths=4)
    sep_collection = LineCollection(sep_segments, colors=sep_colors, linewidths=4)
   
    # Add two sets of CI bars
    ax.add_collection(com_collection)
    ax.add_collection(sep_collection)
    
    # add three lines: the response location, the top and bottom CI
    ax.plot(data['TrialNrOrdered'], data['flipBias'], color = 'b', lw = 0.5)
    ax.plot(data['TrialNrOrdered'], data['flipBias']+data['CISize']/2, color = 'b', lw = 0.5)
    ax.plot(data['TrialNrOrdered'], data['flipBias']-data['CISize']/2, color = 'b', lw = 0.5)

    #clean axes
    clean_axs(ax, 10)
    ax.set_xticks(np.arange(0, nPnts + 1, step=10))  
    ax.set_xlim(0, nPnts + 1)  
    ax.set_ylim(-20, 20)
    


# Plot
#fig, axs = plt.subplots(figsize=(12, 12), ncols = 6, nrows = 5, sharey=True, width_ratios=[1, 1, 1, 1, 1, 0.2])
fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, len(del_lst)+4, width_ratios=[1, 1, 1, 1, 1, 0.1, 0.2, 0.3, 0.2], figure=fig)

# Create subplots for data
axs = [fig.add_subplot(gs[col]) for col in range(len(del_lst))] 
axs = np.array(axs)

for i in np.arange(len(del_lst)):
    plot1AVpair(dfAggAud, axs, i)
    axs[i].set_xticks([])
    axs[i].set_xlabel(del_lst[i])

# # Add two colorbars
cbar_ax_com = fig.add_subplot(gs[len(del_lst)+1]) 
cbar_ax_sep = fig.add_subplot(gs[len(del_lst)+3]) 
sm_com = plt.cm.ScalarMappable(cmap=cmap_com, norm=norm)
cbar_com = plt.colorbar(sm_com, cax=cbar_ax_com)
#cbar_com.set_label("Confidence(%)")
cbar_com.ax.set_title("Common")
sm_sep = plt.cm.ScalarMappable(cmap=cmap_sep, norm=norm)
cbar_sep = plt.colorbar(sm_sep, cax=cbar_ax_sep)
cbar_sep.set_label("Confidence(%)")
cbar_sep.ax.set_title("Separate")
fig.supxlabel('Absolute spatial disparity ($^\circ$)')
fig.supylabel('Flip response bias with confidence interval')
for i in np.arange(1, len(del_lst)):
    axs[i].set_yticks([])
    axs[i].spines['left'].set_visible(False)
    
#plt.title(f"Visual reliability level: {visrel_map[rel]}")
#plt.savefig(os.path.join('./output', 'beh_sortby_' + sorted_by + '_Vreli_' + str(rel) + '.png'), dpi = 300)
plt.show()




