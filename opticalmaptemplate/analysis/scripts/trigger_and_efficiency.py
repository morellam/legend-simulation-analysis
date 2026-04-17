#import LEGENDPlotStyle as lps

import os
import sys
# import math
import pickle
# import uproot
# import random
import argparse
import opanalysis
import numpy as np
import pandas as pd
# from tqdm.notebook import tqdm
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(description = 'script to generate plot and data with panels hit per muon event')
parser.add_argument('-e', '--efficiency', required = True, help = "specify photon detection efficinecy of a single light guide")
parser.add_argument('-l', '--lateral',    required = True, help = "specify number of lateral light guides")
parser.add_argument('-d', '--lid',        required = True, help = "specify number of lids light guides")

args = parser.parse_args()

eff = int(float(vars(args)["efficiency"])*1000)
lat = int(vars(args)["lateral"])
lid = int(vars(args)["lid"])


ar41_windowed_path = "/lfs/l1/legend/users/morella/legend1k_simulation/argon_energy_deposit/shared/legend-simulation-analysis/opticalmaptemplate/analysis/jobs/data/2025_March_CollabMeeting/Ar_and_H"
# ar41_nowindow_path = "/lfs/l1/legend/users/morella/legend1k_simulation/argon_energy_deposit/shared/legend-simulation-analysis/opticalmaptemplate/analysis/jobs/data/no_timing_Ar41_small_moderator"
ar39_path = "/lfs/l1/legend/users/morella/legend1k_simulation/argon_energy_deposit/shared/legend-simulation-analysis/opticalmaptemplate/analysis/jobs/data/2025_March_CollabMeeting/Ar39"


# efficiency     = set([int(file.split("_")[3]) for file in os.listdir(ar39_path)])
# lateral_guides = set([int(file.split("_")[4]) for file in os.listdir(ar39_path)])
# lid_guides     = set([int(file.split("_")[5]) for file in os.listdir(ar39_path)])

ge_flag = False

marker = ["o", "s", "d", "^", "v", "*", "P", "+", "x"]
color = ["cornflowerblue", "tab:green", "tab:orange", "tab:purple", "navy", "tab:grey", "tab:green", "black"]

PE_threshold_list = np.arange(0, 501, 1)
#multiplicity_list = [1,3,5,7]
multiplicity_list = [9,11]

# for eff in tqdm(efficiency):
    #for lat in tqdm(lateral_guides):

fig, ax = plt.subplots(figsize = (8,6))

fig2, ax2 = plt.subplots(figsize = (8,6), sharey = True)

    
# should be more general, I have to change everytime the file name 
try:
    ar41_df = pd.read_pickle(os.path.join(ar41_windowed_path, f"pe_per_panel_{eff}_{lat}_{lid}_Ar_and_H.pkl"))
except:
    print(f"file missing: pe_per_panel_{eff}_{lat}_{lid}_Ar.pkl")
    sys.exit(1)

try:
    ar39_df = pd.read_pickle(os.path.join(ar39_path, f"pe_per_panel_{eff}_{lat}_{lid}_Ar39.pkl"))
except:
    print(f"file missing: pe_per_panel_{eff}_{lat}_{lid}_Ar39.pkl")
    sys.exit(1)

ge77_events = []
with open('Ge77_events_2025_CollabMeeting.txt') as f:
    lines = f.readlines()
    for l in lines:
        ge77_events.append(int(l)-1)                            

if ge_flag:
    ar41_df = ar41_df.loc[ar41_df.index.get_level_values("EventID").isin(ge77_events)]
else:
    ar41_df = ar41_df.loc[~ar41_df.index.get_level_values("EventID").isin(ge77_events)]
                
                
# these lines are needed because the muon delayed coincidence is in [10microseconds, 1ms]
ar41_df = ar41_df[ar41_df.index.get_level_values('time_bin').left >= 0.00001]
ar41_df = ar41_df[ar41_df.index.get_level_values('time_bin').left <  0.001]

# get full list of Ar39 events (1 event = 10 microseconds worth of Ar39)
ar39_events = np.array(ar39_df.index.get_level_values("EventID").unique())

total_nof_ar39_events = 100000

# randomly selecting N (10mus worth of) Ar39 events
random_ar39_events = np.random.choice(ar39_events, total_nof_ar39_events)
ar39_df = ar39_df.loc[ar39_df.index.get_level_values("EventID").isin(random_ar39_events)]

efficiency_applied = {"ar41_eff": [], "total_rate": [], "multiplicity_list" : [], "PE_threshold_list" : []}

# efficiency_applied["PE_threshold_list"] = PE_threshold_list
# efficiency_applied["multiplicity_list"] = multiplicity_list

# loop over different PE threshold on single light guide
for PE_threshold in PE_threshold_list:

    #Apply the PE threshold
    PE_thr_applied_41 = ar41_df.apply(lambda d: [a for a in d if a > PE_threshold])
    PE_thr_applied_39 = ar39_df.apply(lambda d: [a for a in d if a > PE_threshold]) 

    #Drop empty rows
    PE_thr_applied_41 = PE_thr_applied_41[PE_thr_applied_41.apply(len) > 0]
    PE_thr_applied_39 = PE_thr_applied_39[PE_thr_applied_39.apply(len) > 0]

    #Use this for the strictest definition of time windowing - needs improvement
    PE_thr_applied_41 = PE_thr_applied_41.groupby(['EventID','time_bin']).sum()

    #Use this for the most lenient definition of time windowing i.e. all hits in a single event are summed with no loss
    #PE_thr_applied_41 = PE_thr_applied_41.groupby(['EventID']).sum()
    
    PE_thr_applied_39 = PE_thr_applied_39.groupby(['EventID']).sum()

    # loop over different multiplicity conditions
    for M_idx, M in enumerate(multiplicity_list):

        # Majority condition (M = 1 means "single light guide trigger" which corresponds to the case no majority activated)  
        above_majority_41 = PE_thr_applied_41.apply(lambda d: len(d) >= M)
        above_majority_39 = PE_thr_applied_39.apply(lambda d: len(d) >= M)

        #Drop rows which didn't pass the majority condition
        above_majority_41 = above_majority_41[above_majority_41==True]
        above_majority_39 = above_majority_39[above_majority_39==True]

        #At this point, above_majority contains a column of booleans for the previous cut, so we just sum the column
        #Edit: doing it this way does not account for the same event triggering multiple detections
        #So, we go back to the old way of doing it
        #Leaving the incorrect way of doing it here for legacy purposes
        #detected_41 = above_majority_41.sum()
        detected_41 = len(above_majority_41.index.get_level_values(0).unique())
        detected_39 = len(above_majority_39.index.get_level_values(0).unique())
        
        # this is to put the label only once
        label = None
        if PE_threshold == (1):
            label = f"majority: {M}"

        total_nof_ar40_events = len(np.array(ar41_df.index.get_level_values("EventID").unique()))
            
        ar41_efficiency = detected_41 / total_nof_ar40_events
        ar39_efficiency = detected_39 / total_nof_ar39_events

        ax.scatter(PE_threshold, ar41_efficiency * 100, marker = marker[M_idx], color = color[M_idx], label = label, linestyle = "", s = 50)
        ax.set_ylabel("$^{40}$Ar veto efficiency [%]")
        ax.set_xticks(PE_threshold_list)
        ax.set_yscale("linear")
        ax.set_ylim([0, 1.1e2])
        ax.legend()
        ax.grid(axis='y', zorder = 1000)
        #    guide_per_panel * panels * inn/out    guide_per_lid_slice * slice * inn/out * number_of_lids (2)
        ax.set_title(f"lateral: {lat * 12 * 2}  -  lids: {lid * 12 * 2 * 2}  -  PDE : {eff / 10}%")
        
        ar39_rate = opanalysis.get_Ar39_rate()
        ar41_rate = opanalysis.get_muon_rate() * total_nof_ar40_events / 10000000

        # print("muon rate:\t", opanalysis.get_muon_rate())
        # print("Ge77(m) with Ar40 events: ", total_nof_ar40_events)
        # print("total event rate:\t", ar41_rate)

        total_rate = ar41_efficiency * ar41_rate + ar39_efficiency * ar39_rate
        
        efficiency_applied["multiplicity_list"].append(M)
        efficiency_applied["PE_threshold_list"].append(PE_threshold)
        efficiency_applied["ar41_eff"].append(ar41_efficiency)
        efficiency_applied["total_rate"].append(total_rate)
        
        ax2.scatter(PE_threshold, total_rate, marker = marker[M_idx], color = color[M_idx], label = label, linestyle = "", s = 50)
        ax2.set_ylabel("$^{40}$Ar + $^{39}$Ar rate [Hz]")
        ax2.set_xticks(PE_threshold_list)
        ax2.set_yscale("log")
        ax2.legend()
        ax2.grid(axis='y', zorder = 1000)
        #    guide_per_panel * panels * inn/out    guide_per_lid_slice * slice * inn/out * number_of_lids (2)
        ax2.set_title(f"lateral: {lat * 12 * 2}  -  lids: {lid * 12 * 2 * 2}  -  PDE : {eff / 10}%")
    
plt.tight_layout()   

# output_path = "/lfs/l1/legend/users/morella/legend1k_simulation/argon_energy_deposit/shared/legend-simulation-analysis/opticalmaptemplate/analysis/output/"

#output_path = "/lfs/l1/legend/users/morella/legend1k_simulation/argon_energy_deposit/shared/legend-simulation-analysis/opticalmaptemplate/analysis/output/2025_CollabMeeting/efficiency/all_muon_events/"
output_path = "/lfs/l1/legend/users/morella/legend1k_simulation/argon_energy_deposit/shared/legend-simulation-analysis/opticalmaptemplate/analysis/output/2025_CollabMeeting/efficiency/high_Ar39_stat/higherM_data"

with open(os.path.join(output_path, "plots", f'efficiecy_and_rate_plot_{lat}_{lid}_{eff}_ge77.pkl'), 'wb') as f:
    pickle.dump([fig, fig2], f)         

output_df = pd.DataFrame(efficiency_applied)
output_df.to_hdf(os.path.join(output_path, "data", f'efficiecy_and_rate_plot_{lat}_{lid}_{eff}_ge77.h5'), key="efficiency_applied", mode="w")
 
fig.savefig(output_path  + "plots/" + f"Ar41_efficiency_plot_{lat}_{lid}_{eff}_ge77.pdf")
fig2.savefig(output_path + "plots/" + f"Ar41_rate_plot_{lat}_{lid}_{eff}_ge77.pdf")