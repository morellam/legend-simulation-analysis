import os
import math
import pickle
import uproot
import random
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, Normalize
import opanalysis

parser = argparse.ArgumentParser(description = 'script to generate plot and data with panels hit per muon event')
parser.add_argument('-i', '--isotope', required = True, help = "specify isotope, available options: H, Ar, Ar_and_H")
parser.add_argument('-n', '--nfiles', required = True, help = "specify number of files to open (int or \"all\")")

args = parser.parse_args()
if vars(args)["nfiles"] != "all":
    n_files = int(vars(args)["nfiles"])
else:
    n_files = vars(args)["nfiles"]
isotope = vars(args)["isotope"]

isotope_dict = {"Ar": 18, "H": 1}

new_path = "/lfs/l1/legend/users/cbarton/simulations/campaigns/opmapprocessing/24-07-18-Ar41+H2-350mwlsrcryo-allmaps/output"
files = [file for file in sorted(os.listdir(new_path)) if "appliedmap" in file]

panel_df = opanalysis.load_data(new_path, "panel", n_files)
lid_df = opanalysis.load_data(new_path, "lid", n_files)

if isotope != "Ar_and_H":
    panel_df = panel_df[panel_df.protonnumber == isotope_dict[isotope]]
    lid_df   = lid_df[lid_df.protonnumber == isotope_dict[isotope]]

lid_df.rename(columns={"xband" : "zband"}, inplace = True)
lid_df["panel"] = lid_df.panel + 24

array1 = sorted(panel_df.EventID.unique())
array2 = sorted(lid_df.EventID.unique())

only_in_array1 = np.setdiff1d(array1, array2)
only_in_array2 = np.setdiff1d(array2, array1)
common_elements = np.intersect1d(array1, array2)

evt_list = np.unique(np.concatenate((only_in_array1, only_in_array2, common_elements)))

distances_data = {"eff" : [], "n_lid_bar" : [], "n_lat_bar": [], "distance_naiv": [], "distance_geom": []} 
panels_hit_per_evt_data = {"eff" : [], "n_lid_bar" : [], "n_lat_bar": [], "lateral_inner": [], "lids_inner": [], "lateral_outer": [], "lids_outer" : []} 

detection_efficiency_list = [0.005, 0.01, 0.1, 1]

for eff in detection_efficiency_list:

    nof_lateral_guides_list = [5, 12, 20, 30]
    nof_lid_guides_list = [0,2,3,5]

    l_lat = len(nof_lateral_guides_list)
    l_lid = len(nof_lid_guides_list)

    fig, ax = plt.subplots(l_lat,l_lid,figsize = (6*l_lat,4*l_lid))

    for lat_bar_idx, n_lat_bar in enumerate(tqdm(nof_lateral_guides_list)):

        photons_per_panel_inner_lateral = opanalysis.slice_panel(panel_df[panel_df.panel > 12],  300, n_lat_bar, 10, eff)
        photons_per_panel_outer_lateral = opanalysis.slice_panel(panel_df[panel_df.panel <= 12], 300, n_lat_bar, 10, eff)

        photons_per_panel_inner_lateral = photons_per_panel_inner_lateral.groupby(level=0).size()
        photons_per_panel_outer_lateral = photons_per_panel_outer_lateral.groupby(level=0).size()

        photons_per_panel_inner_lateral = photons_per_panel_inner_lateral.reindex(evt_list, fill_value=0)
        photons_per_panel_outer_lateral = photons_per_panel_outer_lateral.reindex(evt_list, fill_value=0)
        
        for lid_bar_idx, n_lid_bar in enumerate(tqdm(nof_lid_guides_list)):
            # divide panels into outer and inner and then divide single panel
            # into n_bar light guides and apply to each guide PDE eff
    
            outer_top_lids = (lid_df.panel >= 1 + 24) & (lid_df.panel <= 12 + 24)
            outer_bot_lids = (lid_df.panel >= 37 + 24) & (lid_df.panel <= 48 + 24)
            inner_top_lids = (lid_df.panel >= 13 + 24) & (lid_df.panel <= 24 + 24)
            inner_bot_lids = (lid_df.panel >= 25 + 24) & (lid_df.panel <= 36 + 24)
            
            photons_per_panel_inner_lids    = opanalysis.slice_panel(lid_df[inner_top_lids | inner_bot_lids], 52, n_lid_bar, 10, eff)
            photons_per_panel_outer_lids    = opanalysis.slice_panel(lid_df[outer_top_lids | outer_bot_lids], 52, n_lid_bar, 10, eff)
    
            photons_per_panel_inner_lids    = photons_per_panel_inner_lids.groupby(level=0).size()   
            photons_per_panel_outer_lids    = photons_per_panel_outer_lids.groupby(level=0).size()   
               
            photons_per_panel_inner_lids    = photons_per_panel_inner_lids.reindex(evt_list, fill_value=0)
            photons_per_panel_outer_lids    = photons_per_panel_outer_lids.reindex(evt_list, fill_value=0)
    
            inn = photons_per_panel_inner_lateral + photons_per_panel_inner_lids
            out = photons_per_panel_outer_lateral + photons_per_panel_outer_lids
    
            panels_hit_per_evt_data["eff"].append(eff)
            panels_hit_per_evt_data["n_lid_bar"].append(n_lid_bar)
            panels_hit_per_evt_data["n_lat_bar"].append(n_lat_bar)
            panels_hit_per_evt_data["lateral_inner"].append(photons_per_panel_inner_lateral)
            panels_hit_per_evt_data["lids_inner"].append(photons_per_panel_inner_lids)
            panels_hit_per_evt_data["lateral_outer"].append(photons_per_panel_outer_lateral)
            panels_hit_per_evt_data["lids_outer"].append(photons_per_panel_outer_lids)
    
            n = 36
            
            h = ax[lat_bar_idx][lid_bar_idx].hist2d(inn, out, range = [[0,n], [0,n]], bins = n, vmin = 0)
            plt.colorbar(h[3])
            
            naiv_median = (np.median(inn), np.median(out))
            ax[lat_bar_idx][lid_bar_idx].plot(naiv_median[0], naiv_median[1], color = "red", marker = "*", markersize = 10)
    
            combined_array = [[a, b] for a, b in zip(inn, out)]
            geom_median = opanalysis.geometrical_median(combined_array)
            ax[lat_bar_idx][lid_bar_idx].plot(geom_median[0], geom_median[1], color = "#07A9FF", marker = "*", markersize = 10)
    
            line = (-1, 1, 0)  # Line equation: y - x = 0
            distance_geom = opanalysis.distance_point_line(geom_median, line)
            distance_naiv = opanalysis.distance_point_line(naiv_median, line)
    
            distances_data["eff"].append(eff)
            distances_data["n_lid_bar"].append(n_lid_bar)
            distances_data["n_lat_bar"].append(n_lat_bar)
            distances_data["distance_naiv"].append(distance_naiv)
            distances_data["distance_geom"].append(distance_geom)
    
            ax[lat_bar_idx][lid_bar_idx].set_title(f"{n_lat_bar} (lateral) - {n_lid_bar} (lids)")
            ax[lat_bar_idx][lid_bar_idx].set_xlabel("inner panels hit")
            ax[lat_bar_idx][lid_bar_idx].set_ylabel("outer panels hit")
            ax[lat_bar_idx][lid_bar_idx].set_xticks(np.arange(0,40,5))
            ax[lat_bar_idx][lid_bar_idx].set_yticks(np.arange(0,40,5))
            ax[lat_bar_idx][lid_bar_idx].axis('scaled')
            
            t = np.arange(0, n, 0.1)
            ax[lat_bar_idx][lid_bar_idx].plot(t,t, color = "red")
    
        plot_name = f"plots/panels_hit_per_evt/{isotope}_lateral_and_lids_{int(eff*1000)}"
        fig.suptitle(f"PDE:{eff*100}%")
        fig.tight_layout()
        plt.savefig(plot_name + ".png")
        plt.savefig(plot_name + ".pdf")
    
        with open(plot_name + ".pkl", 'wb') as file:
           pickle.dump(fig, file)


distance_data = pd.DataFrame(distances_data)
all_data = pd.DataFrame(panels_hit_per_evt_data)

with pd.HDFStore(f"plots/panels_hit_per_evt/panels_hit_per_evt_{isotope}.h5") as store:
   store.put('distances', distance_data)
   store.put('all', all_data)