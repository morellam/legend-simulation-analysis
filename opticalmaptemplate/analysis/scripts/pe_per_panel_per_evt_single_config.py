import os
import math
import pickle
import uproot
import random
import argparse
import itertools
import numpy as np
import pandas as pd
from functools import reduce
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import opanalysis

def nan_to_zero(row):
    
    if isinstance(row, np.ndarray):
        return row
    
    else:
        return np.array([0])

def unstack_and_restack_time(multiindex):
    temp = multiindex.unstack('time')
    print(temp)
    temp = temp.reindex(new_index).apply(nan_to_zero) 
    print(temp)
    print(temp.stack('time'))
    return temp.stack('time')

parser = argparse.ArgumentParser(description = 'script to generate plot and data with panels hit per muon event')
parser.add_argument('-i', '--isotope',    required = True, help = "specify isotope, available options: H, Ar, Ar_and_H")
parser.add_argument('-n', '--nfiles',     required = True, help = "specify number of files to open (int or \"all\")")
parser.add_argument('-e', '--efficiency', required = True, help = "specify photon detection efficinecy of a single light guide")
parser.add_argument('-l', '--lateral',    required = True, help = "specify number of lateral light guides")
parser.add_argument('-d', '--lid',        required = True, help = "specify number of lids light guides")

args = parser.parse_args()
if vars(args)["nfiles"] != "all":
    n_files = int(vars(args)["nfiles"])
else:
    n_files = vars(args)["nfiles"]

isotope   = vars(args)["isotope"]
eff       = float(vars(args)["efficiency"])
n_lat_bar = int(vars(args)["lateral"])
n_lid_bar = int(vars(args)["lid"])

isotope_dict = {"Ar": 18, "H": 1}

name_tag = ""

if isotope == "Ar39":
    # new_path = "/lfs/l1/legend/users/cbarton/simulations/campaigns/opmapprocessing/24-12-09-Ar39-SmallMod-momentumfixed/output/reformatted-10us"
    new_path = "/lfs/l1/legend/users/cbarton/simulations/campaigns/opmapprocessing/25-03-09-Ar39-SmallMod-Xenon100ppm/output/reformatted-10us/"
    name_tag = "reformatted"
else:
    # new_path = "/lfs/l1/legend/users/cbarton/simulations/campaigns/opmapprocessing/24-12-08-Ar41+H2-SmallMod-momentumfixed/output"
    new_path = "/lfs/l1/legend/users/cbarton/simulations/campaigns/opmapprocessing/25-03-12-Ar41+H2-SmallMod-Xenon100ppm/output/appliedmap"
    name_tag = "appliedmap"


panel_df = opanalysis.load_data(new_path, "panel", n_files, name_tag)
lid_df = opanalysis.load_data(new_path, "lid", n_files, name_tag)


if isotope not in ["Ar_and_H", "Ar39"]:
    panel_df = panel_df[panel_df.protonnumber == isotope_dict[isotope]]
    lid_df   = lid_df[lid_df.protonnumber == isotope_dict[isotope]]


lid_df.rename(columns={"xband" : "zband"}, inplace = True)

array1 = sorted(panel_df.EventID.unique())
array2 = sorted(lid_df.EventID.unique())

only_in_array1 = np.setdiff1d(array1, array2)
only_in_array2 = np.setdiff1d(array2, array1)
common_elements = np.intersect1d(array1, array2)

evt_list = np.unique(np.concatenate((only_in_array1, only_in_array2, common_elements)))

all_possible_panels = np.arange(1,73,1)
new_index = list(itertools.product(evt_list, all_possible_panels))


pe_data = {"eff" : [], "n_lid_bar" : [], "n_lat_bar": [], "pe": []} 

if isotope != "Ar39":
    photons_per_panel_inner_lateral = opanalysis.slicing_windowing(panel_df[panel_df.panel > 12],  300, n_lat_bar, 10, eff, 0.00001)
    photons_per_panel_outer_lateral = opanalysis.slicing_windowing(panel_df[panel_df.panel <= 12], 300, n_lat_bar, 10, eff, 0.00001)
else:
    photons_per_panel_inner_lateral = opanalysis.slicing_windowing(panel_df[panel_df.panel > 12],  300, n_lat_bar, 10, eff)
    photons_per_panel_outer_lateral = opanalysis.slicing_windowing(panel_df[panel_df.panel <= 12], 300, n_lat_bar, 10, eff)

# print(photons_per_panel_inner_lateral)

# photons_per_panel_inner_lateral = photons_per_panel_inner_lateral.reindex(new_index).apply(nan_to_zero)
# photons_per_panel_outer_lateral = photons_per_panel_outer_lateral.reindex(new_index).apply(nan_to_zero)

outer_top_lids = (lid_df.panel >= 25) & (lid_df.panel <= 36)
outer_bot_lids = (lid_df.panel >= 61) & (lid_df.panel <= 72)
inner_top_lids = (lid_df.panel >= 37) & (lid_df.panel <= 48)
inner_bot_lids = (lid_df.panel >= 49) & (lid_df.panel <= 60)

if isotope != "Ar39":
    photons_per_panel_inner_lids = opanalysis.slicing_windowing(lid_df[inner_top_lids | inner_bot_lids], 52, n_lid_bar, 10, eff, 0.00001)
    photons_per_panel_outer_lids = opanalysis.slicing_windowing(lid_df[outer_top_lids | outer_bot_lids], 52, n_lid_bar, 10, eff, 0.00001)
else:
    photons_per_panel_inner_lids = opanalysis.slicing_windowing(lid_df[inner_top_lids | inner_bot_lids], 52, n_lid_bar, 10, eff)
    photons_per_panel_outer_lids = opanalysis.slicing_windowing(lid_df[outer_top_lids | outer_bot_lids], 52, n_lid_bar, 10, eff)


# photons_per_panel_inner_lids = photons_per_panel_inner_lids.reindex(new_index).apply(nan_to_zero)
# photons_per_panel_outer_lids = photons_per_panel_outer_lids.reindex(new_index).apply(nan_to_zero)

dfs = [photons_per_panel_inner_lateral, photons_per_panel_outer_lateral, photons_per_panel_inner_lids, photons_per_panel_outer_lids]

pe = reduce(lambda left, right: left.add(right, fill_value=0), dfs)

# pe.to_pickle(f"data/2025_March_CollabMeeting/Ar39/pe_per_panel_{int(eff*1000)}_{n_lat_bar}_{n_lid_bar}_{isotope}.pkl")
pe.to_pickle(f"data/2025_March_CollabMeeting/Ar_and_H/pe_per_panel_{int(eff*1000)}_{n_lat_bar}_{n_lid_bar}_{isotope}.pkl")

#pe = photons_per_panel_inner_lateral + photons_per_panel_outer_lateral + photons_per_panel_inner_lids + photons_per_panel_outer_lids