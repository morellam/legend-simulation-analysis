import opanalysis
import os
import math
import uproot
import random
import itertools
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

from iminuit import Minuit
from iminuit.cost import LeastSquares

output_path = "/lfs/l1/legend/users/morella/legend1k_simulation/argon_energy_deposit/shared/legend-simulation-analysis/opticalmaptemplate/analysis/output/"

# our line model, unicode parameter names are supported :)
def line(x, m, q):
    return m * x + q

sotope_dict = {"Ar": 18, "H": 1}

energy_map_file = "/lfs/l1/legend/users/cbarton/simulations/campaigns/opmapprocessing/24-12-08-Ar41+H2-SmallMod-momentumfixed/newoutput/ArCapSims.root"

energy_df = pd.DataFrame()
energy_tf = uproot.open(energy_map_file)["Steps"]
energy_df["X"]     = energy_tf["X"].array(library="np")
energy_df["Y"]     = energy_tf["Y"].array(library="np")
energy_df["Z"]     = energy_tf["Z"].array(library="np")
energy_df["evtID"] = energy_tf["EventID"].array(library="np")
energy_df["EDep"]  = energy_tf["EDep"].array(library="np")
energy_df["ProtonNumber"]  = energy_tf["ProtonNumber"].array(library="np")

energy_df["R"] = np.sqrt(energy_df.X * energy_df.X + energy_df.Y * energy_df.Y)

iso = "Ar"
isotope_dict = {"Ar": 18, "H": 1}

energy_df = energy_df[energy_df.ProtonNumber == isotope_dict[iso]]

path = "/lfs/l1/legend/users/morella/legend1k_simulation/argon_energy_deposit/shared/legend-simulation-analysis/opticalmaptemplate/analysis/jobs/data/window_timing_Ar41_small_moderator"
file_type = ".pkl"

PDE = sorted(np.unique([int(file.split("_")[3]) for file in os.listdir(path) if file_type in file]))
lateral_guides = sorted(np.unique([int(file.split("_")[4]) for file in os.listdir(path) if file_type in file]))
lid_guides = sorted(np.unique([int(file.split("_")[5].split(".")[0]) for file in os.listdir(path) if file_type in file]))
isotope = np.unique([file.split("_")[-1].split(".")[0] for file in os.listdir(path) if file_type in file])

light_yield_list = []

for eff in tqdm(PDE):
    for i in ["Ar"]:
        for n_lat in tqdm(lateral_guides):
            for n_lid in tqdm(lid_guides):

                tmp_energy = 0
                data = 0
                evt_list = 0
                energy = 0
                pe = 0

                data = pd.read_pickle(os.path.join(path, f"pe_per_panel_{eff}_{n_lat}_{n_lid}_{i}.pkl")) 

                pe_per_event = data.apply(np.sum).groupby(level = 0).sum()
                pe_uncertainty = np.sqrt(pe_per_event)

                evt_list = pe_per_event.index

                energy_df = energy_df[energy_df.evtID.isin(evt_list)]
                energy = energy_df.groupby("evtID").EDep.sum()
                energy = energy.reindex(evt_list, fill_value = 0) / 1000 # MeV          

                res = 0.15
                quantiles = 0

                mean_energy = []
                mean_pe = []
                
                median_energy = []
                median_pe = []
                
                std_energy = []
                std_pe = []
                low_pe_median = []
                low_pe_mean = []
                high_pe_median = []
                high_pe_mean = []
                
                sigma = 0.6827
                low = (1-sigma)/2
                high = (1+sigma)/2

                # compute energy range in which we'll plot data
                energy_range = np.arange(0,1e6,20)

                # multiply energy range by resolution
                resolution_bin_width = energy_range * res
                
                # cumulative sum over the energies to get absolute scale considering 
                # the variable bin width computed at previous step
                bins = np.cumsum(resolution_bin_width)

                # just take the initial (continous) energy and check to which bin it belong
                # the digitize function returns for each energy the index of the bin it belongs
                bin_idx = np.digitize(energy, bins)

                # loop over all unique bin_idx (i.e. unique energy bins)
                # the group together by energy 
                for n in np.unique(bin_idx):
                    
                    interval_energy = (energy.values)[bin_idx == n]
                    interval_pe = (pe_per_event.values)[bin_idx == n]
                    
                    mean_energy.append(np.mean(interval_energy))
                    median_energy.append(np.median(interval_energy))
                
                    mean_pe.append(np.mean(interval_pe))
                    median_pe.append(np.median(interval_pe))
                
                    std_energy.append(np.std(interval_energy) / np.sqrt(len(interval_energy)))
                    
                    if len(interval_pe) == 1:
                        std_pe.append(np.sqrt(np.mean(interval_energy)))
                    else:
                        std_pe.append(np.std(interval_pe) / np.sqrt(len(interval_energy)))
                
                    quantiles = np.quantile(interval_pe, [low, high])
                
                    low_pe_median.append(np.median(interval_pe) - quantiles[0])
                    low_pe_mean.append(np.mean(interval_pe) - quantiles[0])
                
                    high_pe_median.append(quantiles[1] - np.median(interval_pe))
                    high_pe_mean.append(quantiles[1] - np.mean(interval_pe))

                energy_at_99 = {"Ar" : 1000, "H": 200}

                n = np.where(bins < energy_at_99[iso])[0][-1] + 1

                least_squares = LeastSquares(mean_energy[:n], mean_pe[:n], std_pe[:n], line)
                minuit = Minuit(least_squares, m=0, q=0)  
                minuit.migrad()  # finds minimum of least_squares function
                minuit.hesse()   # accurately computes uncertainties
                m = minuit.values[0]
                q = minuit.values[1]
                
                figura, asse = plt.subplots(1,2,figsize = (10,4))
                
                x = np.arange(1,15000,1)
                
                asse[0].errorbar(mean_energy, mean_pe, xerr = std_energy, yerr = std_pe, linestyle = "", marker = ".", color = "black")
                asse[0].plot(x, m*x + q, color = "red", label = "linear fit", zorder = 10)
                
                asse[0].set_xlabel("energy [MeV]")
                
                asse[0].set_ylabel("energy [pe]")
                
                # display legend with some fit info
                fit_info = [
                    f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {minuit.fval:.2f} / {minuit.ndof:.0f} = {minuit.fmin.reduced_chi2:.2f}",
                ]
                for p, v, e in zip(minuit.parameters, minuit.values, minuit.errors):
                    fit_info.append(f"{p} = ${v:.2f} \\pm {e:.2f}$")
                
                asse[0].legend(title="\n".join(fit_info), frameon=False)
                
                l = math.ceil(mean_energy[n-1]) + 0.05 * math.ceil(mean_energy[n-1])

                asse[0].set_xlim([1,l])
                asse[0].set_ylim([1,m*l])
                
                
                residuals = (mean_pe[:n] - (m*np.array(mean_energy[:n]) + q))
                mean = np.mean(residuals)
                std_on_the_mean = np.std(residuals)/np.sqrt(len(residuals))
                label = "$\mu\pm\sigma=(${:.0f}".format(mean) + "$\pm$" + "{:.0f}) pe".format(std_on_the_mean)
                asse[1].hist(residuals, bins = 50, histtype = "step", label = label)
                asse[1].set_xlabel("residuals [pe]")
                asse[1].set_ylabel("counts")
                plt.suptitle(f"{int(res*100)}% resolution")
                plt.legend()
                plt.savefig(os.path.join(output_path, "light_yield", f"light_yield_fit_{i}_{eff}_{n_lat}_{n_lid}.pdf"))

                row = {"eff": eff, "isotope": i, "n_lat_bar": n_lat, "n_lid_bar": n_lid, "ly": m}
                light_yield_list.append(row)


light_yield_data = pd.DataFrame(light_yield_list)
light_yield_data.to_hdf(os.path.join(output_path, "light_yield", f'light_yield.h5'), key="light_yield", mode="w")