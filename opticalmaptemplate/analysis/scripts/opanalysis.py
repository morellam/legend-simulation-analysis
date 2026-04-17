from legend_plot_style import LEGENDPlotStyle as lps
import os
import uproot
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt


def load_data(path, shield_surface, files_to_open):
    """
    Load data from files into unique array

    path: absolute path to the files location
    shield_surface: either "panel" or "lid"
    files_to_open: either string "all" to load data from all the files or int with number of files
    """

    files = sorted([file for file in os.listdir(path) if "appliedmap" in file])

    if files_to_open != "all":
        files = files[:files_to_open]

    df = pd.DataFrame()

    for file in tqdm(files):
        try:
            tmp_df = pd.DataFrame()

            tf = uproot.open(os.path.join(path, file))[f"{shield_surface}hits"]

            for var in tf.keys():
                tmp_df[var]  = tf[var].array(library="np")
            
        except:
            print(f"skipping {file}")
        
        df = pd.concat([tmp_df, df])
        
    evt_list = sorted(df.EventID.unique())
    total_nof_events = len(evt_list)

    return df


def get_cryostat_volume():
    
    cryostat_radius = 3.5 # m
    cryostat_height = 7 # m
    cryostat_volume = np.pi * cryostat_radius * cryostat_radius * cryostat_height # m3    
    return cryostat_volume


def unpack(lista):
    return [a for b in list(lista) for a in b ]


def get_RT_volume():

    RT_radius = 0.95 # m
    RT_height = 4    # m, 4m inside the cryostat
    RT_volume = np.pi * RT_radius * RT_radius * RT_height
    return RT_volume


def get_moderator_volume():
    n_panels = 12

    panel_height    = 3   # m
    panel_width     = 1   # m
    panel_thickness = 0.1 # m
    mod_radius      = 2   # m

    RT_radius = 0.95 # m

    panel_volume = panel_height * panel_width * panel_thickness # m3
    bottom_cap   = np.pi * (mod_radius + panel_thickness) * (mod_radius + panel_thickness) * panel_thickness # m3
    top_cap      = bottom_cap - np.pi * RT_radius * RT_radius * panel_thickness  # m3

    mod_volume = panel_volume * n_panels + bottom_cap + top_cap # m3
    return mod_volume


def get_AAr_volume():

    cryostat_volume = get_cryostat_volume()
    RT_volume       = get_RT_volume()
    mod_volume      = get_moderator_volume()

    AAr_volume = cryostat_volume - RT_volume - mod_volume
    return AAr_volume


def get_Ar39_rate():

    """
    assuming a volume of AAr and given the activity of Ar39 in natural argon from literature
    compute rate of Ar39 in Hz
    """

    lAr_density = 1396 # kg/m3
    AAr_volume = get_AAr_volume()

    AAr_mass = lAr_density * AAr_volume # kg

    ar39_activity = 1.01 # Bq / kg
    ar39_rate = ar39_activity * AAr_mass # Hz

    return ar39_rate


def get_expected_ar39_in_window(window = 1e-5):
    """
    compute how many events of Ar39 you expect in a given time window
    """

    ar39_rate = get_Ar39_rate()

    return ar39_rate * window


def get_muon_rate():
    """
    muon rate is specific of MUSUN sim
    it won't be correct for other MUSUN sims
    """

    muon_rate = 263 # muons / hour

    return muon_rate / 3600 # Hz


def get_guides(H, bar_width, n_bar):

    """
    H: height of the PMMA panel (where light guides will be attached)
    bar_width: light guide width
    n_bar: number of light guides bar to attach

    remember that bar_width and n_bar should be such that bar_width * n_bar = H
    output 
    light guides position in bins, possible empyt spaces
    """

    # Check for invalid input where n_bar is zero
    if n_bar == 0:
        return 0, [0], [0]
    
    if bar_width * n_bar > H:
        raise ValueError('A very specific bad thing happened.')
    # total free space between light guides
    S = H - n_bar*bar_width
    
    # single space between two adjacent light guides
    s = S // (n_bar - 1)

    # residual space (if != 0 you basically split it in 2 and add it at the top and bottom to make guides fit into the panel and be evenly spaced)
    residual_space = H - n_bar*bar_width - s*(n_bar - 1)

    slices = []
    guide_position = []
    
    # if residual space is not zero, skip corresponding zbands
    if residual_space:
        slices.append(0)

        slices.append(residual_space)
        guide_position.append(0)
    
    # start always with a light guide
    slices.append(residual_space+bar_width*2)
    guide_position.append(1)

    # increase by residual_space (if any, otherwise 0) + bar_width * 2 (because zband width = 0.5cm)
    counter = residual_space+bar_width*2
    
    # loop over the entire height H of the panel
    for i in range(n_bar-1):
        slices.append(counter+s*2)
        guide_position.append(0)

        slices.append(counter + s*2 + bar_width*2)
        guide_position.append(1)

        counter += s*2+bar_width*2

    # finish with eventual additional residual space
    if residual_space:
        slices.append(counter + residual_space)
        guide_position.append(0)
        
    return residual_space, slices, guide_position


def slicing(df, surface_length, n_bar, bar_width, detection_efficiency, time_window = 0):

    """
    take data from histogrammed panel surface and slice it into a given number of light guides according to
    the their size and number; then apply single light guide Photon Detection Efficiency (PDE)
    output is the number of PE detected per light guide per panel per event
    """
    
    residual_space, slices, guide_position = get_guides(surface_length, bar_width, n_bar)
    
    if time_window:
        bins = np.arange(0, 0.005, time_window)
        df['time_bin'] = pd.cut(df['time'], bins=bins)
        grouped_zband = df.groupby(["EventID", "panel", "time_bin"]).zband
        grouped_zband_tolist = grouped_zband.apply(lambda d: list(d)).dropna()
        grouped_df = grouped_zband_tolist.apply(np.sum, axis = 0)
    else:
        grouped_df = df.groupby(["EventID", "panel"]).zband.sum()
        sliced_df = grouped_df.apply(lambda d: np.split(d, slices))

    sliced_df = grouped_df.apply(lambda d: np.split(d, slices))

    if residual_space:
        sliced_df = sliced_df.apply(lambda d: [np.sum(array)*pos for array, pos in zip(d[1:-1], guide_position)])
    else:
        sliced_df = sliced_df.apply(lambda d: [np.sum(array)*pos for array, pos in zip(d, guide_position)])

    sliced_df = sliced_df.apply(lambda d: np.random.binomial(d, detection_efficiency))

    return sliced_df


def slice_panel(df, surface_length, n_bar, bar_width, detection_efficiency):
    """
    take output from function "slicing"
    and sum over all PE detected in a single panel/surface
    """
        
    sliced_df = slicing(df, surface_length, n_bar, bar_width, detection_efficiency)

    photons_per_panel = sliced_df.apply(lambda d: sum(d))
    photons_per_panel = photons_per_panel[photons_per_panel != 0]
    
    return photons_per_panel  


def compute_contigous_surface(panels_hit):
    """
    Function to compute the number of contiguous panels being hit.
    - 0: no events actually detected
    - 1: only one panel was hit
    > 1: number of contigous panels (between 2 and max number of contiguous surfaces)

    WARNING: at the moment, surfaces outside the moderator are numbered [1,12] and inside [13,24]
    surfaces 12 and 13 might be considered as contigous, 
    please be careful and shift the inner guides to much more different values 
    (e.g. +100) when passing data to this function 
    """
    
    if panels_hit == 0:
        return 0

    diff = np.diff(panels_hit)
    stop = np.where(diff != 1)[0]

    nof_contiguous_panels = []

    # if stop array has length 0, means either that we have only panel
    # or that all panels in the initial array are contiguous
    
    #print(stop)
    
    if len(stop > 0):
        
        position = 0

        for s in stop:
            
            nof_contiguous_panels.append(len(panels_hit[position : s + 1]))
            position = s + 1

        nof_contiguous_panels.append(len(panels_hit[position : ]))
        
    else:

        nof_contiguous_panels.append(len(panels_hit))

 
    return_list = nof_contiguous_panels
    
    if (panels_hit[0] == 1) & (panels_hit[-1] == 12):

        return_list = []
        
        if nof_contiguous_panels[1:-1]: 
        
            return_list = nof_contiguous_panels[1:-1]
            return_list.append(nof_contiguous_panels[0] + nof_contiguous_panels[-1])
        
        else:
                
            return_list.append(nof_contiguous_panels[0] + nof_contiguous_panels[-1])

    return return_list


def distance_point_line(point, line):
    """
    Compute the distance between a point and a line.
    In this case the absolute value of the numerator is missing so that 
    from the sign of the distance we understand if the point
    is below (d < 0) or above (d > 0) the line.
    
    point: tuple (x0, y0) point's coordinates
    line: tuple (A, B, C) coefficients of the line in the form Ax + By + C = 0
    
    Return the distance between the point and the line.
    """
    x0, y0 = point
    A, B, C = line
    
    # Calcola la distanza utilizzando la formula
    distance = A * x0 + B * y0 + C / np.sqrt(A**2 + B**2)
    
    return distance


def geometrical_median(points, tol=1e-5):
    """
    Calculate the geometric median (Fermat-Weber point) of a set of 2D points.
    
    points: array of shape (n, 2), where n is the number of points
    tol: tolerance for convergence
    """
    points = np.asarray(points)
    median = np.mean(points, axis=0)
    
    while True:
        distances = np.linalg.norm(points - median, axis=1)
        non_zero = distances > tol
        if not np.any(non_zero):
            break
        
        weights = 1 / distances[non_zero]
        new_median = np.sum(points[non_zero] * weights[:, np.newaxis], axis=0) / np.sum(weights)
        
        if np.linalg.norm(new_median - median) < tol:
            break
        
        median = new_median
    
    return median


def plot_values(filename, key, values, ylim = [0,10], marker = {"style": ".", "color": "#1A2A5B", "size": 10}, ax = None):
    """
    Function to plot single valued variable from optical analysis (e.g. light yield)
    as a function of the different experimental configuration (PDE, light guides on the lid, 
    light guides on the lateral panels).

    filename: absolute path + filename of the hdf5 file where output of the analysis is stored
    key: dataframe key saved inside the hdf5 file
    values: name of the column in the dataframe to plot
    ylim: (optional) specify y-axis lim
    """
    
    data = pd.read_hdf(path_or_buf = filename, key = key)

    if ax is None:
        fig, ax = plt.subplots(figsize = (18,3))
    else:
        fig = None

    grouped_data = data.groupby(["eff", "n_lat_bar","n_lid_bar"])[values].mean()

    ax = grouped_data.plot(linestyle="", marker=marker["style"], color=marker["color"], markersize = marker["size"])

    xticks_labels = [f'{n_lid}' for eff, n_lat, n_lid in grouped_data.index]
    ax.set_xticks(range(len(xticks_labels)))
    ax.set_xticklabels(xticks_labels, rotation=0, ha = "center")
    ax.set_xlabel('light guides on the lid')
    ax.set_ylim(ylim)

    subset_color = "#07A9FF"
    shift = -0.5
    lat_position = 0

    lid_options = len(data.n_lid_bar.unique())
    lat_options = len(data.n_lat_bar.unique())
    eff_options = len(data.eff.unique())

    for eff_idx in range(eff_options):
        for lat_idx in range(lat_options):
            
            lat_position = shift + lid_options + lat_idx * lid_options + eff_idx * lid_options * lat_options
            plt.axvline(lat_position, linestyle = "--", color = subset_color, linewidth = 1, zorder = 1)
            text = f"{data.n_lat_bar.unique()[lat_idx]}"
            zposition = ylim[1] - 0.2
            valign = "top"
            ax.text(lat_position - 0.01, zposition, text, rotation=90, verticalalignment=valign, horizontalalignment='right', color = subset_color, fontsize = 13)

        eff_pos = shift + (eff_idx + 1) * lid_options * lat_options
        plt.axvline(eff_pos, color = "black", linestyle = "-", zorder = 2)
        text = f"PDE: {data.eff.unique()[eff_idx]*100} %"
        ax.text(eff_pos - 0.01, ylim[1], text, rotation=0, verticalalignment='bottom', horizontalalignment='right', fontsize = 13)

    plt.tight_layout()
    
    return fig, ax
