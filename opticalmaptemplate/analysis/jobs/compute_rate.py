import subprocess

# detection_efficiency_list = [0.005, 0.01, 0.02, 0.05, 0.1, 1]
# nof_lateral_guides_list = [5, 12, 15, 20, 30]
# nof_lid_guides_list = [0, 2, 3, 5]

detection_efficiency_list = [0.005, 0.01, 0.02, 0.05, 0.1]
nof_lateral_guides_list = [12]
nof_lid_guides_list = [2]

# detection_efficiency_list = [1]
# nof_lateral_guides_list = [30]
# nof_lid_guides_list = [5]

# loop over single light guide PDE
for eff in detection_efficiency_list:

    # loop over number of light guides per lateral panel
    for n_lat_bar in nof_lateral_guides_list:

        # loop over number of light guides per lid slice
        for n_lid_bar in nof_lid_guides_list:

            # Command to be executed in the terminal
            command = f"sbatch submit.sh {eff} {n_lat_bar} {n_lid_bar}" 
            
            # Execute the command in the terminal
            try:
                subprocess.run(command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing command '{command}': {e}")