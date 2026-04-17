import os
import json
import time
import subprocess

# Load configuration from JSON
with open("config.json", "r") as f:
    config = json.load(f)

# Get parameters from JSON
singularity_image  = config["singularity image"]
general_outdir     = config["general output"]
geometry_moderator = config["geometry moderator"]
radius             = config["radius"]
height             = config["height"]
width              = config["width"]
cryo_outer_radius  = config["cryo_outer_radius"] + 3 * 2 + 25 # + thickness * 2 walls + vacuumgap
cryo_outer_height  = config["cryo_outer_height"] + 3 * 2 + 25 # + thickness * 2 walls + vacuumgap

musun_path         = config["muon simulations"]["musun_path"]
muon_outdir        = config["muon simulations"]["output dir"]
end_idx            = config["muon simulations"]["musun files"]
beam_on            = config["muon simulations"]["beam_on"]

# Path variables
macros_dir = "macros"
output_dir = os.path.join(general_outdir, muon_outdir)

# Ensure necessary directories exist
os.makedirs(macros_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

start_idx = 0

# Loop to generate .mac files and submit jobs
for i in range(start_idx, end_idx + 1):
    file_id = f"{i:04d}"  # Format index to 4 digits (e.g., 0085)
    mac_filename = f"{macros_dir}/cosmo_{width}_{file_id}.mac"
    musun_file = f"{musun_path}/musun_gs_50k_{file_id}.dat"
    output_file = f"{output_dir}/muon_{file_id}.root"
    job_name = f"mu{file_id}"

    # Create .mac file
    with open(mac_filename, "w") as f:
        f.write("\n")
        f.write("/WLGD/detector/setGeometry baseline_large_reentrance_tube\n")
        f.write("/WLGD/event/saveAllEvents 0\n")
        f.write("/WLGD/runaction/WriteOutAllNeutronInfoRoot 1\n")
        f.write(f"/WLGD/detector/Cryostat_Radius_Outer {cryo_outer_radius}\n")
        f.write(f"/WLGD/detector/Cryostat_Height {cryo_outer_radius}\n")
        f.write("/WLGD/detector/With_Gd_Water 1\n")
        f.write(f"/WLGD/detector/With_NeutronModerators {geometry_moderator}\n")  # Read from JSON
        f.write("/WLGD/detector/Which_Material PMMA\n")
        f.write("/WLGD/detector/TurbineAndTube_zPosition 42\n")
        f.write(f"/WLGD/detector/TurbineAndTube_Radius {radius}\n")
        f.write(f"/WLGD/detector/TurbineAndTube_Height {height}\n")
        f.write(f"/WLGD/detector/TurbineAndTube_Width {width}\n")
        f.write("/run/initialize\n")
        f.write("/WLGD/generator/setGenerator Musun\n")
        f.write(f"/WLGD/generator/setMUSUNFile {musun_file}\n")
        f.write(f"/run/beamOn {beam_on}\n")

    # Command to execute in Singularity
    singularity_command = (
        f"singularity exec {singularity_image} bash -c "
        f"\"source ../env.sh && ./warwick-legend -o {output_file} -m {mac_filename}\""
    )

    # SLURM command
    slurm_command = [
        "sbatch",
        "--ntasks=1",
        "--nodes=1",
        "--get-user-env",
        "--nodelist=lfl08",
        f"--job-name={job_name}",
        "--wrap",
        f"bash -c '{singularity_command}'"
    ]

    # Submit job
    subprocess.run(slurm_command, check=True)

