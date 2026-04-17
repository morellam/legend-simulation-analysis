#!/bin/bash

### Job configuration starts here #############################################

# Export all current environment variables to the job
#SBATCH --get-user-env

# One task per node (single-threading)
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=200G

# SBATCH --exclude=lfl13,lfl12,lfl00,lfl03,lfl14,lfl15
# SBATCH --exclude=lfl14,lfl15
# SBATCH --nodelist=lfl18
# SBATCH --mail-type=FAIL
# SBATCH --mail-user=michele.morella@gssi.it

# Request x minutes of runtime - the job will be killed if it exceeds this
#SBATCH --time=07-00

### Commands to run the program start here ####################################

# singularity exec /lfs/l1/legend/software/apptainer/legendexp_legend-software_latest.sif /bin/bash run_py.sh $1 $2
# cenv legend-software /bin/bash run_py.sh $1 $2 $3 $4 $5
cenv legend-software /bin/bash run_py.sh $1 $2 $3 
