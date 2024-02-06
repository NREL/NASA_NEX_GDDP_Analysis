#!/bin/bash
#SBATCH --job-name=climate_analysis
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=36
#SBATCH --account=seas
#SBATCH --mail-user=jordan.cox@nrel.gov
#SBATCH --mail-type=ALL

module purge 
module load conda
module load openmpi/4.1.0/gcc-8.4.0

conda activate climate_analysis

cd /projects/seas/NASA-NEX_analysis/

srun -n 54 python3 main.py


