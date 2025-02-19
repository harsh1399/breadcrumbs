#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=u1413898@umail.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o assignment_1-%j
#SBATCH --export=ALL
source ~/miniconda3/etc/profile.d/conda.sh
conda activate breadcrumbs
OUT_DIR=/scratch/general/vast/u1413898/cs6957/breadcrumbs/logs
mkdir -p ${OUT_DIR}
python solution.py --output_dir ${OUT_DIR}