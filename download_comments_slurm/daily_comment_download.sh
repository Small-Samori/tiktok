#!/bin/bash
#SBATCH --job-name=daily_comment_download
#SBATCH --output=./slurm_outputs/daily_comment_download_%j.log
#SBATCH --error=./slurm_errors/daily_comment_download_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=4G
#SBATCH --partition=rbaltman

source $GROUP_HOME/samori/miniconda3/condabin/conda esm_env
python $HOME/tiktok/download_comments_slurm/download_comments.py