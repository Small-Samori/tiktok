#!/bin/bash
#SBATCH --job-name=gpt_annotation
#SBATCH --output=./slurm_outputs/output_%j.log
#SBATCH --error=./slurm_errors/error_%j.err
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --mem=16G
#SBATCH --partition=rbaltman

source $GROUP_HOME/samori/miniconda3/condabin/conda lda_env2
python $HOME/tiktok/persons/annotations_slurm/annotation.py $OAK/samori/tiktok/comments/cleaned/comments_5_cleaned.csv $OAK/samori/tiktok/comments/annotations