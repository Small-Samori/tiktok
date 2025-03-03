#!/bin/bash

TIKTOK_FOLDER=$OAK/samori/tiktok
COMMENT_FOLDER=$TIKTOK_FOLDER/comments
PERSONS_FOLDER=$TIKTOK_FOLDER/persons
MINICONDA_PATH=$GROUP_HOME/samori/miniconda3

ACTIVATE_ENV="source $MINICONDA_PATH/bin/activate lda_env2"
RUN_ANNOTATION="python ./annotation.py $COMMENT_FOLDER/cleaned/comments_5_cleaned.csv $PERSONS_FOLDER"

JOB_ID1=$(sbatch --parsable --partition=rbaltman --time 168:00:00 --mem 16G \
        --job-name gpt_annotation --wrap "$ACTIVATE_ENV;$RUN_ANNOTATION" \
        --output ./slurm_outputs/output_%j.log --error ./slurm_errors/error_%j.err \
        --mail-user iasamori@stanford.edu --mail-type=ALL)