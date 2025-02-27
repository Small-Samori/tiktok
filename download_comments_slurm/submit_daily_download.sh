#!/bin/bash

sbatch $HOME/tiktok/download_comments_slurm/daily_comment_download.sh

# Schedule the next submission in 24 hours
sbatch --begin=now+24hour --partition=rbaltman $HOME/tiktok/download_comments_slurm/submit_daily_download.sh 