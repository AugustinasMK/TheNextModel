#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mincpus 5
#SBATCH --mem 70G
#SBATCH --time 24:00:00
cd /scratch/lustre/home/auma4493/TheNextModel/GLV2
pwd
singularity exec --nv /scratch/lustre/home/auma4493/p38_t20_tr.sif python3 train.py --model_ckpt /scratch/lustre/home/auma4493/TheNextModel/GLV2/checkpoints4/trained_model_4_4.pth --start_epoch 4 --end_epoch 5 --batch_size=16

