#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mincpus 5
#SBATCH --mem 100G
#SBATCH --time 48:00:00
cd /scratch/lustre/home/auma4493/TheNextModel/ViT
pwd
singularity exec --nv /scratch/lustre/home/auma4493/p38_t20_tr.sif python3 Vit.py --model_ckpt /scratch/lustre/home/auma4493/TheNextModel/ViT/vit_checkpoints/trained_model_20_20.pth --start_epoch 20 --end_epoch 21 --batch_size 8 --num_negatives 3 --use_hnm True

