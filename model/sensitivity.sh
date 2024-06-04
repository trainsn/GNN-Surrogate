#!/bin/sh
#PBS -N sensitivity
#PBS -l walltime=01:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l mem=16GB
#PBS -q gpuserial-48core

python sensitivity.py --root /fs/project/PAS0027/mpas_graph/ght_0.2  --gan-loss vanilla --sn --graph-conv ecc --ch 112 --resume /fs/project/PAS0027/mpas_graph/ght_0.2/70/model_420.pth.tar --mode north_atlantic 
