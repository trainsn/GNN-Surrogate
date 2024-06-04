#!/bin/sh
#PBS -N RBF
#PBS -l walltime=4:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l mem=16GB

python -u rbf_train.py --lr 1e-1 --epochs 10000 --root /fs/project/PAS0027/mpas_graph --batch-size 1974191 --log-every 200