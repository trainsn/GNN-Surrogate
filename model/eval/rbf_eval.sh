#!/bin/sh
#PBS -N RBF-eval_epoch2000
#PBS -l walltime=4:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l mem=16GB

python -u rbf_eval.py --root /fs/project/PAS0027/mpas_graph --batch-size 1974191  --epoch 10000 --equator
