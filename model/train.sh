#!/bin/sh
#PBS -N GNNSurro-MPAS
#PBS -l walltime=30:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l mem=40GB

python3 -u main.py --root /fs/ess/PAS0027/mpas_graph/ght_0.5 --gan-loss none --gan-loss-weight 1e-2 --sn --ch 96 --log-every 8 --check-every 24