#!/bin/sh
#PBS -N eval_img-depth-equator
#PBS -l walltime=2:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=8GB

python eval_img.py --root /fs/project/PAS0027/MPAS1 --dir INR --mode depth-equator
python eval_img.py --root /fs/project/PAS0027/MPAS1 --dir Pred --mode depth-equator