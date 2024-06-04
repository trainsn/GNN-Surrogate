#PBS -N adjMat
#PBS -l walltime=2:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=100GB
#PBS -j oe

python adjMat.py

cp /users/PAS0027/trainsn/mpas/mpas_ght/res/EC60to30_0.5/ghtAvgPoolAsgn*.npy /fs/ess/PAS0027/mpas_graph/ght_0.5/graph
cp /users/PAS0027/trainsn/mpas/mpas_ght/res/EC60to30_0.5/ghtUpAsgn*.npy /fs/ess/PAS0027/mpas_graph/ght_0.5/graph
cp /users/PAS0027/trainsn/mpas/mpas_ght/res/EC60to30_0.5/ghtGraphSizes.npy /fs/ess/PAS0027/mpas_graph/ght_0.5/graph
