# GNN-Surrogate
The source code for the PacificVis 2022 submission "GNN-Surrogate: A Hierarchical and Adaptive Graph Neural Network for Parameter Space Exploration of Unstructured-Mesh Ocean Simulations"


## Graph Hierarchy Generation

Given the MPAS-Ocean mesh structure (a netcdf file), a corresponding graph hierarchy is generated (Figure 2(a)). 

After compling MPASGraph, run 

```
cd mpas_graph/build
./MPASGraph netcdf_filepath
```

## Training Data Preperation 

Given a MPAS-Ocean netcdf file, extract the tempearture field and represent it with a 1D array used for future training. 
After compling MPASPerm, run 

```
cd mpas_perm/build
./MPASPerm /path/to/input_root/ /path/to/output_root/ netcdf_filename
```

Next, given the reference, calculate the residual for every ensemble member:
```
cd prepost
python raw2res.py --root dataset --reference refence_ensemble_member
```

## Cutting Policy Generation

After a few simulations are run，generate the graph hierarchy cutting policy (Figure 2(b). 

After compling MPASGHT, run

```
cd mpas_ght/build
./MPASGHT /path/to/graph_root/ threshold /path/to/input_netcdf_root/ /path/to/ght_root/ numFiles binary_filenames
python adjMat.py --root /path/to/ght_root/
```

Next, given the cutting policy, use adaptive resolutions to represent each ensemble member:
```

python res2ght.py --root dataset --ght ght_dir
```

## Model Training 

A deep surrogate model (i.e., GNN-Surrogate) is trained based on the generated training dataset: (Figure 2(d))
```
cd model
python main.py --root dataset --gan-loss none --sn --ch channel_multiplier 
```

## Inference 

In the inference stage, GNN-Surrogate is first used to predict the simulation residual:
```
cd model
python infer.py --root dataset --gan-loss none --sn --ch channel_multiplier --resume trained_model --bwsa bwsa --kappa kappa --cvmix cvmix --mom mom
```

Next, we add the reference back to obtain predicted simulation outputs:
```
cd prepost
python res2raw.py --root dataset --reference refence_ensemble_member --ght ght_dir
```

Finally, we load the predicted simulation output back to the MPAS netcdf file.

After compling MPASPermBack, run
```
cd mpas_permBack/build
./MPASPermBack /path/to/input_root/ /path/to/output_root/ netcdf_filename
```
