# GNN-Surrogate
The source code for our PacificVis 2022 paper "GNN-Surrogate: A Hierarchical and Adaptive Graph Neural Network for Parameter Space Exploration of Unstructured-Mesh Ocean Simulations".

## Getting Started

### Graph Hierarchy Generation
<img src="https://github.com/trainsn/GNN-Surrogate/blob/main/images/overview(a).jpg" width="80%">

Given the MPAS-Ocean mesh structure (a netcdf file), a corresponding graph hierarchy is generated. 

After compling MPASGraph, run 

```
cd mpas_graph/build
./MPASGraph netcdf_filepath
```

### Training Data Preperation 

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

### Cutting Policy Generation
<img src="https://github.com/trainsn/GNN-Surrogate/blob/main/images/overview(b).jpg" width="80%">

After a few simulations are run，generate the graph hierarchy cutting policy.

After compling MPASGHT, run

```
cd mpas_ght/build
./MPASGHT /path/to/graph_root/ threshold /path/to/input_netcdf_root/ /path/to/ght_root/ numFiles binary_filenames
python adjMat.py --root /path/to/ght_root/
```

Next, given the cutting policy, use adaptive resolutions to represent each ensemble member.
```

python res2ght.py --root dataset --ght ght_dir
```

### Model Training 
<img src="https://github.com/trainsn/GNN-Surrogate/blob/main/images/overview(d).jpg" width="60%">

A deep surrogate model (i.e., GNN-Surrogate) is trained based on the generated training dataset:
```
cd model
python main.py --root dataset --gan-loss none --sn --ch channel_multiplier 
```

### Inference 
<img src="https://github.com/trainsn/GNN-Surrogate/blob/main/images/overview(e).jpg" width="20%">

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

Having the new MPAS netcdf file, consider using the following repo for visualizatoin: 
[MPASMap](https://github.com/trainsn/MPASMap) for horizontal cross-sections, 
[MPASCross](https://github.com/trainsn/MPASCross) for vertical cross-sections, 
and [MPASDepth](https://github.com/trainsn/MPASDepth) for Isothermal Layer (ITL) Depth. 

## Citation

If you use this code for your research, please cite our paper.
```
@article{shi2022gnn,
  title={GNN-Surrogate: A Hierarchical and Adaptive Graph Neural Network for Parameter Space Exploration of Unstructured-Mesh Ocean Simulations},
  author={Shi, Neng and Xu, Jiayi and Wurster, Skylar W and Guo, Hanqi and Woodring, Jonathan and Van Roekel, Luke P and Shen, Han-Wei},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2022},
  publisher={IEEE}
}
```

## Acknowledgments
Our code is inspired by [InSituNet](https://github.com/hewenbin/insitu_net).
