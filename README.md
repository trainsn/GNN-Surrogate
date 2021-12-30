# GNN-Surrogate
The source code for the PacificVis 2022 submission "GNN-Surrogate: A Hierarchical and Adaptive Graph Neural Network for Parameter Space Exploration of Unstructured-Mesh Ocean Simulations"


## Graph Hierarchy Generation

Given the MPAS-Ocean mesh structure (a netcdf file), a corresponding graph hierarchy is generated. (Figure 2(a))

After compling, run 

```
cd mpas_graph/build
./MPASGraph netcdf_filepath
```

## Training Data Generation

Given a MPAS-Ocean netcdf file, extract the tempearture field and represent it with a 1D numpy array used for future training. 

After compling, run 

```
cd mpas_perm/build
./MPASPerm /path/to/input_root /path/to/output_root netcdf_filename
```

