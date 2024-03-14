# CoreCLR: Active Learning in 3D Segmentation through Contrastive Learning

## Usage

1. Make sure `conda` or `virtualenv` is installed and create a virtual environment and install 
the libraries in `wsl4mis_requirements.txt`
```
pip install -r model_requirements.txt
```
2. Set up the data (as described in the Data section below)
3. Update the relevant parameters in `exp.yml` (see Experiment Parameters section below 
for details)
4. Run the Active Learning Experiment
```
python run_al_exp.py
```

## Data

In our work we used the ACDC, CHAOS, MSC-MR, and DAVIS datasets. For all four datasets, we used 
the same pre-processing methods found here: https://github.com/HiLab-git/WSL4MIS. We 
additionally have provided this code in the `wsl4mis` directory. Scribbles for the ACDC dataset 
can also be found at the previous link. Scribbles for the MSC-MR dataset can be found here 
https://github.com/BWGZK/CycleMix.

Please store the pre-processed data in `wsl4mis_data` in a top-level directory corresponding to 
the dataset as specified in `active_learning/dataset/data_params.py`. For example, the ACDC 
dataset should be in `wsl4mis_data/ACDC`. Please see the text files in `wsl4mis/data/dataset_name` 
for the subdirectory names corresponding to the train, val, and test sets (note that the 
CHAOS dataset expects an additional subdirectory for the image data as well as text files - in 
the paper we used the CT_LIVER)

## Experiment Parameters

There are four parameter groups in `exp.yml`: `model`, `data_geometry`, `model_uncertainty`, and `policy`. 
Additionally, at the same indent-evel, you can specify a `exp_dir` which creates an experiment directory in the 
specified directory. Each of the parameter group parameters are passed directly to its respective factory. Thus, one 
can inspect the meaning of different parameters by looking at the constructors for each parameter group type 
(i.e. model, data_geomtry, model_uncertainty, and policy type). 

We have provided the yaml files for the weakly-supervised ACDC dataset for our method as well as
several baseline methods. We also provided example yaml files for the strongly-supervised ACDC 
dataset and the other datasets