# Active Learning for Semantic Segmentation

## Usage

1. Clone this project.
```
git clone https://github.com/arvindmvepa/al-seg.git
cd al-seg
```
2. Clone the submodules
```
git submodule update --init --recursive --remote
```
3. Download the data located at ![](https://drive.google.com/drive/folders/1MdxIdD9auRzfH6_Np9hiEyxkAuHAUgZA?usp=sharing) and unzip in the current directory: `smpl_data.zip` and `spml_pretrained.zip` for `SPMLModel` and `wsl4mis.zip` for `DMPLSModel`

4. Make sure `virtualenv` is installed and create a virtual environment for each model you will run and install the libraries in that model's requirement_file (`DMPLSModel` uses `wsl4mis_requirements.txt`)
```
python -m venv /path/to/new/virtual/environment
pip install -r model_requirements.txt
```

5. Update the relevant parameters in `exp.yml` (see Experiment Parameters section below for details)

6. Run the Active Learning Experiment
```
python run_al_exp.py
```

## Experiment Parameters
There are three parameter groups in `exp.yml`: `model`, `model_uncertainty`, and `policy`. Additionally, at the same indent-evel, you can specify a `exp_dir` which creates an experiment directory in the specified directory. Each of the parameter group parameters are passed directly to its respective factory. Thus, one can inspect the meaning of different parameters by looking at the constructors for each parameter group type (i.e. model, model_uncertainty, and policy type). Just to note, in the `policy` group, `ensemble_kwargs` are parameters passed to the `train_model` method and `uncertainty_kwargs` are the parameters passed to the `calculate_uncertainty` method. We can optionally pass `skip` to these `kwarg` arguments to skip these steps for debugging purposes.
