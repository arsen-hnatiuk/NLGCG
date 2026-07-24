# NLGCG
Implementation and numerical testing of the parameter-free NLGCG algorithm

## Usage

### Using conda

To create a conda environment with the required dependencies, run the command
```
conda create --name nlgcg_env --file requirements.txt
conda activate nlgcg_env
```

### Using pip

Optionally, create a new environment and activate it. Then install the package with
```
pip install -e .[tests]
```

### Recreating results

To recreate the experiments from the paper, run
```
python test/experiment_signal.py
python test/experiment_function_approximation.py
python test/experiment_gmm.py
python test/experiment_pinn.py
```
