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
python tests/example_signal.py
python tests/example_function_approximation.py
python tests/exampe_gmm_exact.py
python tests/exampe_gmm.py
python tests/example_pinn.py
```

### Troubleshooting

To get cvxpy installed with pip on newer installs, we need python development
headers, pyproject-metadata, cmake, ninja, setuptools, wheel, and the
following specific versions of cvxpy dependencies
```
python -m pip install "scikit-build-core<0.8"
python -m pip install --no-build-isolation "sparsediffpy<0.4.0,>=0.3.0"
```
before installing cvxpy (or the NLGCG package with optional tests).
