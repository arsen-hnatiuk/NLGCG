# NLGCG
Implementation and numerical testing of the parameter-free NLGCG algorithm

## Usage

### Using conda

To create a conda environment with the required dependencies, run the command
```
conda env create -f environment.yml -n env_nlgcg
conda activate env_nlgcg
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
python tests/exampe_gmm.py
python tests/example_pinn.py
python tests/example_source_identificytion.py
```

To run the experiments, add the `--run_experiments` argument. Otherwise, the code assumes that the experiments have already been run and only generates plots. For verbose outputs, add the `--verbose` argument.

The generate the particle descent matrices in the signal processing experiment, add the `--particle_matrix` argument.

The source identification example only serves to generate the NLGCG aglorithm visualization plot from page ... of the manuscript.

### Troubleshooting

To get cvxpy installed with pip on newer installs, we need python development
headers, pyproject-metadata, cmake, ninja, setuptools, wheel, and the
following specific versions of cvxpy dependencies
```
python -m pip install "scikit-build-core<0.8"
python -m pip install --no-build-isolation "sparsediffpy<0.4.0,>=0.3.0"
```
before installing cvxpy (or the NLGCG package with optional tests).
