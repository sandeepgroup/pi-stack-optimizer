# pi-Stack Optimizer

`pi-stack-optimizer` is a command-line framework for discovering low-energy
one-dimensional molecular stacking geometries from a monomer XYZ structure. It
combines xTB single-point energy calculations with global optimizers including
PSO, GA, GWO, and a PSO plus Nelder-Mead hybrid.

Core capabilities:

- Rigid-body stack optimization from a single monomer.
- Optional intramolecular torsion optimization.
- Symmetric torsion dimension reduction.
- Parallel xTB energy evaluation.
- Cached hyperparameter optimization through `pi-hyperopt`.
- Sphinx documentation for Read the Docs.

## Authors

Arunima Ghosh, Susmita Barik, Roshan J Singh, and Sandeep K. Reddy

## Quick Start

```bash
git clone https://github.com/sandeepgroup/pi-stack-optimizer.git
cd pi-stack-optimizer

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy

source ./activate_pi_stack.sh
pi-stack-generator test/BTA.xyz --workers 1 --threads 1 --max-iters 5
```

The main optimizer requires the `xtb` executable on `PATH`.

```bash
conda install -c conda-forge xtb
xtb --version
```

For hyperparameter optimization, install Optuna:

```bash
python -m pip install optuna
```

## Commands

After sourcing `activate_pi_stack.sh`:

- `pi-stack-generator` runs `pi-stack-generator.py`.
- `pi-hyperopt` runs `hyperparameter-opt/hyperopt.py`.

Without activation, call the scripts directly with Python:

```bash
python pi-stack-generator.py --help
python hyperparameter-opt/hyperopt.py --help
```

## Documentation

The Read the Docs source lives in `docs/`.

Build it locally with:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W -b html docs docs/_build/html
```

The published documentation is available at:
https://stack-pso.readthedocs.io/en/latest/

## Citation

Ghosh, A., Barik, S., Singh, R. J. et al. *pi-stack optimizer: framework for
the design of one-dimensional supramolecular systems.* Journal of Molecular
Modeling, 32, 188 (2026). https://doi.org/10.1007/s00894-026-06725-4

```bibtex
@article{ghosh2026pistack,
  title   = {{$\pi$-stack optimizer: framework for the design of one-dimensional supramolecular systems}},
  author  = {Ghosh, Arunima and Barik, Susmita and Singh, Roshan J. and Reddy, Sandeep K.},
  journal = {Journal of Molecular Modeling},
  volume  = {32},
  number  = {6},
  pages   = {188},
  year    = {2026},
  doi     = {10.1007/s00894-026-06725-4},
  url     = {https://link.springer.com/10.1007/s00894-026-06725-4}
}
```
