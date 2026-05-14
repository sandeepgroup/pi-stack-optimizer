# Installation

## Requirements

- Python 3.9 or newer.
- The `xtb` executable available on `PATH`.
- NumPy for the main optimizer.
- Optuna for `pi-hyperopt`.
- RDKit only for utility scripts that infer torsions from chemistry files.

## Setup

```bash
git clone https://github.com/sandeepgroup/pi-stack-optimizer.git
cd pi-stack-optimizer

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy
```

Install xTB, for example with conda:

```bash
conda install -c conda-forge xtb
xtb --version
```

Optional hyperparameter optimization dependency:

```bash
python -m pip install optuna
```

## Activate Commands

From the repository root:

```bash
source ./activate_pi_stack.sh
```

This adds the project root to `PATH` and `PYTHONPATH`, and defines:

- `pi-stack-generator`, which runs `pi-stack-generator.py`.
- `pi-hyperopt`, which runs `hyperparameter-opt/hyperopt.py`.

## Verify

```bash
pi-stack-generator test/BTA.xyz --workers 1 --threads 1 --max-iters 5
```

If xTB is installed and the geometry passes validation, the run writes
`output.log`, `optimization_results.txt`, and
`molecular_stack_10molecules.xyz`.

Complete source documentation is in `docs/`.
