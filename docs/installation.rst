Installation
============

Prerequisites
-------------

The optimizer requires:

* Python 3.9 or newer.
* The ``xtb`` command-line executable available on ``PATH``.
* NumPy for the main optimizer.
* Optuna only if you use ``pi-hyperopt``.
* RDKit only if you use the helper scripts in ``utils/`` that infer torsions
  from chemical structures.

The repository currently does not include a root ``requirements.txt`` file, so
install the Python dependencies explicitly.

Clone the Repository
--------------------

.. code-block:: bash

   git clone https://github.com/sandeepgroup/pi-stack-optimizer.git
   cd pi-stack-optimizer

Create an Environment
---------------------

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install numpy

For hyperparameter optimization:

.. code-block:: bash

   python -m pip install optuna

For local documentation builds:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt

xTB Setup
---------

Install xTB and make sure the executable is visible as ``xtb``:

.. code-block:: bash

   conda install -c conda-forge xtb
   xtb --version

The optimizer controls per-calculation threading through ``--threads``. At
runtime it sets ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``, and
``OPENBLAS_NUM_THREADS`` to that value.

Activate Convenience Commands
-----------------------------

From the repository root:

.. code-block:: bash

   source ./activate_pi_stack.sh

This adds the repository root to ``PATH`` and ``PYTHONPATH`` and defines:

* ``pi-stack-generator`` as a wrapper around ``pi-stack-generator.py``.
* ``pi-hyperopt`` as a wrapper around ``hyperparameter-opt/hyperopt.py``.

Without activation, run the scripts directly:

.. code-block:: bash

   python pi-stack-generator.py test/BTA.xyz --workers 1 --max-iters 5
   python hyperparameter-opt/hyperopt.py --help

Smoke Test
----------

After activation, a short run is:

.. code-block:: bash

   pi-stack-generator test/BTA.xyz --workers 1 --threads 1 --max-iters 5

The command should create ``output.log``, ``optimization_results.txt``, and
``molecular_stack_10molecules.xyz`` in the working directory if xTB is
installed and the geometry passes validation.
