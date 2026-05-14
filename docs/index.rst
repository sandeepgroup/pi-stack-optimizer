pi-Stack Optimizer Documentation
================================

``pi-stack-optimizer`` is a command-line framework for finding low-energy
one-dimensional molecular stacks from a monomer XYZ structure. It combines
semi-empirical xTB single-point energy calculations with global optimizers:
Particle Swarm Optimization (PSO), Genetic Algorithm (GA), Grey Wolf Optimizer
(GWO), and a PSO plus Nelder-Mead hybrid.

The current repository exposes two shell entry points after activation:

* ``pi-stack-generator`` for molecular stack optimization.
* ``pi-hyperopt`` for cached Optuna-based hyperparameter searches.

The documentation in this tree matches the current source code in the GitHub
repository. In particular, the final visualization stack is always written as
``molecular_stack_10molecules.xyz``; ``--n-layer`` controls the number of
layers used for energy evaluation, not the number of molecules in that final
visualization file.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   inputs_outputs
   theory
   strategies
   hyperparameter_optimization
   implementation_details
   modules

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
