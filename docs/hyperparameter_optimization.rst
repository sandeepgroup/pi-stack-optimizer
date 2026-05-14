Hyperparameter Optimization
===========================

``pi-hyperopt`` runs cached Optuna studies around ``pi-stack-generator.py``.
It tests optimizer hyperparameters on one or more molecule folders, stores
trial outputs, and memoizes completed results in SQLite so repeated searches
can reuse previous evaluations.

Requirements
------------

Install Optuna before using this command:

.. code-block:: bash

   python -m pip install optuna

Directory Layout
----------------

By default, molecule inputs are read from ``input``. Each molecule should live
in its own subdirectory and contain an XYZ file. If a torsions JSON file is
present, the hyperopt runner forwards it with ``--torsions-file``.

Example:

.. code-block:: text

   input/
   |-- molecule_a/
   |   |-- monomer.xyz
   |   `-- torsions.json
   `-- molecule_b/
       `-- monomer.xyz

Basic Use
---------

Sequential per-molecule studies:

.. code-block:: bash

   pi-hyperopt --molecules-root input --optimizer pso --trials-per-molecule 50 --progress

Joint study across all molecules:

.. code-block:: bash

   pi-hyperopt \
     --molecules-root input \
     --optimizer pso \
     --trials-per-molecule 100 \
     --joint-study \
     --progress

Forward extra stack-optimizer arguments after ``--base-args``. This option must
be last:

.. code-block:: bash

   pi-hyperopt \
     --optimizer pso \
     --trials-per-molecule 20 \
     --base-args --workers 8 --threads 1 --n-layer 3 --method gfn1

CLI Reference
-------------

.. list-table::
   :header-rows: 1
   :widths: 34 24 42

   * - Option
     - Default
     - Description
   * - ``--molecules-root``
     - ``input``
     - Directory containing molecule subdirectories.
   * - ``--stack-script``
     - ``../pi-stack-generator.py``
     - Path to the main optimizer script.
   * - ``--optimizer``
     - ``pso``
     - Optimizer to tune: ``pso``, ``ga``, ``gwo``, or ``pso-nm``.
   * - ``--trials-per-molecule``
     - ``50``
     - Number of Optuna trials.
   * - ``--joint-study``
     - ``False``
     - Optimize average performance across all discovered molecules.
   * - ``--progress``
     - ``False``
     - Show progress bars and detailed output.
   * - ``--study-dir``
     - ``output/studies``
     - Optuna SQLite database directory.
   * - ``--results-dir``
     - ``output/results``
     - JSON result summary directory.
   * - ``--runs-dir``
     - ``output/runs``
     - Trial execution directory.
   * - ``--cache-dir``
     - ``output/cache_hyperopt``
     - Persistent result-cache directory.
   * - ``--clear-cache``
     - ``False``
     - Clear cached evaluations before starting.
   * - ``--cache-stats``
     - ``False``
     - Print cache statistics and exit.
   * - ``--reduced-iterations``
     - ``50``
     - ``--max-iters`` value used for each trial run.
   * - ``--base-args``
     - none
     - Remaining arguments forwarded to ``pi-stack-generator.py``.
   * - ``--keep-run-dirs``
     - ``False``
     - Preserve per-trial directories after successful runs.

Tuned Parameter Ranges
----------------------

``pso`` tunes ``swarm_size``, ``inertia``, ``cognitive``, and ``social``.

``ga`` tunes ``ga_population``, ``ga_mutation_rate``,
``ga_mutation_sigma``, ``ga_crossover_rate``, ``ga_elite_fraction``, and
``ga_tournament_size``.

``gwo`` tunes ``gwo_pack_size``, ``gwo_a_start``, and ``gwo_a_end``.

``pso-nm`` tunes the PSO parameters plus Nelder-Mead controls:
``hybrid_nm_max_iters``, ``hybrid_nm_initial_step``, ``hybrid_nm_alpha``,
``hybrid_nm_gamma``, ``hybrid_nm_rho``, ``hybrid_nm_sigma``, and
``hybrid_nm_tol``.
