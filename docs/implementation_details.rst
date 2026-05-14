Implementation Details
======================

Repository Structure
--------------------

.. code-block:: text

   pi-stack-optimizer/
   |-- pi-stack-generator.py
   |-- activate_pi_stack.sh
   |-- modules/
   |   |-- constants.py
   |   |-- geometry.py
   |   |-- logging_helpers.py
   |   |-- objective.py
   |   |-- optimizer.py
   |   |-- reporting.py
   |   |-- stacking.py
   |   |-- system_utils.py
   |   |-- torsion.py
   |   |-- xtb_workers.py
   |   `-- xyz_io.py
   |-- hyperparameter-opt/
   |   |-- hyperopt.py
   |   `-- util/
   |-- utils/
   |-- test/
   |-- doc/
   `-- docs/

Main Workflow
-------------

``pi-stack-generator.py`` performs the following steps:

1. Parse CLI options.
2. Configure ``output.log`` mirroring.
3. Set thread-count environment variables from ``--threads``.
4. Read and center the monomer XYZ file.
5. Validate the monomer geometry.
6. Align the pi core to the XY plane.
7. Load optional torsion definitions.
8. Optionally detect symmetric torsion groups.
9. Start an xTB server for an initial energy test.
10. Start an ``XTBWorkerPool`` for parallel objective evaluation.
11. Build a ``BatchObjective``.
12. Create the requested optimizer.
13. Run optimization.
14. Compute final binding energy.
15. Write result files.

Geometry
--------

``modules.geometry`` provides:

* Rigid-body transformation matrix construction.
* Monomer alignment to the XY plane.
* Bond-graph construction using covalent radii.
* Topology-preservation checks after torsion changes.
* Intermolecular and intramolecular clash penalties.

SciPy's ``cKDTree`` is used for faster clash searching when SciPy is installed.
The code falls back to NumPy if SciPy is unavailable.

Objective Evaluation
--------------------

``BatchObjective.batch_evaluate`` accepts a batch of candidate parameter
vectors. For each vector it:

1. Applies reduced torsions to the monomer.
2. Rejects topology-changing conformations.
3. Builds an ``n``-layer stack.
4. Computes geometric penalties.
5. Rejects catastrophic overlaps before xTB.
6. Submits monomer and stack xTB calculations to the worker pool.
7. Converts the binding energy to kJ/mol per interface.
8. Adds penalty terms.

xTB Execution
-------------

``OptimizedXTBServer`` shells out to the ``xtb`` binary with:

.. code-block:: text

   xtb coord.xyz --gfn <method> --sp --chrg <charge> --uhf <mult - 1>

Each server reuses a temporary working directory and caches energies by atom
types and coordinates rounded to 0.01 Angstrom. The worker pool starts separate
processes, each with its own optimized xTB server.

Optimization
------------

``modules.optimizer`` exposes a shared optimizer interface through
``create_optimizer``. Supported methods are:

* ``pso``
* ``ga``
* ``gwo``
* ``pso-nm``

Only PSO-based methods can write ``pso_trajectory.csv``.

Known Current Limitations
-------------------------

* The main CLI has no ``--solvent`` argument even though ``XTBConfig`` has an
  internal ``solvent`` field.
* The final visualization output is fixed at 10 molecules.
* The repository does not currently ship a root ``requirements.txt``.
* Some utility scripts require optional dependencies that the main optimizer
  does not require.
