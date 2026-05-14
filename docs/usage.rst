Command-Line Usage
==================

The main entry point is ``pi-stack-generator`` after sourcing
``activate_pi_stack.sh``.

.. code-block:: bash

   pi-stack-generator <xyz> [options]

The positional ``xyz`` argument is a monomer XYZ file. The script recenters the
monomer, aligns the pi core to the XY plane, evaluates stack candidates with
xTB, and writes result files in the current working directory.

Basic Examples
--------------

Run a default PSO dimer evaluation:

.. code-block:: bash

   pi-stack-generator monomer.xyz

Use three layers for energy evaluation:

.. code-block:: bash

   pi-stack-generator monomer.xyz --n-layer 3

Use the hybrid PSO plus Nelder-Mead optimizer:

.. code-block:: bash

   pi-stack-generator monomer.xyz --optimizer pso-nm --swarm-size 100 --max-iters 500

Use a faster xTB Hamiltonian for screening:

.. code-block:: bash

   pi-stack-generator monomer.xyz --method gfn0 --optimizer gwo --max-iters 100

Optimize selected torsions:

.. code-block:: bash

   pi-stack-generator monomer.xyz --torsions-file torsions.json --enable-symmetric-torsions

General Options
---------------

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Option
     - Default
     - Description
   * - ``--n-layer``
     - ``2``
     - Number of monomer layers used in the energy evaluation.
   * - ``--optimizer``
     - ``pso``
     - Optimizer: ``pso``, ``ga``, ``gwo``, or ``pso-nm``.
   * - ``--max-iters``
     - ``300``
     - Maximum PSO/GWO iterations or GA generations.
   * - ``--seed``
     - ``42``
     - Random seed for reproducible initialization.
   * - ``--verbose-every``
     - ``1``
     - Print optimizer progress every N iterations or generations.

PSO Options
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Option
     - Default
     - Description
   * - ``--swarm-size``
     - ``60``
     - Number of PSO particles.
   * - ``--inertia``
     - ``0.73``
     - Velocity inertia weight.
   * - ``--cognitive``
     - ``1.50``
     - Personal-best attraction coefficient.
   * - ``--social``
     - ``1.50``
     - Global-best attraction coefficient.
   * - ``--tol``
     - ``0.01``
     - PSO early-stopping tolerance.
   * - ``--patience``
     - ``20``
     - PSO early-stopping patience in iterations.
   * - ``--print-trajectories``
     - ``False``
     - Write ``pso_trajectory.csv`` for PSO and PSO-NM runs.

Genetic Algorithm Options
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 18 48

   * - Option
     - Default
     - Description
   * - ``--ga-population``
     - ``80``
     - Population size.
   * - ``--ga-mutation-rate``
     - ``0.10``
     - Per-gene mutation probability.
   * - ``--ga-mutation-sigma``
     - ``0.30``
     - Gaussian mutation noise scale.
   * - ``--ga-crossover-rate``
     - ``0.90``
     - Crossover probability.
   * - ``--ga-elite-fraction``
     - ``0.10``
     - Fraction of best candidates copied into the next generation.
   * - ``--ga-tournament-size``
     - ``3``
     - Parent-selection tournament size.
   * - ``--ga-tol``
     - ``0.01``
     - GA early-stopping tolerance.
   * - ``--ga-patience``
     - ``20``
     - GA early-stopping patience in generations.

Grey Wolf Optimizer Options
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Option
     - Default
     - Description
   * - ``--gwo-pack-size``
     - ``50``
     - Number of wolves.
   * - ``--gwo-a-start``
     - ``2.0``
     - Initial GWO exploration parameter.
   * - ``--gwo-a-end``
     - ``0.0``
     - Final GWO exploration parameter.
   * - ``--gwo-tol``
     - ``0.01``
     - GWO early-stopping tolerance.
   * - ``--gwo-patience``
     - ``20``
     - GWO early-stopping patience in iterations.

Hybrid PSO plus Nelder-Mead Options
-----------------------------------

The hybrid first runs PSO, then refines the best PSO point with a Nelder-Mead
simplex search.

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Option
     - Default
     - Description
   * - ``--hybrid-nm-max-iters``
     - ``200``
     - Maximum Nelder-Mead iterations.
   * - ``--hybrid-nm-initial-step``
     - ``0.20``
     - Initial simplex step.
   * - ``--hybrid-nm-alpha``
     - ``1.0``
     - Reflection coefficient.
   * - ``--hybrid-nm-gamma``
     - ``2.0``
     - Expansion coefficient.
   * - ``--hybrid-nm-rho``
     - ``0.50``
     - Contraction coefficient.
   * - ``--hybrid-nm-sigma``
     - ``0.50``
     - Shrink coefficient.
   * - ``--hybrid-nm-tol``
     - ``0.01``
     - Simplex convergence tolerance.

xTB Backend Options
-------------------

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Option
     - Default
     - Description
   * - ``--workers``
     - ``4``
     - Number of parallel xTB worker processes.
   * - ``--threads``
     - ``1``
     - Number of OpenMP/BLAS threads per xTB calculation.
   * - ``--method``
     - ``gfn2``
     - xTB method. Recognized values are ``gfn2``, ``gfn1``, ``gfn0``, and ``gfnff``.
   * - ``--charge``
     - ``0``
     - Molecular charge.
   * - ``--mult``
     - ``1``
     - Spin multiplicity. The xTB ``--uhf`` value is ``max(mult - 1, 0)``.

The current CLI does not expose a solvent option. The internal ``XTBConfig``
has a ``solvent`` field, but ``pi-stack-generator.py`` always leaves it unset.

Torsion and Symmetry Options
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 38 18 44

   * - Option
     - Default
     - Description
   * - ``--torsions-file``
     - disabled
     - JSON file describing rotatable torsions.
   * - ``--enable-symmetric-torsions``
     - ``False``
     - Detect torsions with matching or opposite current dihedral values and reduce the optimization dimension.
   * - ``--symmetric-torsion-tolerance``
     - ``10.0``
     - Tolerance in degrees for symmetry grouping.

Penalty Options
---------------

.. list-table::
   :header-rows: 1
   :widths: 38 18 44

   * - Option
     - Default
     - Description
   * - ``--penalty-weight``
     - ``2.0``
     - Multiplier for intermolecular clash penalty.
   * - ``--clash-cutoff``
     - ``1.6``
     - Intermolecular clash cutoff in Angstrom.
   * - ``--intramol-penalty-weight``
     - ``5.0``
     - Multiplier for intramolecular clash penalty.
   * - ``--intramol-cutoff``
     - ``1.2``
     - Intramolecular clash cutoff in Angstrom.

Logging and Output
------------------

All normal ``print`` output is mirrored to ``output.log`` in the current
working directory. Final machine-readable artifacts are described in
:doc:`inputs_outputs`.
