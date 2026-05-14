Input and Output Files
======================

Input Files
-----------

Monomer XYZ
~~~~~~~~~~~

The required positional input is a standard XYZ file:

.. code-block:: text

   12
   Benzene molecule
   C  -1.21310   0.68840   0.00000
   C  -1.20280  -0.70630   0.00000
   C   0.01030  -1.39470   0.00000
   C   1.21310  -0.68840   0.00000
   C   1.20280   0.70630   0.00000
   C  -0.01030   1.39470   0.00000
   H  -2.15190   1.22980   0.00000
   H  -2.15130  -1.22940   0.00000
   H   0.01900  -2.48390   0.00000
   H   2.15190  -1.22980   0.00000
   H   2.15130   1.22940   0.00000
   H  -0.01900   2.48390   0.00000

The geometry should be in Angstrom and should not contain severe atomic
overlaps. The program aligns the monomer internally, so the input orientation
does not need to be pre-aligned to the stacking axis.

Torsions JSON
~~~~~~~~~~~~~

Use ``--torsions-file`` to optimize selected internal torsions. The file is a
JSON document with a ``torsions`` list:

.. code-block:: json

   {
     "indexing": "0-based",
     "torsions": [
       {
         "name": "Linker rotation",
         "atoms": [0, 1, 2, 3],
         "rotate_side": "d"
       }
     ]
   }

Fields:

* ``indexing`` is optional. Use ``"0-based"`` for Python-style atom indices or
  ``"1-based"`` for chemistry-style indices. If omitted, ``"0-based"`` is
  used.
* ``atoms`` is required and contains four atom indices ``i-j-k-l``. The
  central bond ``j-k`` is the rotation axis.
* ``rotate_side`` is optional. ``"d"`` rotates the side connected to atom
  ``l`` and is the default. ``"p"`` rotates the side connected to atom ``i``.
* ``name`` is optional and is used only for logs.

Output Files
------------

The main optimization script writes output files in the current working
directory.

``output.log``
~~~~~~~~~~~~~~

Full text log mirrored from standard output. It includes system information,
input file names, geometry validation, optimizer settings, progress, final
results, output filenames, and elapsed time.

``optimization_results.txt``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Final summary containing:

* Best objective value in kJ/mol.
* Binding energy per interface in kJ/mol.
* Best rigid-body parameters and reduced torsion parameters.
* Full torsion angles in degrees when torsions are enabled.

``molecule_after_torsion.xyz``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Written only when torsions are used. It contains the single monomer after the
best torsion angles have been applied, before stacking.

``molecular_stack_10molecules.xyz``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The final visualization stack. The file always contains 10 monomer copies.
This is independent of ``--n-layer``. ``--n-layer`` controls the number of
layers used inside the objective function during energy evaluation.

``pso_trajectory.csv``
~~~~~~~~~~~~~~~~~~~~~~

Written only when ``--print-trajectories`` is passed for a PSO or PSO-NM run.
Columns are:

``iteration``, ``particle_id``, ``is_global_best``, ``param_0`` ...
``param_N``, ``fitness``.

The first seven parameters are ``cos_like``, ``sin_like``, ``Tx``, ``Ty``,
``Tz``, ``Cx``, and ``Cy``. Any remaining parameters are reduced torsion
variables.

Hyperparameter Output
---------------------

``pi-hyperopt`` creates output directories controlled by its CLI defaults:

* ``output/studies`` for Optuna SQLite study databases.
* ``output/results`` for JSON result summaries.
* ``output/runs`` for per-trial execution directories.
* ``output/cache_hyperopt`` for the persistent SQLite result cache.
