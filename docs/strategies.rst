Optimization Strategies
=======================

Particle Swarm Optimization
---------------------------

PSO is the default method. It keeps a population of particles, each with a
position, velocity, personal best, and access to the global best. Each
iteration updates velocity from inertia, personal-best attraction, and
global-best attraction.

Use PSO when you want the most general-purpose optimizer for a new system.
Increase ``--swarm-size`` and ``--max-iters`` for flexible molecules or noisy
energy landscapes.

Genetic Algorithm
-----------------

The GA maintains a population, keeps an elite fraction, selects parents by
tournament selection, applies simulated binary crossover, and applies Gaussian
mutation. It can be useful when the landscape is rough or when torsional
flexibility makes local progress unreliable.

Useful controls are ``--ga-population``, ``--ga-mutation-rate``,
``--ga-mutation-sigma``, and ``--ga-crossover-rate``.

Grey Wolf Optimizer
-------------------

GWO tracks the best three candidates as alpha, beta, and delta leaders. The
pack moves according to these leaders while the exploration parameter decreases
from ``--gwo-a-start`` to ``--gwo-a-end``.

Use GWO for fast screening and rigid-body cases where fewer tunable parameters
are desirable.

Hybrid PSO plus Nelder-Mead
---------------------------

``pso-nm`` first runs PSO for global exploration. After PSO finishes, it starts
a Nelder-Mead simplex search from the best PSO point. This is more expensive
than PSO alone, but it can refine a promising geometry.

Use it for final production runs after you already have reasonable screening
settings.

Practical Recipes
-----------------

Fast rigid-body screening:

.. code-block:: bash

   pi-stack-generator monomer.xyz --optimizer gwo --method gfn0 --max-iters 100

Balanced default run:

.. code-block:: bash

   pi-stack-generator monomer.xyz --optimizer pso --swarm-size 60 --max-iters 300

Flexible molecule run:

.. code-block:: bash

   pi-stack-generator monomer.xyz \
     --torsions-file torsions.json \
     --enable-symmetric-torsions \
     --optimizer pso \
     --swarm-size 100 \
     --max-iters 500

Final refinement:

.. code-block:: bash

   pi-stack-generator monomer.xyz --optimizer pso-nm --swarm-size 100 --max-iters 500
