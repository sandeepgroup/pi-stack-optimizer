**π-Stack Optimizer** is a high-performance computational framework for discovering energetically favorable stacking configurations in molecular systems. Leveraging semi-empirical quantum chemistry (xTB) together with multiple global optimization algorithms (PSO, GA, GWO, and PSO+Nelder–Mead), the framework provides researchers with a flexible, efficient, and reproducible workflow for π-stacking studies.
The framework systematically explores translational, rotational, and optional intramolecular torsional degrees of freedom to identify stable low-energy supramolecular assemblies directly from a monomeric building block. Additional features include:
  - Parallel energy evaluations
  - Automatic symmetry detection
  - Hyperparameter optimization
  - Comprehensive logging and reproducibility support
  - Modular architecture for extensibility
  
The tool is suitable for both exploratory supramolecular research and large-scale computational chemistry workflows.
Ghosh, A., Barik, S., Singh, R. J, & Reddy, S. K. (2026). *π-Stack Optimizer: A high-performance computational framework for discovering energetically favorable stacking configurations in molecular systems.*

**Authors:** 
Arunima Ghosh, Susmita Barik, Roshan J Singh, & Sandeep K. Reddy

**How to activate the repository scripts in your shell**

Source the activation script to get `pi-stack-generator` and `pi-hyperopt` available in your shell session.

From the repo root:

```bash
source ./activate_pi_stack.sh
```
This performs the following:
- Adds the project root to your `PATH` so scripts can be executed directly.
- Adds the project root to `PYTHONPATH` so `import modules.*` resolves.
- Defines two shell functions:
  - `pi-stack-generator` -> runs `pi-stack-generator.py` with the same args.
  - `pi-hyperopt` -> runs `hyperparameter-opt/hyperopt.py` with the same args.

If preferred, make the scripts directly executable:

```bash
chmod +x pi-stack-generator.py
chmod +x hyperparameter-opt/hyperopt.py
```
**Documentation:**

Complete documentation, installation instructions, and tutorials are available at: https://stack-pso.readthedocs.io/en/latest/

For more details, please refer to the corresponding publication. If you use this framework in your work, we kindly request that you cite the following paper:

**Citation**

Ghosh, A., Susmita, B., Singh, R.J. et al. *π-Stack optimizer: framework for the design of one-dimensional supramolecular systems.* Journal of Molecular Modeling, 32, 188 (2026). https://doi.org/10.1007/s00894-026-06725-4


**BibTeX**

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

