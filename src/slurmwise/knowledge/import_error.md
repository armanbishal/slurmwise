---
id: import_error
title: Missing Python module or failed environment module
kinds: [import_error, module_load]
tags: [ModuleNotFoundError, ImportError, venv, conda, module load, PYTHONPATH, Lmod]
severity: user
---
# Missing Python module or failed environment module

## Symptoms
- `ModuleNotFoundError: No module named '...'` or `ImportError: cannot import name ...`
- `Lmod has detected the following error: The following module(s) are unknown`
- The log may echo "activating env" but the import still fails

## Causes
- `source venv/bin/activate` or `conda activate` missing from the sbatch script
- Package installed with `pip install --user` for a different Python
- Missing `module load` so a different python is first on PATH
- A requested environment module name is wrong or unavailable

## Fixes
1. Add the activation before python: `source /path/to/.venv/bin/activate`.
2. Print the interpreter at the top of the job: `which python; python -c "import sys; print(sys.prefix)"`.
3. Install into the env the job actually uses (from a compute node if needed).
4. For an unknown module, check `module avail <name>` and load the exact version.

## Verify
- `which python` and `pip show <package>` inside the job confirm the right environment.
