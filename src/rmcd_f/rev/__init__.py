"""Revision layer for the MPCF paper.

This package implements the reproducibility, statistics, logging and plotting
requirements of the revision plan while leaving the MPCF solver core
(``rmcd_f.task_path_fortification``) mathematically unchanged.

Modules
-------
``ids``       canonical ``graph_id`` / ``base_id`` / ``instance_id`` conventions
``schema``    the unified run-level record and the ``runs_long.csv`` writer
``registry``  explicit baseline variant registry (one row per variant)
``config``    frozen experiment configuration, config hashes and code revision
``env``       automatic ``environment.json`` capture
``harness``   one generic executor that turns a solver call into run records
``panels``    frozen instance panels (main / scaling / role / heterogeneity)
``stats``     graph- and base-ID clustered statistics
``plots``     CSV-driven figure generation
``gates``     pre-run acceptance gates
"""

from __future__ import annotations

REVISION_VERSION = "mpcf-revision-1.0"

__all__ = ["REVISION_VERSION"]
