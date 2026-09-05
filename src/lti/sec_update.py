"""Thin wrappers around the ``secfsdstools`` update / automation pipeline."""

from __future__ import annotations

import lti.config as config  # noqa: F401  (ensures SECFSDSTOOLS_CFG is set)

import pandas as pd


def _cfg():
    from secfsdstools.a_config.configmgt import ConfigurationManager

    return ConfigurationManager.read_config_file()


def run_full_update(force: bool = False) -> None:
    """Download all quarterly SEC zips, build parquet + SQLite index.

    When ``PostUpdateProcesses`` is set in the cfg (i.e. not smoke mode), this also
    runs the ``memory_optimized_automation`` pipeline that produces the concatenated
    standardized bags under ``data/sec/automated/_2_all/_3_standardized_by_stmt``.
    """
    from secfsdstools.update import update

    update(force_update=force)


def run_pipeline_only() -> None:
    """Re-run only the post-update automation pipeline (zips/index already present)."""
    # Trigger secfsdstools' update-check + config validation first, in isolation.
    # Otherwise the automation module below becomes the first import that pulls in
    # ``c_update_check``, which then tries to import that same module while it is
    # only half-loaded -> "define_extra_processes not found" (circular import).
    import secfsdstools.c_update_check  # noqa: F401

    from secfsdstools.c_automation.task_framework import execute_processes
    from secfsdstools.x_examples.automation.memory_optimized_automation import (
        define_extra_processes,
    )

    cfg = _cfg()
    processes = define_extra_processes(cfg)
    execute_processes(processes)


def index_dataframe() -> pd.DataFrame:
    """All indexed filings: columns ``adsh, cik, name, form, filed, period, ...``."""
    from secfsdstools.c_index.indexdataaccess import ParquetDBIndexingAccessor

    accessor = ParquetDBIndexingAccessor(db_dir=str(_cfg().db_dir))
    return accessor.read_all_indexreports_df()


def latest_quarter() -> str | None:
    """Name of the most recent processed quarter zip, e.g. ``2025q2.zip``."""
    from secfsdstools.c_index.indexdataaccess import ParquetDBIndexingAccessor

    accessor = ParquetDBIndexingAccessor(db_dir=str(_cfg().db_dir))
    return accessor.find_latest_quarter_file_name()


def all_quarter_zip_names() -> list[str]:
    """Every available quarter zip name, excluding the empty ``2009q1.zip``."""
    from secfsdstools.c_index.indexdataaccess import ParquetDBIndexingAccessor

    accessor = ParquetDBIndexingAccessor(db_dir=str(_cfg().db_dir))
    return [
        x.fileName
        for x in accessor.read_all_indexfileprocessing()
        if x.fullPath
        and not x.fullPath.endswith("2009q1.zip")
        and x.fileName.endswith(".zip")
        and getattr(x, "status", "processed") == "processed"
    ]
