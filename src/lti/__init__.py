"""LongTermInvestments — SEC fundamentals screener & backtester.

Importing this package renders a valid ``.secfsdstools.cfg`` and points the
``SECFSDSTOOLS_CFG`` environment variable at it (see :mod:`lti.config`), so any
downstream ``secfsdstools`` import picks up the project-local data directories.
"""

from lti import config as config  # noqa: F401  (import side-effect: configure secfsdstools)

__all__ = ["config"]
