"""Project paths and one-time ``secfsdstools`` configuration.

This module has an *import side-effect*: :func:`ensure_secfsdstools_cfg` runs at
import time so that ``SECFSDSTOOLS_CFG`` is set before anything imports
``secfsdstools``. Always ``import lti`` (or ``import lti.config``) before importing
``secfsdstools``.

Environment variables read here:

``LTI_USER_AGENT_EMAIL``
    Overrides the SEC ``User-Agent`` email (default ``malouf.dominic@gmail.com``).
``LTI_SMOKE``
    When set to ``1``/``true``, the generated cfg omits ``PostUpdateProcesses`` so
    ``lti update`` only downloads + indexes the raw SEC data and skips the
    multi-hour standardization pipeline. Also switches the derived-data filenames
    to their ``*.smoke.parquet`` variants.
"""

from __future__ import annotations

import os
import string
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DEFAULT_USER_AGENT_EMAIL = "malouf.dominic@gmail.com"

_TEMPLATE_PATH = PROJECT_ROOT / ".secfsdstools.cfg.template"
_RENDERED_CFG_PATH = PROJECT_ROOT / ".secfsdstools.cfg"


def user_agent_email() -> str:
    return os.environ.get("LTI_USER_AGENT_EMAIL", DEFAULT_USER_AGENT_EMAIL)


def is_smoke() -> bool:
    return os.environ.get("LTI_SMOKE", "").strip().lower() in {"1", "true", "yes"}


def user_agent() -> str:
    """HTTP ``User-Agent`` header value the SEC asks for on bulk endpoints."""
    return f"LongTermInvestments/1.0 ({user_agent_email()})"


@dataclass(frozen=True)
class Paths:
    root: Path

    # secfsdstools native dirs
    sec_dld: Path
    sec_db: Path
    sec_parquet: Path

    # memory_optimized_automation pipeline outputs
    sec_automated: Path
    concat_std_bs: Path
    concat_std_is: Path
    concat_std_cf: Path

    # our derived artifacts
    derived_dir: Path
    fundamentals_parquet: Path
    cik_ticker_parquet: Path

    # price cache
    prices_dir: Path
    adj_close_parquet: Path
    prices_meta_parquet: Path

    def all_dirs(self) -> list[Path]:
        return [
            self.sec_dld,
            self.sec_db,
            self.sec_parquet,
            self.sec_automated,
            self.derived_dir,
            self.prices_dir,
        ]


@dataclass(frozen=True)
class UniverseConfig:
    start_year: int = 2009
    forms: tuple[str, ...] = ("10-K",)
    market_cap_floor: float = 50_000_000.0
    benchmark: str = "SPY"


# A small, stable set of large caps used by the smoke path (plus the benchmark).
SMOKE_TICKERS: tuple[str, ...] = (
    "AAPL", "MSFT", "JNJ", "PG", "KO", "JPM", "XOM", "WMT",
    "HD", "PFE", "CSCO", "INTC", "VZ", "T", "CVX",
)


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    root = PROJECT_ROOT
    sec = root / "data" / "sec"
    automated = sec / "automated"
    concat_std = automated / "_2_all" / "_3_standardized_by_stmt"
    derived = root / "data" / "derived"
    prices = root / "data" / "prices"

    suffix = ".smoke.parquet" if is_smoke() else ".parquet"

    return Paths(
        root=root,
        sec_dld=sec / "dld",
        sec_db=sec / "db",
        sec_parquet=sec / "parquet",
        sec_automated=automated,
        concat_std_bs=concat_std / "BS",
        concat_std_is=concat_std / "IS",
        concat_std_cf=concat_std / "CF",
        derived_dir=derived,
        fundamentals_parquet=derived / f"fundamentals{suffix}",
        cik_ticker_parquet=derived / "cik_ticker.parquet",
        prices_dir=prices,
        adj_close_parquet=prices / "adj_close.parquet",
        prices_meta_parquet=prices / "_prices_meta.parquet",
    )


@lru_cache(maxsize=1)
def get_universe_config() -> UniverseConfig:
    return UniverseConfig()


def _render_cfg_text() -> str:
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    rendered = string.Template(template).substitute(
        PROJECT_ROOT=str(PROJECT_ROOT),
        USER_AGENT_EMAIL=user_agent_email(),
    )
    if is_smoke():
        rendered = "\n".join(
            line
            for line in rendered.splitlines()
            if not line.lower().lstrip().startswith("postupdateprocesses")
        ) + "\n"
    return rendered


def ensure_secfsdstools_cfg() -> Path:
    """Render ``.secfsdstools.cfg`` from the template, create data dirs, set env var.

    Idempotent: the file is only rewritten when its contents would change.
    """
    paths = get_paths()
    for directory in paths.all_dirs():
        directory.mkdir(parents=True, exist_ok=True)

    desired = _render_cfg_text()
    if not _RENDERED_CFG_PATH.exists() or _RENDERED_CFG_PATH.read_text(encoding="utf-8") != desired:
        _RENDERED_CFG_PATH.write_text(desired, encoding="utf-8")

    os.environ["SECFSDSTOOLS_CFG"] = str(_RENDERED_CFG_PATH)
    return _RENDERED_CFG_PATH


# import side-effect — must run before any `import secfsdstools`
ensure_secfsdstools_cfg()
