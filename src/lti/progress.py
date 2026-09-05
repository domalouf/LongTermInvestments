"""A one-shot textual progress dashboard for the data-build pipeline.

``lti progress`` renders a simple ASCII bar for each stage so you can check how
far the (multi-hour) Phase 3 build has got without tailing logs.
"""

from __future__ import annotations

from dataclasses import dataclass

import lti.config as config


@dataclass
class Stage:
    label: str
    done: float
    total: float
    detail: str = ""

    @property
    def pct(self) -> float:
        return 0.0 if self.total <= 0 else max(0.0, min(1.0, self.done / self.total))

    def render(self, width: int = 28) -> str:
        filled = int(round(self.pct * width))
        bar = "#" * filled + "-" * (width - filled)
        return f"  {self.label:<16} [{bar}] {self.pct * 100:5.1f}%  {self.detail}"


def _count_dirs(path) -> int:
    try:
        return sum(1 for p in path.iterdir() if p.is_dir())
    except (FileNotFoundError, NotADirectoryError):
        return 0


def _count_processed_quarters() -> tuple[int, int]:
    try:
        from lti import sec_update

        names = sec_update.all_quarter_zip_names()
        return len(names), max(len(names), 1)
    except Exception:  # noqa: BLE001
        return 0, 1


def collect_stages() -> list[Stage]:
    paths = config.get_paths()
    stages: list[Stage] = []

    # 1. SEC raw data + index
    n_q, _ = _count_processed_quarters()
    latest = None
    try:
        from lti import sec_update

        latest = sec_update.latest_quarter()
    except Exception:  # noqa: BLE001
        pass
    stages.append(
        Stage("SEC data", 1 if n_q else 0, 1, f"{n_q} quarters processed"
              + (f", latest {latest}" if latest else ""))
    )

    # 2. standardization pipeline (skipped in smoke mode) — broken into sub-stages
    if config.is_smoke():
        stages.append(Stage("Std. pipeline", 1, 1, "skipped (smoke mode)"))
    else:
        n_total = max(n_q, 1)
        auto = paths.sec_automated
        n_filtered = _count_dirs(auto / "_1_by_quarter" / "_1_filtered_joined_by_stmt" / "quarter")
        n_std = _count_dirs(auto / "_1_by_quarter" / "_2_standardized_by_stmt")
        n_concat = sum(
            (p / "result.parquet").exists()
            for p in (paths.concat_std_bs, paths.concat_std_is, paths.concat_std_cf)
        )
        stages.append(Stage(" ├ filter", n_filtered, n_total, f"{n_filtered}/{n_total} quarters"))
        stages.append(Stage(" ├ standardize", n_std, n_total, f"{n_std}/{n_total} quarters"))
        stages.append(Stage(" └ concat", n_concat, 3, f"{n_concat}/3 merged bags (BS/IS/CF)"))

    # 3. fundamentals table
    fpath = paths.fundamentals_parquet
    if fpath.exists():
        try:
            import pyarrow.parquet as pq

            rows = pq.ParquetFile(fpath).metadata.num_rows
        except Exception:  # noqa: BLE001
            rows = "?"
        stages.append(Stage("Fundamentals", 1, 1, f"{rows} rows -> {fpath.name}"))
    else:
        stages.append(Stage("Fundamentals", 0, 1, "not built"))

    # 4. cik -> ticker map
    tpath = paths.cik_ticker_parquet
    stages.append(
        Stage("Ticker map", 1 if tpath.exists() else 0, 1, "present" if tpath.exists() else "not built")
    )

    # 5. prices vs universe
    try:
        from lti import prices

        meta = prices._load_meta()
        universe = _universe_size()
        n_ok = int((meta["status"] == "ok").sum()) if not meta.empty else 0
        n_seen = len(meta)
        stages.append(
            Stage(
                "Prices",
                n_seen,
                max(universe, n_seen, 1),
                f"{n_ok} ok / {n_seen} fetched / ~{universe} in universe",
            )
        )
    except Exception:  # noqa: BLE001
        stages.append(Stage("Prices", 0, 1, "no cache"))

    return stages


def _universe_size() -> int:
    paths = config.get_paths()
    if not paths.fundamentals_parquet.exists():
        return 0
    try:
        import pandas as pd

        # match what `lti fetch-prices` actually targets: a ticker AND shares data
        df = pd.read_parquet(paths.fundamentals_parquet, columns=["ticker", "shares_outstanding"])
        df = df[df["ticker"].notna() & df["shares_outstanding"].notna()]
        return int(df["ticker"].nunique())
    except Exception:  # noqa: BLE001
        return 0


def render(width: int = 28) -> str:
    header = "Pipeline progress" + ("  (smoke mode)" if config.is_smoke() else "")
    lines = [header, "=" * (len(header) + 2)]
    lines += [s.render(width) for s in collect_stages()]
    return "\n".join(lines)
