"""Fields the ``secfsdstools`` standardizers don't produce, read from the raw SEC data.

The standardized balance-sheet bag stops at the Assets / Liabilities / Equity
aggregates (see :data:`lti.fundamentals.BS_MAP`). That is not enough for
Greenblatt-style capital metrics: enterprise value needs interest-bearing debt,
and return on capital needs net fixed assets. Both live in the raw ``num.txt``.
SIC — needed for the financials / utilities exclusions — lives in ``sub.txt``.

Both files are already on disk as parquet under ``data/sec/parquet/quarter/``,
one directory per quarter zip, so this module reads them directly rather than
going back through ``secfsdstools``.

Results are cached in ``data/derived/`` because a full scan touches ~70 quarters.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import lti.config as config

LOGGER = logging.getLogger(__name__)

# --- raw us-gaap tags we pull -----------------------------------------------

# Net fixed assets for the return-on-capital denominator.
PPE_TAGS = ["PropertyPlantAndEquipmentNet"]

# Goodwill, so invested capital can exclude it (Greenblatt's ROC is about the
# capital the business actually needs, not what a past acquirer overpaid).
GOODWILL_TAGS = ["Goodwill"]

# Interest-bearing debt. No single tag is reliably present, so :func:`_derive_debt`
# combines them in two tiers:
#
#   * "roll-up" tags are alternative names for the *same* subtotal, so they are
#     coalesced — first one present wins;
#   * "instrument" tags are individual borrowings that a filer lists separately
#     instead of a subtotal, so they are summed, and only consulted when no
#     roll-up exists for that half of the balance sheet.
#
# Deliberately excluded: ``AvailableForSaleSecuritiesDebtSecurities*`` (an asset —
# debt the company *holds*, not owes), ``AssetRetirementObligation*`` (a
# provision, not borrowing) and ``DebtInstrumentUnamortizedDiscount`` (a contra
# adjustment, not a balance).
DEBT_NONCURRENT_ROLLUP_TAGS = [
    "LongTermDebtAndCapitalLeaseObligations",  # us-gaap defines this as noncurrent
    "LongTermDebtNoncurrent",
]
DEBT_NONCURRENT_INSTRUMENT_TAGS = [
    "ConvertibleDebtNoncurrent",
    "ConvertibleLongTermNotesPayable",
    "LongTermLineOfCredit",
    "SecuredLongTermDebt",
    "UnsecuredLongTermDebt",
    "LongTermNotesPayable",
    "LongTermLoansPayable",
    "OtherLongTermDebtNoncurrent",
    "OtherLongTermNotesPayable",
    "NotesPayableRelatedPartiesNoncurrent",
    "CapitalLeaseObligationsNoncurrent",
]
DEBT_CURRENT_LTD_TAGS = [
    "LongTermDebtAndCapitalLeaseObligationsCurrent",
    "LongTermDebtCurrent",
]
DEBT_CURRENT_OTHER_ROLLUP_TAGS = ["ShortTermBorrowings"]
DEBT_CURRENT_OTHER_INSTRUMENT_TAGS = [
    "CommercialPaper",
    "NotesPayableCurrent",
    "NotesAndLoansPayableCurrent",
    "LinesOfCreditCurrent",
    "LoansPayableCurrent",
    "LoansPayableToBankCurrent",
    "ShortTermBankLoansAndNotesPayable",
    "ConvertibleNotesPayableCurrent",
    "ConvertibleDebtCurrent",
    "SecuredDebtCurrent",
    "NotesPayableRelatedPartiesClassifiedCurrent",
    "CapitalLeaseObligationsCurrent",
]
DEBT_CURRENT_ROLLUP_TAGS = ["DebtCurrent"]
DEBT_TOTAL_TAGS = ["LongTermDebt"]  # us-gaap: includes both current and noncurrent

RAW_BS_TAGS: list[str] = [
    *PPE_TAGS,
    *GOODWILL_TAGS,
    *DEBT_NONCURRENT_ROLLUP_TAGS,
    *DEBT_NONCURRENT_INSTRUMENT_TAGS,
    *DEBT_CURRENT_LTD_TAGS,
    *DEBT_CURRENT_OTHER_ROLLUP_TAGS,
    *DEBT_CURRENT_OTHER_INSTRUMENT_TAGS,
    *DEBT_CURRENT_ROLLUP_TAGS,
    *DEBT_TOTAL_TAGS,
]


# --- quarter discovery ------------------------------------------------------


def _quarter_dirs() -> list[Path]:
    root = config.get_paths().sec_parquet / "quarter"
    if not root.exists():
        raise FileNotFoundError(f"{root} not found — run `lti update` first")
    return sorted(d for d in root.iterdir() if d.is_dir())


def _coalesce(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """First non-null value across ``cols``, in order. Missing columns are skipped."""
    present = [c for c in cols if c in df.columns]
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for c in present:
        out = out.fillna(df[c])
    return out


def _sum_present(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Sum of whichever of ``cols`` are reported; NaN when none of them are."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    block = df[present]
    return block.sum(axis=1, min_count=1)


# --- SIC --------------------------------------------------------------------


def build_sic_map(force: bool = False) -> pd.DataFrame:
    """``adsh -> sic`` for every indexed filing, cached to ``sic_by_adsh.parquet``."""
    out_path = config.get_paths().sic_parquet
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    frames = []
    for qdir in _quarter_dirs():
        sub = qdir / "sub.txt.parquet"
        if not sub.exists():
            LOGGER.warning("rawtags: %s has no sub.txt.parquet; skipping", qdir.name)
            continue
        frames.append(pd.read_parquet(sub, columns=["adsh", "sic"]))

    if not frames:
        raise RuntimeError("no sub.txt.parquet files found under data/sec/parquet/quarter")

    df = pd.concat(frames, ignore_index=True).dropna(subset=["adsh"])
    df = df.drop_duplicates("adsh", keep="last")
    df["sic"] = pd.to_numeric(df["sic"], errors="coerce").astype("Int32")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    LOGGER.info("rawtags: wrote %d adsh->sic rows -> %s", len(df), out_path)
    return df


# --- raw balance-sheet tags -------------------------------------------------


def _read_quarter_tags(qdir: Path, tags: list[str], qtrs: int) -> pd.DataFrame | None:
    """One row per ``adsh`` for ``tags``, at the filing's latest period.

    ``qtrs`` selects the duration: 0 for balance-sheet instants, 4 for a full year.
    """
    num = qdir / "num.txt.parquet"
    if not num.exists():
        LOGGER.warning("rawtags: %s has no num.txt.parquet; skipping", qdir.name)
        return None

    # 2009q1 is an empty placeholder whose columns are all null-typed, which the
    # pyarrow filter below cannot bind against. secfsdstools skips it too.
    import pyarrow.parquet as pq

    if pq.ParquetFile(num).metadata.num_rows == 0:
        return None

    df = pd.read_parquet(
        num,
        columns=["adsh", "tag", "ddate", "qtrs", "uom", "segments", "coreg", "value"],
        filters=[("tag", "in", tags), ("qtrs", "==", qtrs), ("uom", "==", "USD")],
    )
    if df.empty:
        return None

    # consolidated parent figures only: no dimensional breakdowns, no co-registrants
    df = df[df["segments"].isna() & df["coreg"].isna()]
    if df.empty:
        return None

    # every tag must come from the same balance-sheet date — num.txt also carries
    # the prior-year comparative column of the same filing
    df = df[df["ddate"] == df.groupby("adsh")["ddate"].transform("max")]
    df = df.drop_duplicates(["adsh", "tag"], keep="last")

    wide = df.pivot(index="adsh", columns="tag", values="value")
    wide["ddate_raw"] = df.groupby("adsh")["ddate"].first()
    return wide.reset_index()


def _derive_debt(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the raw debt tags into ``total_debt``.

    Each half of the balance sheet prefers a reported subtotal and only falls
    back to summing individual instruments (see the tag lists above). The total
    prefers the sum of the two halves, falls back to the ``LongTermDebt`` total
    tag when only one half is reported, and otherwise takes whichever half exists
    — an undercount, flagged by ``total_debt_partial``.

    Operating leases (ASC 842, 2019+) are deliberately excluded: they did not
    exist for the first decade of the sample and including them would make debt
    non-comparable across it. Capital/finance leases are included, since they
    were on-balance-sheet throughout.
    """
    noncurrent = _coalesce(df, DEBT_NONCURRENT_ROLLUP_TAGS).fillna(
        _sum_present(df, DEBT_NONCURRENT_INSTRUMENT_TAGS)
    )

    current_ltd = _coalesce(df, DEBT_CURRENT_LTD_TAGS)
    current_other = _coalesce(df, DEBT_CURRENT_OTHER_ROLLUP_TAGS).fillna(
        _sum_present(df, DEBT_CURRENT_OTHER_INSTRUMENT_TAGS)
    )
    current = _coalesce(df, DEBT_CURRENT_ROLLUP_TAGS).fillna(
        current_ltd.add(current_other, fill_value=0.0).where(
            current_ltd.notna() | current_other.notna()
        )
    )

    total_tag = _coalesce(df, DEBT_TOTAL_TAGS)

    both = noncurrent.notna() & current.notna()
    either = noncurrent.notna() | current.notna()

    total = pd.Series(np.nan, index=df.index, dtype="float64")
    total = total.mask(both, noncurrent.add(current, fill_value=0.0))
    total = total.fillna(total_tag)
    total = total.fillna(noncurrent.add(current, fill_value=0.0).where(either))

    df = df.copy()
    df["debt_noncurrent"] = noncurrent
    df["debt_current"] = current
    df["total_debt"] = total.clip(lower=0.0)
    # only one half reported and no total tag to fall back on -> known undercount
    df["total_debt_partial"] = either & ~both & total_tag.isna()
    return df


# A filing that reports no debt tag at all is usually debt-free rather than
# badly tagged — but only if its noncurrent liabilities are small relative to
# assets. Above this ratio the absence is treated as a tagging miss instead.
ZERO_DEBT_MAX_NONCURRENT_LIAB_RATIO = 0.05


def add_debt_provenance(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in genuinely debt-free filings and record where ``total_debt`` came from.

    Adds ``debt_source``:

    ``reported``
        at least one debt tag was present in the filing.
    ``assumed_zero``
        no debt tag, but noncurrent liabilities are under
        :data:`ZERO_DEBT_MAX_NONCURRENT_LIAB_RATIO` of assets — almost always a
        company with no borrowings, so ``total_debt`` is set to 0.
    ``unknown``
        no debt tag and material noncurrent liabilities: the filer used a tag we
        don't recognise. ``total_debt`` stays NaN so enterprise value refuses to
        quote a number rather than understating it.
    """
    if "total_debt" not in df.columns:
        return df

    df = df.copy()
    reported = df["total_debt"].notna()

    if {"liabilities_noncurrent", "liabilities", "liabilities_current", "assets"} <= set(df.columns):
        # The test only means something on a *classified* balance sheet. Filers
        # whose operating cycle runs over a year — homebuilders, most notably —
        # don't split current from noncurrent, and the standardizer then files
        # every liability as current with noncurrent left at zero. That looks
        # exactly like having no long-term borrowings, which is how a homebuilder
        # carrying a billion in senior notes ends up "debt-free".
        classified = (df["liabilities_noncurrent"] > 0) | (
            df["liabilities_current"] < df["liabilities"] * 0.999
        )
        ratio = df["liabilities_noncurrent"] / df["assets"].replace(0, np.nan)
        assumed_zero = (
            ~reported
            & classified.fillna(False)
            & ratio.notna()
            & (ratio < ZERO_DEBT_MAX_NONCURRENT_LIAB_RATIO)
        )
    else:
        assumed_zero = pd.Series(False, index=df.index)

    df["total_debt"] = df["total_debt"].mask(assumed_zero, 0.0)
    df["debt_source"] = np.select(
        [reported, assumed_zero], ["reported", "assumed_zero"], default="unknown"
    )
    return df


def build_raw_bs_tags(force: bool = False) -> pd.DataFrame:
    """Per-``adsh`` net PP&E, goodwill and interest-bearing debt from raw ``num.txt``.

    Cached to ``raw_bs_tags.parquet``; a full scan reads ~70 quarter files.
    """
    out_path = config.get_paths().raw_bs_tags_parquet
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    qdirs = _quarter_dirs()
    frames = []
    for i, qdir in enumerate(qdirs, 1):
        wide = _read_quarter_tags(qdir, RAW_BS_TAGS, qtrs=0)
        if wide is not None:
            frames.append(wide)
        if i % 10 == 0 or i == len(qdirs):
            LOGGER.info("rawtags: scanned %d/%d quarters", i, len(qdirs))

    if not frames:
        raise RuntimeError("no usable num.txt.parquet files found")

    df = pd.concat(frames, ignore_index=True)
    for col in RAW_BS_TAGS:
        if col not in df.columns:
            df[col] = np.nan
    # a filing can appear in more than one quarter zip; the later copy wins
    df = df.drop_duplicates("adsh", keep="last")
    df = _derive_debt(df)

    keep = [
        "adsh",
        "ddate_raw",
        "PropertyPlantAndEquipmentNet",
        "Goodwill",
        "debt_noncurrent",
        "debt_current",
        "total_debt",
        "total_debt_partial",
    ]
    df = df[keep].rename(
        columns={"PropertyPlantAndEquipmentNet": "ppe_net", "Goodwill": "goodwill"}
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    LOGGER.info("rawtags: wrote %d rows -> %s", len(df), out_path)
    return df


# --- raw income-statement tags ---------------------------------------------

# EBIT, as the filer actually tagged it. The standardized income statement
# always carries an ``OperatingIncomeLoss`` value, but when the filer didn't
# report one the standardizer *derives* it — and for filers whose income
# statement doesn't follow the usual shape (homebuilders, PEOs, integrated
# oil) the derivation is badly wrong, handing back near-100% operating margins.
# Where the filer did tag it, the standardizer passes it straight through, so
# the presence of this tag is the signal that EBIT can be trusted.
RAW_IS_TAGS: list[str] = ["OperatingIncomeLoss"]


def build_raw_is_tags(force: bool = False) -> pd.DataFrame:
    """Per-``adsh`` as-reported annual operating income from raw ``num.txt``."""
    out_path = config.get_paths().raw_is_tags_parquet
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    qdirs = _quarter_dirs()
    frames = []
    for i, qdir in enumerate(qdirs, 1):
        wide = _read_quarter_tags(qdir, RAW_IS_TAGS, qtrs=4)
        if wide is not None:
            frames.append(wide)
        if i % 10 == 0 or i == len(qdirs):
            LOGGER.info("rawtags: scanned %d/%d quarters (IS)", i, len(qdirs))

    if not frames:
        raise RuntimeError("no usable num.txt.parquet files found")

    df = pd.concat(frames, ignore_index=True)
    if "OperatingIncomeLoss" not in df.columns:
        df["OperatingIncomeLoss"] = np.nan
    df = df.drop_duplicates("adsh", keep="last")
    df = df[["adsh", "OperatingIncomeLoss"]].rename(
        columns={"OperatingIncomeLoss": "operating_income_reported"}
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    LOGGER.info("rawtags: wrote %d rows -> %s", len(df), out_path)
    return df
