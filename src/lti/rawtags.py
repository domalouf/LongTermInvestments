"""Fields the ``secfsdstools`` standardizers don't produce, read from the raw SEC data.

The standardized balance-sheet bag stops at the Assets / Liabilities / Equity
aggregates (see :data:`lti.fundamentals.BS_MAP`). That is not enough for
Greenblatt-style capital metrics: enterprise value needs interest-bearing debt,
and return on capital needs net fixed assets. Both live in the raw ``num.txt``,
as do the share counts the standardized income statement misses for filers that
footnote their weighted average. SIC — needed for the financials / utilities
exclusions — lives in ``sub.txt``.

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


# --- share counts -------------------------------------------------------------

# The standardized income statement carries a weighted-average share count only
# when the filer presents one on the *face* of the income statement, and the SEC
# financial statement data sets hold nothing from the footnotes. Procter & Gamble,
# Chevron, Merck, Alphabet and plenty more put it in the EPS footnote — which left
# about a quarter of $10B+ companies with no share count and so no market cap. The
# balance sheet (or the common-stock column of the equity statement beside it)
# still carries the count at the period end.
# Partnerships (Energy Transfer, Enterprise Products, Plains) count units, not shares.
SHARE_OUTSTANDING_TAGS = ["CommonStockSharesOutstanding", "LimitedPartnersCapitalAccountUnitsOutstanding"]
SHARE_ISSUED_TAGS = ["CommonStockSharesIssued", "LimitedPartnersCapitalAccountUnitsIssued"]
SHARE_TREASURY_TAGS = ["TreasuryStockCommonShares", "TreasuryStockShares"]
SHARE_INSTANT_TAGS = [*SHARE_OUTSTANDING_TAGS, *SHARE_ISSUED_TAGS, *SHARE_TREASURY_TAGS]
SHARE_WAVG_TAGS = [
    "WeightedAverageNumberOfSharesOutstandingBasic",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageLimitedPartnershipUnitsOutstanding",
]

# the common-stock (or common-unit) column of the equity statement: the whole count
_EQUITY_COMPONENT = (
    r"(?:EquityComponents=Common(?:Stock|Units)[A-Za-z]*|LimitedPartnersCapitalAccountByClass=CommonUnits);?"
)
_CLASS_OF_STOCK = r"ClassOfStock=([^;]+)"


def _share_kind(segments: pd.Series) -> pd.Series:
    """``total`` (no dimension), ``equity`` (the common-stock column of the equity
    statement — the same count), ``class`` (one share class) or ``other``."""
    seg = segments.fillna("").astype(str)
    return pd.Series(
        np.select(
            [seg.eq(""), seg.str.fullmatch(_EQUITY_COMPONENT), seg.str.contains("ClassOfStock=", regex=False)],
            ["total", "equity", "class"],
            default="other",
        ),
        index=segments.index,
    )


def _summarize_shares(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per ``adsh`` from raw share-count facts (``adsh, tag, segments, value``),
    each tag already at its latest date.

    ``shares_bs``
        shares outstanding at the balance-sheet date: the dimensionless value,
        else the equity statement's common-stock column, else issued less
        treasury. Zero counts as missing — filers use it for a class they list
        but have none of.
    ``shares_bs_direct``
        ``shares_bs`` was tagged as outstanding rather than worked out from the
        issued count — which overstates it whenever the treasury shares went
        untagged (Boeing's issued count is 1.01B; about 760M are outstanding).
    ``shares_wavg``
        weighted-average basic (else diluted) shares for the year.
    ``shares_multi_class``
        the filing lists two or more share classes, so a single total may mix
        classes with different economic weights. A filing that lists exactly
        one class (Baker Hughes' Class A) is single-class: that class is the total.
    """
    rows = rows.assign(kind=_share_kind(rows["segments"]))

    classes = rows[(rows["kind"] == "class") & rows["tag"].isin([*SHARE_OUTSTANDING_TAGS, *SHARE_ISSUED_TAGS])]
    classes = classes.assign(share_class=classes["segments"].astype(str).str.extract(_CLASS_OF_STOCK, expand=False))
    n_classes = classes.groupby("adsh")["share_class"].nunique()
    sole_class = classes[classes["adsh"].map(n_classes).eq(1)]

    # a dimensionless value first, then the equity statement's common-stock column
    # (the same shares), then a filing's one and only class
    whole = pd.concat([rows[rows["kind"].isin(["total", "equity"])], sole_class], ignore_index=True)
    whole["_rank"] = whole["kind"].map({"total": 0, "equity": 1, "class": 2})
    whole = whole.sort_values(["adsh", "tag", "_rank", "value"], ascending=[True, True, True, False])
    wide = whole.drop_duplicates(["adsh", "tag"]).pivot(index="adsh", columns="tag", values="value")

    outstanding = _coalesce(wide, SHARE_OUTSTANDING_TAGS)
    issued = _coalesce(wide, SHARE_ISSUED_TAGS)
    treasury = _coalesce(wide, SHARE_TREASURY_TAGS)
    net_issued = issued - treasury.fillna(0.0)
    shares_bs = outstanding.where(outstanding > 0).fillna(net_issued.where(net_issued > 0))

    out = pd.DataFrame(index=wide.index.union(n_classes.index))
    out["shares_bs"] = shares_bs.reindex(out.index)
    out["shares_bs_direct"] = (outstanding > 0).reindex(out.index, fill_value=False).astype(bool)
    out["shares_wavg"] = _coalesce(wide, SHARE_WAVG_TAGS).reindex(out.index)
    out["shares_multi_class"] = n_classes.reindex(out.index).fillna(0).ge(2)
    return out.rename_axis("adsh").reset_index()


def _read_quarter_shares(qdir: Path) -> pd.DataFrame | None:
    num = qdir / "num.txt.parquet"
    if not num.exists():
        LOGGER.warning("rawtags: %s has no num.txt.parquet; skipping", qdir.name)
        return None

    import pyarrow.parquet as pq

    if pq.ParquetFile(num).metadata.num_rows == 0:
        return None

    df = pd.read_parquet(
        num,
        columns=["adsh", "tag", "ddate", "qtrs", "uom", "segments", "coreg", "value"],
        filters=[("tag", "in", SHARE_INSTANT_TAGS + SHARE_WAVG_TAGS), ("uom", "==", "shares")],
    )
    df = df[df["coreg"].isna()]
    instant = df["tag"].isin(SHARE_INSTANT_TAGS) & (df["qtrs"] == 0)
    duration = df["tag"].isin(SHARE_WAVG_TAGS) & (df["qtrs"] == 4)
    df = df[instant | duration]
    if df.empty:
        return None
    # each tag at its latest date: the balance-sheet date, or the year just ended —
    # the equity statement also carries opening balances, the income statement prior years
    df = df[df["ddate"] == df.groupby(["adsh", "tag"])["ddate"].transform("max")]
    return _summarize_shares(df[["adsh", "tag", "segments", "value"]])


def build_raw_share_tags(force: bool = False) -> pd.DataFrame:
    """Per-``adsh`` share counts from raw ``num.txt`` (see :func:`_summarize_shares`)."""
    out_path = config.get_paths().raw_share_tags_parquet
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    qdirs = _quarter_dirs()
    frames = []
    for i, qdir in enumerate(qdirs, 1):
        summary = _read_quarter_shares(qdir)
        if summary is not None:
            frames.append(summary)
        if i % 10 == 0 or i == len(qdirs):
            LOGGER.info("rawtags: scanned %d/%d quarters (shares)", i, len(qdirs))

    if not frames:
        raise RuntimeError("no usable num.txt.parquet files found")

    # a filing can appear in more than one quarter zip; the later copy wins
    df = pd.concat(frames, ignore_index=True).drop_duplicates("adsh", keep="last")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    LOGGER.info("rawtags: wrote %d rows -> %s", len(df), out_path)
    return df


# Two share counts agree within this factor: wide enough for a weighted average
# against a year-end count in a year of heavy buybacks or issuance ...
SHARE_AGREEMENT = 1.5
# ... and are a scale error apart beyond this one — a count or an EPS tagged in
# thousands or millions, which is how the real failures look.
SHARE_SCALE_ERROR = 10.0
# Net income / EPS only stands in for a share count when EPS is big enough that
# its rounding to the cent doesn't matter.
IMPLIED_MIN_ABS_EPS = 0.10


def _fold(a: pd.Series, b: pd.Series) -> pd.Series:
    """``max(a/b, b/a)`` — how many times apart two positive numbers are; NaN if either isn't."""
    r = a.where(a > 0) / b.where(b > 0)
    return np.maximum(r, 1.0 / r)


def _power_of_1000_apart(ratio: pd.Series) -> pd.Series:
    """Whether a positive ratio sits within 25% of 1000, a million, a thousandth, ...
    — the fingerprint of a value tagged in the wrong unit."""
    lr = np.log10(ratio.where(ratio > 0))
    k = np.round(lr / 3)
    return ((k != 0) & ((lr - 3 * k).abs() <= np.log10(1.25))).fillna(False)


def reconcile_shares(df: pd.DataFrame) -> pd.DataFrame:
    """One share count per filing, cross-checked across three sources, and EPS checked against it.

    * **reported** — the weighted average from the income statement (the
      standardized value, else the raw tag): the count EPS divides by;
    * **balance sheet** — shares outstanding at the period end (``shares_bs``);
    * **implied** — net income / EPS.

    Every source has real errors: counts tagged in thousands or millions
    (Bruker's weighted average is 146, Waters' balance-sheet count 59,388),
    placeholder zeros, and EPS off by a factor of a million (Halliburton's
    2,930,000). So a count is kept when a second source backs it up, and when
    the reported count is the odd one out and the other two agree, it is
    replaced. Net income / EPS only ever *confirms*: net income is before
    preferred dividends, so for a company with large ones it sits far from the
    count without anything being wrong.

    When the reported and balance-sheet counts are more than a scale error
    apart with nothing to break the tie, the count is left NaN — a missing
    market cap drops a company from a screen, a wrong one sorts it to the top.
    A moderate gap (a weighted average against the year-end count in an IPO
    year) keeps the reported count.

    Without a reported count: the balance-sheet count, confirmed by net income
    / EPS where possible; net income / EPS instead when the balance-sheet count
    was worked out from an issued count (treasury may be untagged) or the filing
    lists several share classes (a single total can add classes of different
    economic weight; net income / EPS counts in units of the share EPS is quoted for).

    Finally EPS × shares is checked against net income. A gap of a clean power
    of 1000 is a unit error — Halliburton's EPS, AMTX's share counts (both of
    them, so they "agree"), or a net income tagged in thousands — and nothing in
    the filing says which of the three it is. So the share count and EPS are
    both left NaN rather than one of them "corrected" into a new wrong number.

    Adds ``shares_source`` (``reported``, ``balance_sheet``, ``implied``,
    ``conflict`` or ``missing``) and ``eps_source`` (``reported``, ``conflict``
    or ``missing``); the inputs survive as ``shares_reported`` and ``eps_reported``.
    """
    out = df.copy()
    nan = pd.Series(np.nan, index=out.index, dtype="float64")

    def col(name: str) -> pd.Series:
        return out[name].astype("float64") if name in out.columns else nan

    reported = col("shares_outstanding")
    w = reported.where(reported > 0).fillna(col("shares_wavg").where(col("shares_wavg") > 0))
    b = col("shares_bs").where(col("shares_bs") > 0)
    direct = (
        out["shares_bs_direct"].fillna(False).astype(bool)
        if "shares_bs_direct" in out.columns
        else pd.Series(True, index=out.index)
    )
    multi = (
        out["shares_multi_class"].fillna(False).astype(bool)
        if "shares_multi_class" in out.columns
        else pd.Series(False, index=out.index)
    )
    eps, ni = col("eps"), col("net_income")
    implied = (ni / eps).where(eps.abs() >= IMPLIED_MIN_ABS_EPS)
    i = implied.where(implied > 0)

    def agree(x, y):
        return (_fold(x, y) <= SHARE_AGREEMENT).fillna(False)

    def apart(x, y):
        return (_fold(x, y) > SHARE_SCALE_ERROR).fillna(False)

    b_or_i = b.where(direct, i)  # when the two agree: the tagged count, else net income / EPS
    has_w = w.notna()
    rules = [
        # (condition, value, source) — the first match wins
        (has_w & (agree(w, b) | agree(w, i)), w, "reported"),
        (has_w & agree(b, i), b_or_i, "balance_sheet"),  # the reported count is the odd one out
        (has_w & apart(w, b), nan, "conflict"),
        (has_w, w, "reported"),  # a moderate gap: keep the count EPS divides by
        (agree(b, i), b_or_i, "balance_sheet"),
        (i.notna() & (multi | (b.notna() & ~direct)), i, "implied"),
        (b.notna(), b, "balance_sheet"),
        (i.notna(), i, "implied"),
    ]
    conds = [c.to_numpy() for c, _, _ in rules]
    shares = pd.Series(np.select(conds, [v.to_numpy() for _, v, _ in rules], default=np.nan), index=out.index)
    source = pd.Series(np.select(conds, [s for _, _, s in rules], default="missing"), index=out.index)
    # the "balance_sheet" rows that took net income / EPS say so
    source = source.mask((source == "balance_sheet") & shares.ne(b) & shares.eq(i), "implied")

    unit_error = _power_of_1000_apart(eps * shares / ni.where(ni != 0))

    out["shares_reported"] = reported
    out["shares_outstanding"] = shares.mask(unit_error)
    out["shares_source"] = source.mask(unit_error, "conflict")
    if "eps" in out.columns:
        out["eps_reported"] = eps
        out["eps"] = eps.mask(unit_error)
        eps_source = pd.Series(np.where(eps.notna(), "reported", "missing"), index=out.index)
        out["eps_source"] = eps_source.mask(unit_error, "conflict")
    return out
