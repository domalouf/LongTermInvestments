"""SIC-code classification, and the exclusions a capital-based screen needs.

Greenblatt's *The Little Book that Beats the Market* drops financials and
utilities from the universe. That is not squeamishness about the industries:
return on capital is meaningless for a bank, whose balance sheet *is* its
product, and regulated utilities earn an allowed return on a rate base, so
ranking them on ROC measures the regulator rather than the business.
"""

from __future__ import annotations

import pandas as pd

# SIC divisions, as published by the SEC. (low, high, label) — inclusive.
_DIVISIONS: list[tuple[int, int, str]] = [
    (100, 999, "Agriculture, Forestry & Fishing"),
    (1000, 1499, "Mining"),
    (1500, 1799, "Construction"),
    (2000, 3999, "Manufacturing"),
    (4000, 4999, "Transportation & Public Utilities"),
    (5000, 5199, "Wholesale Trade"),
    (5200, 5999, "Retail Trade"),
    (6000, 6799, "Finance, Insurance & Real Estate"),
    (7000, 8999, "Services"),
    (9100, 9729, "Public Administration"),
]

FINANCIALS_RANGE = (6000, 6799)
UTILITIES_RANGE = (4900, 4999)


def sic_division(sic: pd.Series) -> pd.Series:
    """Map SIC codes to their division label; unknown / missing codes give NA."""
    codes = pd.to_numeric(sic, errors="coerce")
    out = pd.Series(pd.NA, index=sic.index, dtype="string")
    for low, high, label in _DIVISIONS:
        # nullable dtypes make between() return NA for a missing code, and mask()
        # would treat that as a hit — every unclassified row would take the label
        # of whichever division happened to be tested last
        out = out.mask(codes.between(low, high).fillna(False).astype(bool), label)
    return out


def is_financial(sic: pd.Series) -> pd.Series:
    """SIC division H — banks, brokers, insurers, REITs and other holding companies."""
    codes = pd.to_numeric(sic, errors="coerce")
    return codes.between(*FINANCIALS_RANGE).fillna(False).astype(bool)


def is_utility(sic: pd.Series) -> pd.Series:
    """Electric, gas, water and sanitary services (SIC 4900-4999)."""
    codes = pd.to_numeric(sic, errors="coerce")
    return codes.between(*UTILITIES_RANGE).fillna(False).astype(bool)


def add_sector_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``sector``, ``is_financial`` and ``is_utility`` from an existing ``sic``."""
    if "sic" not in df.columns:
        return df
    df = df.copy()
    df["sector"] = sic_division(df["sic"])
    df["is_financial"] = is_financial(df["sic"])
    df["is_utility"] = is_utility(df["sic"])
    return df
