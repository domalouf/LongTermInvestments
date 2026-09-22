"""SIC-code classification, and the exclusions a capital-based screen needs.

Greenblatt's *The Little Book that Beats the Market* drops financials and
utilities from the universe. That is not squeamishness about the industries:
return on capital is meaningless for a bank, whose balance sheet *is* its
product, and regulated utilities earn an allowed return on a rate base, so
ranking them on ROC measures the regulator rather than the business.
"""

from __future__ import annotations

import re

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

# Commodity Contracts Brokers & Dealers — in practice almost entirely exchange-
# traded commodity, currency and crypto products (GLD, SLV, USO, IBIT, ...).
COMMODITY_POOL_SIC = 6221
# The code also covers a few real brokers and operating companies that have filed
# under it (StoneX, Seaboard, WisdomTree), so a filer only counts as a pool if its
# name says it is one.
_POOL_NAME = re.compile(
    r"\b(?:TRUST|FUNDS?|ETF|ETN|LP|L\.P\.?|PARTNERS)\b|SHARES\b|FUTURES|BITCOIN|ETHEREUM|CRYPTO|COMMODIT",
    re.IGNORECASE,
)


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


def is_investment_company(sic: pd.Series) -> pd.Series:
    """No SIC code at all. The SEC assigns none to business development companies
    and other registered investment companies — Ares Capital, FS KKR, Main Street,
    all 52 current tickered filers without one — which file 10-Ks like operating
    companies but are lending funds. Not a financial by SIC (a missing code stays
    unclassified), but excluded wherever financials are."""
    return pd.to_numeric(sic, errors="coerce").isna()


def is_utility(sic: pd.Series) -> pd.Series:
    """Electric, gas, water and sanitary services (SIC 4900-4999)."""
    codes = pd.to_numeric(sic, errors="coerce")
    return codes.between(*UTILITIES_RANGE).fillna(False).astype(bool)


def is_commodity_pool(sic: pd.Series, company: pd.Series) -> pd.Series:
    """Gold / silver / oil / currency / crypto trusts and funds that file 10-Ks.

    They report "earnings" (the mark-to-market on what they hold) and a share
    count, so they get a P/E like any company — and at the top of a gold rally
    that P/E sorts them straight to the top of a cheapness ranking. Not
    businesses; never part of the universe.
    """
    codes = pd.to_numeric(sic, errors="coerce")
    named = company.astype("string").str.contains(_POOL_NAME, na=False)
    return (codes == COMMODITY_POOL_SIC).fillna(False).astype(bool) & named.astype(bool)


def drop_financials(df: pd.DataFrame) -> pd.DataFrame:
    """Drop banks, insurers, REITs — and the BDCs that carry no SIC at all.

    The exclusion every screen that values an operating business wants, in one
    place. Lenient: a frame carrying neither ``is_financial`` nor ``sic`` comes
    back untouched. :func:`lti.ranking._drop_sector` is the strict version, which
    refuses rather than quietly answer a different question than it was asked.
    """
    if "is_financial" in df.columns:
        df = df[~df["is_financial"].fillna(False).astype(bool)]
    if "sic" in df.columns:  # BDCs and other investment companies carry no SIC
        df = df[~is_investment_company(df["sic"])]
    return df


def add_sector_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``sector``, ``is_financial`` and ``is_utility`` from an existing ``sic``."""
    if "sic" not in df.columns:
        return df
    df = df.copy()
    df["sector"] = sic_division(df["sic"])
    df["is_financial"] = is_financial(df["sic"])
    df["is_utility"] = is_utility(df["sic"])
    return df
