"""Trading costs and taxes: what a backtest's return is worth to whoever trades it.

A gross backtest buys and sells at the close for free and never files a tax
return, which flatters a strategy that turns over every year against SPY, bought
once and held. :class:`Book` keeps one portfolio's positions as tax lots and
charges the two frictions that matter over a long horizon:

* **Trading costs.** Every dollar traded pays ``cost_bps`` of itself, one way —
  the half-spread plus any commission: a buy costs ``1 + c`` a dollar of stock,
  a sale brings in ``1 − c``. That is the price actually paid and received, so
  it is also what goes into the cost basis and the proceeds.
* **Taxes** (``tax=None`` is a tax-free account — an IRA, a 401(k)).
  Dividends are taxed as they arrive at ``TaxRates.dividends``, and only the
  after-tax part is reinvested. Gains are taxed when a sale realizes them —
  paid at once rather than the following April, which is slightly conservative
  — at ``short_term`` for a lot held a year or less and ``long_term`` for one
  held longer. Lots sell first in, first out, the US default. Losses offset
  gains of their own term first, then of the other, and what's left carries
  forward, keeping its term. The $3,000 a year of losses that can offset
  ordinary income is left out: it is a fixed sum, meaningless at an arbitrary
  portfolio size.

The dividend is inferred per holding as the gap between its total return and
its price return — ``adj_close`` against the split-adjusted ``close`` — so no
separate cash-flow bookkeeping is needed. State tax and the 3.8% net
investment income tax are not modelled: fold them into the rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# a middle estimate for the half-spread on the $500M+ companies these screens
# buy; large caps trade tighter, small caps wider
DEFAULT_COST_BPS = 10.0


@dataclass(frozen=True)
class TaxRates:
    """Federal rates for a middle bracket, no state tax: set your own."""

    short_term: float = 0.24  # held a year or less: ordinary income
    long_term: float = 0.15   # held more than a year
    dividends: float = 0.15   # qualified dividends


@dataclass
class Trade:
    """What one rebalance traded, and what that cost."""

    value_before: float
    bought: float  # at the mid price
    sold: float
    cost: float
    tax: float  # on the gains the sales realized
    short_term_gain: float  # realized, net of losses of the same term
    long_term_gain: float

    @property
    def turnover(self) -> float:
        """One-way: the share of the portfolio replaced (a full swap is 1.0)."""
        return (self.bought + self.sold) / 2 / self.value_before if self.value_before > 0 else float("nan")


class Book:
    """One portfolio as cash plus tax lots, rebalanced and marked forward.

    The lots are kept sorted by ticker, then purchase date, so a sale can take
    the oldest first with one cumulative sum.
    """

    def __init__(self, cash: float, cost_bps: float = DEFAULT_COST_BPS, tax: TaxRates | None = None):
        self.cash = float(cash)
        self.cost = cost_bps / 1e4
        self.tax = tax
        self.lots = pd.DataFrame(
            {
                "ticker": pd.Series(dtype=object),
                "acquired": pd.Series(dtype="datetime64[ns]"),
                "basis": pd.Series(dtype="float64"),
                "value": pd.Series(dtype="float64"),
            }
        )
        # unused capital losses, by term, as positive amounts
        self.carry_short = 0.0
        self.carry_long = 0.0

    @property
    def value(self) -> float:
        return self.cash + float(self.lots["value"].sum())

    def positions(self) -> pd.Series:
        """Market value by ticker."""
        return self.lots.groupby("ticker", sort=False)["value"].sum()

    def _gains_tax(self, short: float, long: float) -> tuple[float, float, float]:
        """Tax on a rebalance's net gains by term, and the carry-forwards it leaves."""
        if self.tax is None:
            return 0.0, 0.0, 0.0
        short -= self.carry_short
        long -= self.carry_long
        if short < 0 < long:
            long, short = long + short, 0.0
        elif long < 0 < short:
            short, long = short + long, 0.0
        due = max(short, 0.0) * self.tax.short_term + max(long, 0.0) * self.tax.long_term
        return due, max(-short, 0.0), max(-long, 0.0)

    def rebalance(self, date, weights: pd.Series) -> Trade:
        """Trade to ``weights`` (by ticker) of what's left after paying for the trades.

        The costs and taxes come out of the portfolio, so the amount to invest
        depends on the trades it takes to get there. The shortfall is close to
        linear in that amount, so a secant search settles it in a few steps;
        whatever it leaves over, a fraction of a cent, stays as cash.
        """
        date = pd.Timestamp(date)
        c = self.cost
        w = weights[weights > 0].groupby(level=0).sum()
        w = w / w.sum()
        before = self.value

        lots = self.lots
        names = pd.Index(lots["ticker"].unique()).union(w.index)
        codes = names.get_indexer(lots["ticker"])
        value = lots["value"].to_numpy()
        basis = lots["basis"].to_numpy()
        held = np.bincount(codes, weights=value, minlength=len(names))
        # the value of the ticker's earlier lots — what a sale uses up before reaching this one
        ahead = lots.groupby("ticker", sort=False)["value"].cumsum().to_numpy() - value
        is_long = (lots["acquired"] < date - pd.DateOffset(years=1)).to_numpy()  # held more than a year
        target_w = w.reindex(names, fill_value=0.0).to_numpy()

        def trade(invested: float):
            diff = target_w * invested - held
            buys, sells = np.clip(diff, 0.0, None), np.clip(-diff, 0.0, None)
            taken = np.clip(sells[codes] - ahead, 0.0, value)
            basis_sold = np.divide(taken * basis, value, out=np.zeros_like(value), where=value > 0)
            gain = taken * (1 - c) - basis_sold
            short, long = float(gain[~is_long].sum()), float(gain[is_long].sum())
            due, carry_short, carry_long = self._gains_tax(short, long)
            shortfall = self.cash + sells.sum() * (1 - c) - due - buys.sum() * (1 + c)
            return shortfall, (buys, sells, taken, basis_sold, short, long, due, carry_short, carry_long)

        tol = 1e-10 * max(before, 1.0)
        x_prev, (f_prev, out) = before, trade(before)
        x = before + f_prev / (1 + c)  # the shortfall moves about a dollar per dollar invested
        for _ in range(50):
            if abs(f_prev) <= tol:
                break
            f, out_x = trade(x)
            slope = (f - f_prev) / (x - x_prev) if x != x_prev else -1.0
            x_prev, f_prev, out = x, f, out_x
            x -= f / (slope if slope < 0 else -1.0)
        buys, sells, taken, basis_sold, short, long, due, self.carry_short, self.carry_long = out

        kept = lots.assign(value=value - taken, basis=basis - basis_sold)
        kept = kept[kept["value"] > 1e-9 * max(before, 1.0)]
        bought = buys > 0
        new = pd.DataFrame(
            {"ticker": names[bought], "acquired": date, "basis": buys[bought] * (1 + c), "value": buys[bought]}
        )
        frames = [f for f in (kept, new) if not f.empty]
        if frames:
            self.lots = pd.concat(frames, ignore_index=True).sort_values(
                ["ticker", "acquired"], kind="stable", ignore_index=True
            )
        else:
            self.lots = self.lots.iloc[0:0]
        self.cash = f_prev
        return Trade(before, float(buys.sum()), float(sells.sum()), c * float(buys.sum() + sells.sum()), due, short, long)

    def grow(self, total: pd.Series, price: pd.Series) -> float:
        """Mark every lot forward and tax the dividends it earned on the way.

        ``total`` is the growth of $1 in each ticker including dividends, ``price``
        the same without them; the gap is the dividend. A ticker missing from
        ``total`` stays flat, one missing from ``price`` is taken to have paid
        nothing. Returns the dividend tax paid.
        """
        if self.lots.empty:
            return 0.0
        g = self.lots["ticker"].map(total).fillna(1.0).to_numpy(dtype="float64")
        p = self.lots["ticker"].map(price).to_numpy(dtype="float64")
        p = np.where(np.isnan(p), g, p)
        v0 = self.lots["value"].to_numpy()
        dividends = v0 * np.clip(g - p, 0.0, None)
        tax = dividends * (self.tax.dividends if self.tax else 0.0)
        # what's reinvested — the dividend after its tax — is new cost basis
        self.lots["value"] = v0 * g - tax
        self.lots["basis"] = self.lots["basis"].to_numpy() + dividends - tax
        return float(tax.sum())

    def liquidation_value(self, date) -> float:
        """The cash if everything were sold on ``date``: less the cost of selling and the tax."""
        date = pd.Timestamp(date)
        value = self.lots["value"].to_numpy()
        gain = value * (1 - self.cost) - self.lots["basis"].to_numpy()
        long = (self.lots["acquired"] < date - pd.DateOffset(years=1)).to_numpy()
        due, _, _ = self._gains_tax(float(gain[~long].sum()), float(gain[long].sum()))
        return self.cash + float(value.sum()) * (1 - self.cost) - due
