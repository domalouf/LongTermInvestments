"""Shared visual language for the Streamlit app.

One place for the palette, the Plotly chrome and the page furniture, so every
page reads as the same product. Import :func:`configure_page` at the top of a
page and use :func:`show` instead of ``st.plotly_chart``.

The palette is the validated dark-mode categorical set (blue, orange, aqua,
yellow, magenta, ...) checked against this app's actual surface, ``#0e1117``:
all five slots clear the lightness band, chroma floor, adjacent-pair
colour-vision separation and 3:1 contrast. Charts here carry at most three
series, and the first three slots additionally clear the stricter all-pairs
gate, so scatter and bubble forms are safe too. Don't add a sixth hue — fold the
tail into "other" or facet instead.
"""

from __future__ import annotations

import streamlit as st

# --- palette ----------------------------------------------------------------

SERIES: list[str] = [
    "#3987e5",  # 1 blue
    "#d95926",  # 2 orange
    "#199e70",  # 3 aqua
    "#c98500",  # 4 yellow
    "#d55181",  # 5 magenta
]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = SERIES

# Diverging poles — warm/cool, so the midpoint reads as "nothing".
POS, NEG = "#3987e5", "#e66767"
MID = "#383835"

# One hue, light -> dark, for continuous magnitude only.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Status — reserved; never a series colour, always with a word beside it.
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"

# Chrome
SURFACE = "#0e1117"
CARD = "#161a22"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

_CSS = f"""
<style>
  .block-container {{ padding-top: 2.6rem; padding-bottom: 4rem; max-width: 1500px; }}
  h1, h2, h3 {{ letter-spacing: -0.015em; }}
  h1 {{ font-size: 1.9rem !important; font-weight: 650 !important; }}
  h2 {{ font-size: 1.22rem !important; font-weight: 620 !important;
        margin-top: 2.1rem !important; padding-bottom: .45rem;
        border-bottom: 1px solid {GRID}; }}
  h3 {{ font-size: 1.02rem !important; font-weight: 600 !important; }}

  /* page lede under the title */
  .lti-lede {{ color: {INK_2}; font-size: .95rem; line-height: 1.55;
               max-width: 74ch; margin: -.35rem 0 .4rem; }}
  .lti-note {{ color: {MUTED}; font-size: .82rem; line-height: 1.5;
               max-width: 84ch; margin: .1rem 0 .9rem; }}

  /* stat tiles */
  div[data-testid="stMetric"] {{
      background: {CARD}; border: 1px solid {GRID};
      border-radius: 10px; padding: .85rem 1rem .7rem;
  }}
  div[data-testid="stMetricLabel"] p {{
      color: {MUTED} !important; font-size: .74rem !important;
      text-transform: uppercase; letter-spacing: .07em; font-weight: 600 !important;
  }}
  div[data-testid="stMetricValue"] {{ font-size: 1.75rem !important; font-weight: 600 !important; }}

  /* tables: tabular figures so columns line up */
  div[data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}

  section[data-testid="stSidebar"] {{ border-right: 1px solid {GRID}; }}
  section[data-testid="stSidebar"] h2 {{ border-bottom: none; margin-top: .6rem !important; }}

  div[data-testid="stExpander"] details {{
      border: 1px solid {GRID}; border-radius: 10px; background: {CARD};
  }}
  hr {{ border-color: {GRID}; }}
</style>
"""


def configure_page(title: str, icon: str, *, layout: str = "wide") -> None:
    """``st.set_page_config`` plus the shared stylesheet. Call first, once."""
    st.set_page_config(page_title=title, page_icon=icon, layout=layout)
    st.markdown(_CSS, unsafe_allow_html=True)


def header(title: str, lede: str | None = None, note: str | None = None) -> None:
    """Page title, a one-line explanation, and optional smaller print."""
    st.title(title)
    if lede:
        st.markdown(f'<p class="lti-lede">{lede}</p>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<p class="lti-note">{note}</p>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<p class="lti-note">{text}</p>', unsafe_allow_html=True)


# --- plotly -----------------------------------------------------------------


def style(fig, *, height: int | None = None, legend: bool | None = None, **layout):
    """Apply the shared chart chrome: recessive grid, muted axes, no chartjunk."""
    n_traces = len(fig.data)
    show_legend = legend if legend is not None else n_traces >= 2

    fig.update_layout(
        template="plotly_dark",
        font=dict(family=FONT, size=12, color=INK_2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # top margin leaves room for the legend at y=1.01 and for reference-line
        # annotations anchored to the top of the plot area
        margin=dict(l=8, r=16, t=30, b=8),
        showlegend=show_legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11.5, color=INK_2),
            title=dict(text=""),
        ),
        hoverlabel=dict(
            font=dict(family=FONT, size=12, color=INK),
            bgcolor=CARD, bordercolor=AXIS,
        ),
        colorway=SERIES,
        separators=".,",
    )
    if height is not None:
        fig.update_layout(height=height)

    axis = dict(
        showgrid=True, gridcolor=GRID, gridwidth=1, griddash="solid",
        zeroline=False, showline=False,
        tickfont=dict(size=11, color=MUTED),
        title=dict(font=dict(size=11.5, color=MUTED)),
        automargin=True,
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    fig.update_layout(**layout)
    return fig


def show(fig, *, height: int | None = None, legend: bool | None = None, **layout) -> None:
    """:func:`style` the figure and render it full-width."""
    st.plotly_chart(
        style(fig, height=height, legend=legend, **layout),
        width="stretch",
        config={"displayModeBar": False},
    )


def bar_marks(fig, color: str | None = BLUE, *, radius: int = 4, gap: float = 0.34):
    """Thin bars with rounded data-ends and a surface gap between neighbours.

    ``color=None`` keeps whatever colour the trace already carries — a bar chart
    that colours each bar by its own sign sets that when it builds the trace.
    """
    if color is not None:
        fig.update_traces(marker_color=color)
    fig.update_traces(marker_line_width=0)
    try:  # cornerradius needs a recent plotly; not worth failing a page over
        fig.update_traces(marker_cornerradius=radius)
    except (ValueError, TypeError):
        pass
    fig.update_layout(bargap=gap, bargroupgap=0.12)
    return fig


def zero_line(fig, *, axis: str = "y", value: float = 0.0):
    """A hairline at zero — the reference a signed chart is read against."""
    kw = dict(line_width=1, line_color=AXIS, layer="below")
    if axis == "y":
        fig.add_hline(y=value, **kw)
    else:
        fig.add_vline(x=value, **kw)
    return fig


# --- formatting -------------------------------------------------------------


def money(x: float) -> str:
    """Compact dollars: $1.2B, $340M, $12.30."""
    if x is None or x != x:
        return "—"
    a = abs(x)
    if a >= 1e12:
        return f"${x / 1e12:,.1f}T"
    if a >= 1e9:
        return f"${x / 1e9:,.1f}B"
    if a >= 1e6:
        return f"${x / 1e6:,.0f}M"
    return f"${x:,.2f}"
