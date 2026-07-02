
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# signal light semantics: red = before, amber = demand, green = after
C_BG      = "#0B1220"
C_SURFACE = "#131E2E"
C_BORDER  = "#1D2B3F"
C_RED     = "#E63946"
C_AMBER   = "#F4A825"
C_GREEN   = "#2EC27E"
C_TEXT    = "#E9EEF5"
C_MUTED   = "#8B9BB0"
C_TRACK   = "#22334A"

G_MIN = 0.12  # same safety floor as the engine

ARTERIES = [
    ("Batha",      "Batha"),
    ("Firyan",     "Firyan"),
    ("PrinceMohd", "Prince Mohammed"),
]

PERIOD_ORDER = ["Night (0-5)", "Morning peak (7-9)", "Off-peak", "Evening peak (16-20)"]

BASE_DIR  = Path(__file__).parent
DATA_FILE = BASE_DIR / "scats_results.csv"
IMG_FILE  = BASE_DIR / "study_area.png"


# ── data ──────────────────────────────────────────────────────────

def bucket_period(hour: int) -> str:
    # same bucketing as the engine so the numbers line up
    if 7 <= hour <= 9:
        return "Morning peak (7-9)"
    if 16 <= hour <= 20:
        return "Evening peak (16-20)"
    if 0 <= hour <= 5:
        return "Night (0-5)"
    return "Off-peak"


def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")  # engine writes a BOM
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["period"] = df["hour"].map(bucket_period)
    demand_cols = [f"demand_{k}" for k, _ in ARTERIES]
    # imbalance explains the gains better than peak/off-peak ever did
    df["imbalance"] = df[demand_cols].max(axis=1) - df[demand_cols].min(axis=1)
    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    tot_before = df["delay_before"].sum()
    tot_after  = df["delay_after"].sum()
    best_idx = int(df["improvement_pct"].idxmax())
    q75 = df["imbalance"].quantile(0.75)
    return {
        "overall":       (tot_before - tot_after) / tot_before * 100,
        "n_snapshots":   len(df),
        "n_days":        df["date"].nunique(),
        "best_pct":      float(df.loc[best_idx, "improvement_pct"]),
        "best_ts":       df.loc[best_idx, "timestamp"],
        "high_imb_mean": float(df.loc[df["imbalance"] >= q75, "improvement_pct"].mean()),
        "night_mean":    float(df.loc[df["period"] == "Night (0-5)", "improvement_pct"].mean()),
        "corr_imb":      float(df["imbalance"].corr(df["improvement_pct"])),
    }


def period_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("period")["improvement_pct"].agg(["mean", "size"])
    order = [p for p in PERIOD_ORDER if p in g.index]  # only periods that survived the filter
    g = g.reindex(order).reset_index()
    g.columns = ["period", "mean_improvement", "count"]
    return g


def hourly_heat(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="hour", columns="date",
                          values="improvement_pct", aggfunc="mean")


# ── charts ────────────────────────────────────────────────────────

FONT_STACK = "IBM Plex Sans, sans-serif"

def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0F1929",
        font=dict(family=FONT_STACK, color=C_TEXT, size=13),
        margin=dict(l=10, r=10, t=48, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_family=FONT_STACK),
    )
    fig.update_xaxes(gridcolor=C_BORDER, zerolinecolor=C_BORDER)
    fig.update_yaxes(gridcolor=C_BORDER, zerolinecolor=C_BORDER)
    return fig


def fig_timeline(df: pd.DataFrame, smooth: bool = False) -> go.Figure:
    if smooth:
        df = df.copy()
        for c in ("delay_before", "delay_after"):
            df[c] = df[c].rolling(5, center=True, min_periods=1).mean()
    fig = go.Figure()
    fig.add_scatter(x=df["timestamp"], y=df["delay_before"], name="Before (equal split)",
                    line=dict(color=C_RED, width=1.6),
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>before: %{y:.2f}<extra></extra>")
    fig.add_scatter(x=df["timestamp"], y=df["delay_after"], name="After (SCATS split)",
                    line=dict(color=C_GREEN, width=1.6),
                    fill="tonexty", fillcolor="rgba(46,194,126,0.10)",
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>after: %{y:.2f}<extra></extra>")
    fig.update_layout(title="Delay index before vs after full observation window",
                      yaxis_title="Delay index (relative)",
                      hovermode="x unified")
    fig.update_xaxes(
        rangeslider=dict(visible=True, bgcolor=C_SURFACE,
                         bordercolor=C_BORDER, borderwidth=1, thickness=0.09),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=3, label="3d", step="day", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor=C_SURFACE, activecolor=C_GREEN, font=dict(color=C_TEXT),
        ),
    )
    return style_fig(fig, height=430)


def fig_periods(stats: pd.DataFrame) -> go.Figure:
    colors = [C_MUTED if p.startswith("Night") else C_GREEN for p in stats["period"]]
    fig = go.Figure(go.Bar(
        x=stats["period"], y=stats["mean_improvement"],
        marker_color=colors,
        text=[f"{v:.1f}%" for v in stats["mean_improvement"]],
        textposition="outside", textfont=dict(size=14),
        customdata=stats["count"],
        hovertemplate="%{x}<br>avg improvement: %{y:.1f}%<br>snapshots: %{customdata}<extra></extra>",
    ))
    fig.update_layout(title="Average improvement by time of day",
                      yaxis_title="Improvement %",
                      yaxis_range=[0, stats["mean_improvement"].max() * 1.25])
    return style_fig(fig)


def fig_imbalance(df: pd.DataFrame) -> go.Figure:
    period_colors = {
        "Night (0-5)": C_MUTED, "Morning peak (7-9)": "#5AA9E6",
        "Off-peak": C_AMBER, "Evening peak (16-20)": "#C77DFF",
    }
    fig = go.Figure()
    for period in PERIOD_ORDER:
        sub = df[df["period"] == period]
        if sub.empty:
            continue
        fig.add_scatter(
            x=sub["imbalance"], y=sub["improvement_pct"], mode="markers",
            name=period,
            marker=dict(size=6, color=period_colors[period], opacity=0.75,
                        line=dict(width=0)),
            hovertemplate=("imbalance: %{x:.2f}<br>improvement: %{y:.1f}%<extra>" + period + "</extra>"),
        )
    fig.update_layout(
        title="The more unbalanced the arterials, the bigger the gain",
        xaxis_title="Demand imbalance (max − min share)",
        yaxis_title="Improvement %",
    )
    return style_fig(fig, height=420)


def fig_green_day(df: pd.DataFrame, day) -> go.Figure:
    sub = df[df["date"] == day]
    fig = go.Figure()
    fills = ["#1f7a53", C_GREEN, "#7be0b0"]
    for (key, name), color in zip(ARTERIES, fills):
        fig.add_scatter(
            x=sub["timestamp"], y=sub[f"green_{key}"], name=name,
            mode="lines", stackgroup="one", line=dict(width=0.5, color=color),
            hovertemplate="%{x|%H:%M}<br>" + name + ": %{y:.0%}<extra></extra>",
        )
    fig.add_hline(y=G_MIN, line_dash="dot", line_color=C_AMBER,
                  annotation_text="12% safety floor", annotation_font_color=C_AMBER)
    fig.update_layout(title=f"Green-time shares through {day}",
                      yaxis_title="Green share", yaxis_tickformat=".0%",
                      yaxis_range=[0, 1])
    return style_fig(fig)


def fig_heatmap(pivot: pd.DataFrame) -> go.Figure:
    # nan-safe scale (nan is truthy, so no `or` shortcuts here)
    zmax = max(45.0, float(np.nanmax(pivot.values))
               if pivot.size and not np.isnan(pivot.values).all() else 45.0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=pivot.index,
        colorscale=[[0, C_RED], [0.5, C_AMBER], [1, C_GREEN]],
        zmin=0, zmax=zmax,
        colorbar=dict(title="Improv. %", outlinewidth=0),
        hovertemplate="day: %{x}<br>hour: %{y}:00<br>improvement: %{z:.1f}%<extra></extra>",
        hoverongaps=False,
    ))
    fig.update_layout(title="Improvement heatmap: hour × day (blank cells = sampling outages)",
                      xaxis_title="Day", yaxis_title="Hour")
    fig.update_yaxes(dtick=2, autorange="reversed")
    return style_fig(fig, height=420)


def fig_signal_bars(row: pd.Series) -> go.Figure:
    """The signature piece: one snapshot, read like a signal cycle."""
    names  = [n for _, n in ARTERIES][::-1]
    greens = [row[f"green_{k}"]  for k, _ in ARTERIES][::-1]
    dems   = [row[f"demand_{k}"] for k, _ in ARTERIES][::-1]

    fig = go.Figure()
    # track = full cycle, green = allocated, diamond = demand
    fig.add_bar(y=names, x=[1] * len(names), orientation="h",
                marker_color=C_TRACK, width=0.55, hoverinfo="skip", showlegend=False)
    fig.add_bar(y=names, x=greens, orientation="h",
                marker=dict(color=C_GREEN, line=dict(color="#57E6A6", width=1)),
                width=0.55, name="Allocated green share",
                text=[f"{g:.0%}" for g in greens], textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color="#06281A", size=15, family=FONT_STACK),
                hovertemplate="%{y}<br>allocated: %{x:.0%}<extra></extra>")
    fig.add_scatter(y=names, x=dems, mode="markers", name="Observed demand",
                    marker=dict(symbol="diamond", size=15, color=C_AMBER,
                                line=dict(color="#7A5410", width=1.5)),
                    hovertemplate="%{y}<br>demand: %{x:.0%}<extra></extra>")
    fig.add_vline(x=G_MIN, line_dash="dot", line_color=C_MUTED,
                  annotation_text="12% floor", annotation_position="top",
                  annotation_font=dict(color=C_MUTED, size=12))
    fig.update_layout(
        barmode="overlay",
        title="Allocated green vs observed demand",
        xaxis=dict(tickformat=".0%", range=[0, 1.02], title="Share of signal cycle"),
        yaxis=dict(tickfont=dict(size=14)),
    )
    return style_fig(fig, height=330)


# ── page style ────────────────────────────────────────────────────

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, .stApp {{
    background:
      radial-gradient(1200px 500px at 85% -10%, rgba(46,194,126,.07), transparent 60%),
      radial-gradient(900px 400px at 10% 110%, rgba(244,168,37,.05), transparent 60%),
      {C_BG} !important;
    color: {C_TEXT};
    font-family: 'IBM Plex Sans', sans-serif;
}}
[data-testid="stSidebar"] {{
    background: {C_SURFACE};
    border-right: 1px solid {C_BORDER};
}}
h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif !important; }}

.signal-dots {{ display:flex; gap:8px; margin-bottom:10px; }}
.signal-dots span {{ width:12px; height:12px; border-radius:50%; display:inline-block; }}
.dot-r {{ background:{C_RED};   box-shadow:0 0 10px {C_RED}66; }}
.dot-a {{ background:{C_AMBER}; box-shadow:0 0 10px {C_AMBER}66; }}
.dot-g {{ background:{C_GREEN}; box-shadow:0 0 12px {C_GREEN}88; }}

.hero-title {{ font-family:'Space Grotesk'; font-weight:700; font-size:2.05rem;
               margin:0; line-height:1.3; }}
.hero-sub   {{ color:{C_MUTED}; font-size:1.02rem; margin-top:6px; }}
.hero-wrap  {{ padding: 6px 2px 2px; border-bottom:1px solid {C_BORDER};
               margin-bottom: 6px; }}

[data-testid="stMetric"] {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 14px;
    padding: 14px 16px;
}}
[data-testid="stMetricValue"] {{
    font-family:'Space Grotesk'; font-weight:700; font-size:1.9rem;
}}
[data-testid="stMetricLabel"] p {{ color:{C_MUTED}; font-size:.95rem; }}

.map-frame img {{
    border-radius: 14px;
    border: 1px solid {C_BORDER};
    box-shadow: 0 8px 30px rgba(0,0,0,.45);
}}
.legend-pill {{
    display:inline-block; padding:4px 12px; border-radius:999px;
    background:{C_SURFACE}; border:1px solid {C_BORDER};
    color:{C_MUTED}; font-size:.9rem; margin-right:8px;
}}
.legend-pill b.white  {{ color:#FFFFFF; }}
.legend-pill b.yellow {{ color:{C_AMBER}; }}

hr {{ border-color:{C_BORDER}; }}
</style>
"""


# ── app ───────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Smart Traffic Control. Riyadh",
                       page_icon="🚦", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    @st.cache_data(show_spinner="Loading engine results…")
    def _load(path_str: str) -> pd.DataFrame:
        return load_and_prepare(Path(path_str))

    if not DATA_FILE.exists():
        st.error(
            "Couldn't find **scats_results.csv** next to this file.\n\n"
            "Generate it first:\n"
            "`python scats_engine.py --input riyadh_traffic_links.csv --output scats_results.csv`"
        )
        st.stop()

    df = _load(str(DATA_FILE))

    need = {"timestamp", "delay_before", "delay_after", "improvement_pct"} | \
           {f"{p}_{k}" for k, _ in ARTERIES for p in ("green", "demand")}
    missing = need - set(df.columns)
    if missing:
        st.error(f"File found but missing columns: {sorted(missing)}  "
                 "make sure it's the untouched output of scats_engine.py.")
        st.stop()

    k_full = compute_kpis(df)  # full-period numbers (the official 34.7%)

    with st.sidebar:
        st.markdown("### ⚙️ View lens")
        d_min, d_max = df["date"].min(), df["date"].max()
        d_range = st.date_input("Date range", value=(d_min, d_max),
                                min_value=d_min, max_value=d_max)
        sel_periods = st.multiselect("Time-of-day periods", PERIOD_ORDER, default=PERIOD_ORDER)
        smooth = st.toggle("Smooth timeline (5-pt rolling mean)", value=True,
                           help="Visual smoothing only  the numbers are untouched.")

        st.divider()
        st.markdown("### 🧭 About")
        st.caption(
            "SCATS-style adaptive control simulation on three arterials in "
            "south-east Riyadh (Zone A). Data: periodic sampling every ~15 min "
            f"via the TomTom API over {k_full['n_days']} days."
        )
        st.caption("🔗 The research RAG (Ollama · Llama 3 + bge-m3) runs locally, "
                   "deliberately outside this dashboard.")
        with st.expander("▶️ How to run"):
            st.code("streamlit run dashboard.py", language="bash")

    view = df.copy()
    if isinstance(d_range, tuple) and len(d_range) == 2:
        view = view[(view["date"] >= d_range[0]) & (view["date"] <= d_range[1])]
    if sel_periods:
        view = view[view["period"].isin(sel_periods)]
    if view.empty:
        st.warning("No snapshots match these filters  widen the range in the sidebar.")
        st.stop()

    k = compute_kpis(view)  # KPIs follow the current filters

    st.markdown(
        """
        <div class="hero-wrap">
          <div class="signal-dots"><span class="dot-r"></span><span class="dot-a"></span><span class="dot-g"></span></div>
          <p class="hero-title">Smart Traffic Control  Riyadh · Zone A</p>
          <p class="hero-sub">Demand-proportional green-time allocation with a safety floor 
          a simulation built on real field data sampled every 15 minutes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_img, col_txt = st.columns([3, 2], gap="large")
    with col_img:
        if IMG_FILE.exists():
            st.markdown('<div class="map-frame">', unsafe_allow_html=True)
            st.image(str(IMG_FILE), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                '<span class="legend-pill"><b class="white">⬤ White</b> = intersections</span>'
                '<span class="legend-pill"><b class="yellow">⬤ Yellow</b> = standalone signal</span>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Drop **study_area.jpg** next to this file to show the study area here.")
    with col_txt:
        st.markdown("#### Study area")
        st.markdown(
            "Three intersecting arterials competing for one virtual signal cycle:\n"
            "- **Batha**  sharp peak spikes\n"
            "- **Firyan**  spillover receiver\n"
            "- **Prince Mohammed bin Abdulrahman**\n\n"
            "Congestion metric: `delay ÷ free-flow time`  a documented proxy "
            "for degree of saturation when the data has no vehicle counts."
        )

    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total delay index cut", f"{k['overall']:.1f}%",
              help=f"Under current filters, vs a naive equal split. "
                   f"Full-period figure: {k_full['overall']:.1f}%. Relative/illustrative.")
    c2.metric("Snapshots in view", f"{k['n_snapshots']:,}",
              f"{k['n_days']} days observed", delta_color="off")
    c3.metric("Best single-moment cut", f"{k['best_pct']:.1f}%",
              k["best_ts"].strftime("%m-%d %H:%M"), delta_color="off",
              help="The algorithm's strongest single reallocation.")
    c4.metric("Gain under high imbalance", f"{k['high_imb_mean']:.1f}%",
              f"+{k['high_imb_mean'] - k['night_mean']:.1f} pts vs balanced night",
              help="Top-quartile imbalance vs night  the proof that gains "
                   "follow imbalance, not peak hours.")

    st.markdown("---")

    st.markdown("### 🚦 Signal Timing Simulator  live the moment")
    st.caption("Drag to any observed snapshot and watch the algorithm's call: "
               "green bar = allocated share, amber diamond = observed demand.")

    view_idx = view.reset_index(drop=True)
    default_pos = int(view_idx["improvement_pct"].idxmax())  # start on the strongest moment
    if len(view_idx) > 1:
        pos = st.slider("Pick a snapshot", 0, len(view_idx) - 1, default_pos,
                        label_visibility="collapsed")
    else:
        pos = 0  # one snapshot left → no slider
    row = view_idx.loc[pos]

    st.markdown(f"**🕐 {row['timestamp']:%A %Y-%m-%d at %H:%M}**  {row['period']}")
    st.plotly_chart(fig_signal_bars(row), use_container_width=True,
                    config={"displaylogo": False})

    m1, m2, m3 = st.columns(3)
    m1.metric("Delay before", f"{row['delay_before']:.2f}")
    m2.metric("Delay after", f"{row['delay_after']:.2f}")
    m3.metric("Cut at this moment", f"{row['improvement_pct']:.1f}%",
              delta=f"-{row['delay_before'] - row['delay_after']:.2f} pts")

    st.markdown("---")

    st.plotly_chart(fig_timeline(view, smooth=smooth), use_container_width=True,
                    config={"displaylogo": False})

    r1, r2 = st.columns(2, gap="large")
    with r1:
        st.plotly_chart(fig_periods(period_stats(view)), use_container_width=True,
                        config={"displaylogo": False})
    with r2:
        st.plotly_chart(fig_imbalance(view), use_container_width=True,
                        config={"displaylogo": False})

    r3, r4 = st.columns(2, gap="large")
    with r3:
        busiest = view["date"].value_counts().idxmax()  # richest day makes the best default
        days = sorted(view["date"].unique())
        day = st.selectbox("Pick a day for the green-share breakdown", days,
                           index=days.index(busiest))
        st.plotly_chart(fig_green_day(view, day), use_container_width=True,
                        config={"displaylogo": False})
    with r4:
        st.plotly_chart(fig_heatmap(hourly_heat(view)), use_container_width=True,
                        config={"displaylogo": False})

    with st.expander("📄 Data in view (after filters)  inspect & download"):
        show_cols = ["timestamp", "period", "delay_before", "delay_after",
                     "improvement_pct", "imbalance"] + \
                    [f"green_{a}"  for a, _ in ARTERIES] + \
                    [f"demand_{a}" for a, _ in ARTERIES]
        st.dataframe(view[show_cols].round(3), use_container_width=True, height=320)
        st.download_button(
            "⬇️ Download CSV",
            data=view[show_cols].to_csv(index=False).encode("utf-8-sig"),
            file_name="scats_view.csv", mime="text/csv",
        )

    st.markdown("---")
    with st.expander("Notes"):
        st.markdown(
            f"""
1. **The congestion metric is a proxy.** The data is descriptive (travel time + delay),
   with no vehicle counts, so `delay ÷ free-flow time` stands in for degree of
   saturation. a direct transform of the peer-reviewed Travel Time Index.
2. **The {k_full['overall']:.1f}% cut (full period) is relative and illustrative**, based on a
   convex delay model and a {G_MIN:.0%} safety floor. not a field result. Real
   deployments typically achieve **10–20%**.
3. **Sampling is periodic, not continuous:** every ~15 min with a few off-peak
   outages shown honestly as blank cells in the heatmap.
4. **The key insight:** imbalance ↔ improvement correlation = **{k_full['corr_imb']:.2f}** 
   the algorithm helps most when approaches are unbalanced, not when everything
   is equally jammed.
            """
        )
    st.caption("Salem Alsairi. self-collected TomTom data · "
               "engine: scats_engine.py · dashboard: phase 4/4 ")


if __name__ == "__main__":
    main()
