"""
app.py
======
Streamlit front-end for the AI Business Analyst Assistant.

Thin by design: all data + KPI logic lives in analytics.py. This file only
handles the UI, the OpenAI call, and presenting the computed evidence next to
the model's interpretation — the project's core idea made visible.
"""

import os
import streamlit as st
import altair as alt
from dotenv import load_dotenv
from openai import OpenAI

import analytics as an
import db
from pipeline import run_question, interpret_result

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Business Analyst Assistant",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# OpenAI key handling — dashboard works WITHOUT a key; only the AI Q&A needs it.
# ---------------------------------------------------------------------------
load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY")
try:
    if not _api_key and "OPENAI_API_KEY" in st.secrets:
        _api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass
if _api_key:
    _api_key = str(_api_key).strip().strip('"').strip("'")
    _api_key = _api_key.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")

AI_READY = bool(_api_key and _api_key.startswith("sk-"))


@st.cache_resource
def get_client(key):
    return OpenAI(api_key=key)


client = get_client(_api_key) if AI_READY else None


# ---------------------------------------------------------------------------
# Data (cached so groupby work runs once, not on every interaction)
# ---------------------------------------------------------------------------
@st.cache_data
def load():
    df = an.load_data()
    return df, an.build_aggregations(df)


@st.cache_resource
def get_warehouse():
    """DuckDB connection + schema description, built once and cached.
    Powers the text-to-SQL Q&A; the dashboard still uses analytics.py."""
    con = db.get_connection()
    return con, db.get_schema(con)


try:
    df, AGG = load()
    con, SCHEMA = get_warehouse()
except FileNotFoundError:
    st.error("Could not find data/business_data.csv. Run `python generate_data.py` first.")
    st.stop()

H = AGG["headline"]


def money(v):
    v = float(v)
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"${v/1e3:.0f}K"
    return f"${v:,.0f}"


# ---------------------------------------------------------------------------
# Visual system — "Consulting report": white, generous whitespace, hairline
# rules, a serif display face used sparingly, mono for every number, one ink
# navy accent. No emoji in the chrome, no rounded candy cards.
# ---------------------------------------------------------------------------
INK = "#16202c"
NAVY = "#1e3a5f"
MUTED = "#5b6776"
LINE = "#e3e4e0"
POS = "#2f7d5b"
NEG = "#b0432f"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {INK}; }}
[data-testid="stAppViewContainer"] {{ background: #ffffff; }}
[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }}

.eyebrow {{ font-family:'IBM Plex Mono',monospace; font-size:0.72rem; letter-spacing:0.18em;
           text-transform:uppercase; color:{MUTED}; margin-bottom:0.5rem; }}
.display {{ font-family:'Fraunces',serif; font-weight:600; font-size:2.55rem; line-height:1.08;
           letter-spacing:-0.01em; color:{INK}; margin:0 0 0.5rem 0; }}
.lede {{ font-size:1.02rem; color:{MUTED}; max-width:60ch; line-height:1.55; }}
.rule {{ border:0; border-top:1px solid {LINE}; margin:1.6rem 0; }}

.sect {{ font-family:'Fraunces',serif; font-weight:600; font-size:1.3rem; color:{INK};
         margin:0.2rem 0 0.2rem 0; }}
.sect-note {{ font-size:0.9rem; color:{MUTED}; margin-bottom:0.9rem; }}

.kpi {{ border:1px solid {LINE}; padding:1.05rem 1.15rem; background:#fff; height:100%; }}
.kpi-l {{ font-family:'IBM Plex Mono',monospace; font-size:0.68rem; letter-spacing:0.12em;
         text-transform:uppercase; color:{MUTED}; }}
.kpi-v {{ font-family:'IBM Plex Mono',monospace; font-size:1.62rem; font-weight:500;
         color:{INK}; margin-top:0.35rem; }}
.kpi-n {{ font-size:0.82rem; color:{MUTED}; margin-top:0.25rem; line-height:1.35; }}

.evidence-cap {{ font-family:'IBM Plex Mono',monospace; font-size:0.72rem; letter-spacing:0.1em;
                text-transform:uppercase; color:{NAVY}; margin:0.2rem 0 0.4rem 0; }}
.signal {{ border-left:2px solid {NAVY}; padding:0.15rem 0 0.15rem 0.85rem; margin-bottom:0.9rem; }}
.signal-t {{ font-family:'IBM Plex Mono',monospace; font-size:0.7rem; letter-spacing:0.1em;
            text-transform:uppercase; color:{MUTED}; }}
.signal-v {{ font-size:0.96rem; color:{INK}; margin-top:0.15rem; }}

div[data-testid="stButton"] > button {{ border-radius:0; border:1px solid {LINE};
    font-family:'Inter'; font-weight:500; color:{INK}; }}
div[data-testid="stButton"] > button:hover {{ border-color:{NAVY}; color:{NAVY}; }}
section[data-testid="stSidebar"] {{ background:#fafaf8; border-right:1px solid {LINE}; }}
.stCode, code {{ font-family:'IBM Plex Mono',monospace !important; }}
</style>
""", unsafe_allow_html=True)


def chart_base(c):
    return (c.configure_view(strokeWidth=0)
             .configure_axis(labelFont="Inter", titleFont="Inter", labelColor=MUTED,
                             titleColor=MUTED, grid=False, domainColor=LINE, tickColor=LINE)
             .configure_axisX(labelAngle=0))


# ===========================================================================
# Header
# ===========================================================================
st.markdown('<div class="eyebrow">APAC E-Commerce · Analytics Copilot</div>', unsafe_allow_html=True)
st.markdown('<div class="display">Ask a business question.<br>Get the numbers, then the read.</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="lede">Python and Pandas compute the KPI evidence first. The language model '
    'only interprets numbers it is given — it never guesses them. Every answer below shows the '
    'computed evidence alongside the interpretation, so you can check the work.</div>',
    unsafe_allow_html=True)
st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ===========================================================================
# KPI strip
# ===========================================================================
k1, k2, k3, k4 = st.columns(4)
for col, label, val, note in [
    (k1, "Total Revenue", money(H["total_revenue"]), f'{H["n_months"]} months · {H["n_countries"]} countries'),
    (k2, "Net Profit", money(H["total_net_profit"]), f'{H["net_margin"]:.1%} net margin'),
    (k3, "Top Market", H["top_country"], "Highest total revenue"),
    (k4, "Best Marketing ROI", H["best_efficiency_country"], "Most revenue per marketing $"),
]:
    col.markdown(f'<div class="kpi"><div class="kpi-l">{label}</div>'
                 f'<div class="kpi-v">{val}</div><div class="kpi-n">{note}</div></div>',
                 unsafe_allow_html=True)

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ===========================================================================
# Sidebar
# ===========================================================================
SAMPLES = [
    "Which country has the best marketing efficiency?",
    "Which category has the highest net profit margin?",
    "Which category has the highest return rate?",
    "How does revenue trend across the months?",
    "Which channel generates the most revenue?",
    "Compare customer segments by average order value.",
    "Summarise country performance.",
]

if "q_box" not in st.session_state:
    st.session_state.q_box = ""


def _set_sample(q):
    st.session_state.q_box = q


with st.sidebar:
    st.markdown('<div class="eyebrow">Sample questions</div>', unsafe_allow_html=True)
    for s in SAMPLES:
        st.button(s, key=f"s_{s}", on_click=_set_sample, args=(s,), use_container_width=True)
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="kpi-n">{len(df):,} rows · {H["n_months"]} months<br>'
        f'{H["n_countries"]} countries · {H["n_channels"]} channels<br>'
        f'{H["n_segments"]} segments · {H["n_categories"]} categories</div>',
        unsafe_allow_html=True)
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Query engine — text-to-SQL</div>', unsafe_allow_html=True)
    st.markdown('<div class="kpi-n">The AI writes DuckDB SQL, which is safety-checked, then '
                'executed. Ask about any dimension or metric below.</div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow" style="margin-top:0.8rem">Dimensions</div>', unsafe_allow_html=True)
    st.markdown('<div class="kpi-n">Country · Channel · Segment · Category · Subcategory · Month</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="eyebrow" style="margin-top:0.8rem">Metrics</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="kpi-n">Revenue · Orders · AOV · Net Profit · Margin · Return Rate · '
                'Conversion · Marketing Efficiency · Satisfaction</div>', unsafe_allow_html=True)

# ===========================================================================
# AI assistant — the signature: evidence (Python) above interpretation (LLM)
# ===========================================================================
st.markdown('<div class="sect">Ask the analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sect-note">Pick a sample on the left or type your own. The AI writes SQL, '
            'it is safety-checked, executed in DuckDB, and the result is explained — so every answer '
            'shows the exact query and data it stands on.</div>',
            unsafe_allow_html=True)

question = st.text_area("Business question", key="q_box", height=90,
                        label_visibility="collapsed",
                        placeholder="e.g. Which country has the best marketing efficiency?")
go = st.button("Run analysis", type="primary")

if go and question.strip():
    st.markdown(f'**Question** · {question}')
    if not AI_READY:
        st.info("Add an OpenAI API key (Streamlit Secrets or .env) to ask questions. "
                "The dashboard below works without a key.")
    else:
        with st.spinner("Writing SQL, running it, interpreting…"):
            result = run_question(question, con, SCHEMA, client)

        if result.blocked:
            st.error(f"Query blocked by safety guardrail — {result.error}")
        elif result.error:
            st.error(f"Could not answer — {result.error}")
        else:
            # 1. the SQL the AI wrote (transparent + verifiable)
            cap = "SQL written by the AI" + (" · self-corrected" if result.repaired else "")
            st.markdown(f'<div class="evidence-cap">{cap}</div>', unsafe_allow_html=True)
            st.code(result.sql, language="sql")

            # 2. the real query result from DuckDB
            st.markdown('<div class="evidence-cap">Query result — executed in DuckDB</div>',
                        unsafe_allow_html=True)
            st.dataframe(result.data, use_container_width=True, hide_index=True)

            # 3. the AI's business interpretation of that result
            st.markdown('<div class="evidence-cap">AI Interpretation</div>', unsafe_allow_html=True)
            try:
                with st.spinner("Interpreting the result…"):
                    text = interpret_result(question, result.sql, result.data, client)
                with st.container(border=True):
                    st.markdown(text)
            except Exception as e:
                st.error(f"Interpretation failed: {e}. The SQL and result above are still valid.")
elif go:
    st.warning("Type a business question first.")

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ===========================================================================
# Charts
# ===========================================================================
country, category, monthly = AGG["country"], AGG["category"], AGG["monthly"]

c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="sect">Revenue by country</div>', unsafe_allow_html=True)
    ch = (alt.Chart(country).mark_bar(color=NAVY)
          .encode(x=alt.X("Revenue:Q", title=None, axis=alt.Axis(format="$~s")),
                  y=alt.Y("Country:N", sort="-x", title=None),
                  tooltip=["Country", alt.Tooltip("Revenue:Q", format="$,.0f"),
                           alt.Tooltip("NetProfit:Q", format="$,.0f")])
          .properties(height=220))
    st.altair_chart(chart_base(ch), use_container_width=True)
with c2:
    st.markdown('<div class="sect">Marketing efficiency by country</div>', unsafe_allow_html=True)
    ch = (alt.Chart(country).mark_bar(color=NAVY)
          .encode(x=alt.X("RevenuePerMarketingDollar:Q", title="Revenue per marketing $"),
                  y=alt.Y("Country:N", sort="-x", title=None),
                  tooltip=["Country", alt.Tooltip("RevenuePerMarketingDollar:Q", format=".2f")])
          .properties(height=220))
    st.altair_chart(chart_base(ch), use_container_width=True)

st.markdown('<div class="sect">Revenue trend — the Q4 holiday peak repeats each year</div>',
            unsafe_allow_html=True)
line = (alt.Chart(monthly).mark_area(
            line={"color": NAVY, "strokeWidth": 2},
            color=alt.Gradient(gradient="linear",
                               stops=[alt.GradientStop(color="#ffffff", offset=0),
                                      alt.GradientStop(color="#dbe4ee", offset=1)],
                               x1=1, x2=1, y1=1, y2=0))
        .encode(x=alt.X("Month:N", title=None, axis=alt.Axis(values=list(monthly["Month"][::3]))),
                y=alt.Y("Revenue:Q", title=None, axis=alt.Axis(format="$~s")),
                tooltip=["Month", alt.Tooltip("Revenue:Q", format="$,.0f")])
        .properties(height=240))
st.altair_chart(chart_base(line), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.markdown('<div class="sect">Revenue by category</div>', unsafe_allow_html=True)
    ch = (alt.Chart(category).mark_bar(color=NAVY)
          .encode(x=alt.X("Revenue:Q", title=None, axis=alt.Axis(format="$~s")),
                  y=alt.Y("ProductCategory:N", sort="-x", title=None),
                  tooltip=["ProductCategory", alt.Tooltip("Revenue:Q", format="$,.0f")])
          .properties(height=220))
    st.altair_chart(chart_base(ch), use_container_width=True)
with c4:
    st.markdown('<div class="sect">Net margin by category</div>', unsafe_allow_html=True)
    ch = (alt.Chart(category).mark_bar()
          .encode(x=alt.X("NetMarginPct:Q", title="Net margin", axis=alt.Axis(format=".0%")),
                  y=alt.Y("ProductCategory:N", sort="-x", title=None),
                  color=alt.condition(alt.datum.NetMarginPct > 0.05, alt.value(NAVY), alt.value("#9bb0c4")),
                  tooltip=["ProductCategory", alt.Tooltip("NetMarginPct:Q", format=".1%"),
                           alt.Tooltip("ReturnRate:Q", format=".1%")])
          .properties(height=220))
    st.altair_chart(chart_base(ch), use_container_width=True)

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ===========================================================================
# Detail tabs
# ===========================================================================
t1, t2, t3 = st.tabs(["Performance tables", "Data explorer", "How it works"])

FMT = {"Revenue": "${:,.0f}", "Orders": "{:,.0f}", "MarketingSpend": "${:,.0f}",
       "NetProfit": "${:,.0f}", "GrossMarginPct": "{:.1%}", "NetMarginPct": "{:.1%}",
       "ReturnRate": "{:.2%}", "ConversionRate": "{:.2%}", "AverageOrderValue": "${:.2f}",
       "RevenuePerMarketingDollar": "${:.2f}", "CustomerSatisfaction": "{:.2f}"}


def show(table, idx):
    cols = [idx] + [c for c in ["Revenue", "NetProfit", "NetMarginPct", "AverageOrderValue",
                                "ReturnRate", "RevenuePerMarketingDollar", "CustomerSatisfaction"]
                    if c in table.columns]
    st.dataframe(table[cols].style.format({k: v for k, v in FMT.items() if k in cols}),
                 use_container_width=True, hide_index=True)


with t1:
    sigs = st.columns(3)
    for col, t, v in [
        (sigs[0], "Growth", f'{H["top_category"]} is the top revenue category.'),
        (sigs[1], "Efficiency", f'{H["best_efficiency_country"]} earns the most per marketing $.'),
        (sigs[2], "Risk", f'{H["lowest_sat_country"]} has the lowest satisfaction.'),
    ]:
        col.markdown(f'<div class="signal"><div class="signal-t">{t} signal</div>'
                     f'<div class="signal-v">{v}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sect">By country</div>', unsafe_allow_html=True)
    show(country, "Country")
    st.markdown('<div class="sect">By category</div>', unsafe_allow_html=True)
    show(category, "ProductCategory")
    st.markdown('<div class="sect">By channel</div>', unsafe_allow_html=True)
    show(AGG["channel"], "Channel")
    st.markdown('<div class="sect">By customer segment</div>', unsafe_allow_html=True)
    show(AGG["segment"], "CustomerSegment")

with t2:
    st.markdown('<div class="sect-note">Filter the raw rows. 6,912 rows across the full grain.</div>',
                unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    sel_country = fc1.multiselect("Country", sorted(df["Country"].unique()))
    sel_channel = fc2.multiselect("Channel", sorted(df["Channel"].unique()))
    sel_cat = fc3.multiselect("Category", sorted(df["ProductCategory"].unique()))
    view = df.drop(columns=["MonthPeriod"])
    if sel_country:
        view = view[view["Country"].isin(sel_country)]
    if sel_channel:
        view = view[view["Channel"].isin(sel_channel)]
    if sel_cat:
        view = view[view["ProductCategory"].isin(sel_cat)]
    st.markdown(f'<div class="kpi-n">Showing {len(view):,} rows</div>', unsafe_allow_html=True)
    st.dataframe(view.head(500), use_container_width=True, hide_index=True)

with t3:
    st.markdown('<div class="sect">The two-layer design</div>', unsafe_allow_html=True)
    steps = st.columns(3)
    for col, n, t, d in [
        (steps[0], "01", "Detect intent", "The KPI engine reads the question and picks one dimension "
         "(country, channel, segment, category, subcategory, month) and one metric."),
        (steps[1], "02", "Compute evidence", "Pandas calculates that exact KPI from 6,912 rows. "
         "These numbers are the evidence — produced before the model is called."),
        (steps[2], "03", "Interpret", "The model receives the computed numbers and writes the answer, "
         "evidence, interpretation, and next step. It explains; it does not calculate."),
    ]:
        col.markdown(f'<div class="signal"><div class="signal-t">{n} · {t}</div>'
                     f'<div class="signal-v" style="font-size:0.9rem">{d}</div></div>',
                     unsafe_allow_html=True)
    st.markdown('<div class="sect-note" style="margin-top:1rem">Data + KPI logic lives in '
                '<code>analytics.py</code> (no Streamlit, unit-testable). This file is only UI + '
                'the model call. Synthetic data is generated by <code>generate_data.py</code> with '
                'documented business patterns — not random noise.</div>', unsafe_allow_html=True)