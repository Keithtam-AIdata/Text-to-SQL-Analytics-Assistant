"""
analytics.py
============
Pure data + KPI logic for the AI Business Analyst Assistant.

This module deliberately has NO Streamlit and NO OpenAI imports. It only knows
how to read the dataset, run KPI calculations, and assemble text context. That
separation means:
  * the business logic can be unit-tested on its own (see test_analytics.py),
  * the Streamlit app stays thin and only handles UI + the API call,
  * swapping the front-end or the LLM later touches nothing in here.

Core principle of the whole project:
    Python computes the KPI evidence. The LLM only explains it.
Everything an answer is grounded in is produced here as plain numbers.
"""

from __future__ import annotations
import pandas as pd

DATA_PATH = "data/business_data.csv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Month is stored as "2024-01"; make a real period for correct time ordering.
    df["MonthPeriod"] = pd.PeriodIndex(df["Month"], freq="M")
    return df


# ---------------------------------------------------------------------------
# Intent detection — which DIMENSION and which METRIC is the question about?
# Ordered most-specific-first so "subcategory" wins over "category", etc.
# This two-axis design (dimension × metric) is what fixes the old engine's
# dead-code bug: marketing efficiency is detected as a METRIC, so it can never
# be swallowed by an earlier "revenue" branch the way the nested ifs did.
# ---------------------------------------------------------------------------
DIMENSION_ALIASES: list[tuple[str, list[str]]] = [
    ("ProductSubcategory", ["subcategory", "sub-category", "subcategories", "sub category"]),
    ("CustomerSegment",    ["segment", "new customer", "returning customer", "vip", "loyalty", "customer type"]),
    ("Channel",            ["channel", "online store", "mobile app", "marketplace", "retail", "in-store", "in store"]),
    ("CampaignType",       ["campaign", "promotion type", "promo type"]),
    ("Month",              ["month", "monthly", "trend", "over time", "growth", "season", "seasonal", "time series", "timeline"]),
    ("Quarter",            ["quarter", "quarterly", " q1", " q2", " q3", " q4"]),
    ("ProductCategory",    ["category", "categories", "product line", "product category", "product type"]),
    ("Country",            ["country", "countries", "region", "regional", "market", "geograph",
                            "hong kong", "singapore", "japan", "australia"]),
]

METRIC_RULES: list[tuple[str, list[str]]] = [
    ("marketing_efficiency", ["marketing efficiency", "marketing roi", "roas", "per marketing dollar",
                              "return on ad", "return on marketing"]),
    ("margin",       ["profit margin", "net margin", "gross margin", "margin", "profitability rate"]),
    ("net_profit",   ["net profit", "profitab", "bottom line", "most profit", "highest profit",
                      "lowest profit", "profit"]),
    ("discount",     ["discount", "markdown", "promotion depth"]),
    ("aov",          ["average order value", "aov", "basket size", "order value"]),
    ("return_rate",  ["return rate", "returns", "return"]),
    ("conversion",   ["conversion", "convert"]),
    ("satisfaction", ["satisfaction", "csat", "customer experience", " cx"]),
    ("marketing_spend", ["marketing spend", "ad spend", "marketing budget", "marketing"]),
    ("shipping",     ["shipping", "fulfilment", "fulfillment", "logistics cost"]),
    ("customers",    ["customers", "customer count", "buyers"]),
    ("orders",       ["orders", "order volume", "transactions"]),
    ("revenue",      ["revenue", "sales", "income", "turnover", "top line"]),
]

# How each metric is aggregated.
SUM_COLS = {
    "revenue": "Revenue", "orders": "Orders", "customers": "Customers",
    "marketing_spend": "MarketingSpend", "shipping": "ShippingCost", "net_profit": "NetProfit",
}
MEAN_COLS = {
    "aov": "AverageOrderValue", "return_rate": "ReturnRate", "conversion": "ConversionRate",
    "satisfaction": "CustomerSatisfaction", "margin": "GrossMarginPct", "discount": "DiscountRate",
}
METRIC_LABEL = {
    "revenue": "Total Revenue", "orders": "Total Orders", "customers": "Total Customers",
    "marketing_spend": "Total Marketing Spend", "shipping": "Total Shipping Cost",
    "net_profit": "Total Net Profit", "aov": "Average Order Value", "return_rate": "Average Return Rate",
    "conversion": "Average Conversion Rate", "satisfaction": "Average Customer Satisfaction",
    "margin": "Average Gross Margin %", "discount": "Average Discount Rate",
    "marketing_efficiency": "Marketing Efficiency (Revenue per Marketing $)",
}


def detect_dimension(question: str) -> str:
    ql = f" {question.lower()} "
    for dim, kws in DIMENSION_ALIASES:
        if any(k in ql for k in kws):
            return dim
    return "Country"


def detect_metric(question: str) -> str:
    ql = question.lower()
    for metric, kws in METRIC_RULES:
        if any(k in ql for k in kws):
            return metric
    return "revenue"


def is_summary_request(question: str) -> bool:
    ql = question.lower()
    return any(k in ql for k in
               ["summar", "overview", "overall", "breakdown", "how is", "how are",
                "tell me about", "performance across", "compare"])


# ---------------------------------------------------------------------------
# The KPI engine: turn a question into a block of computed evidence.
# ---------------------------------------------------------------------------
def _fmt_series(s: pd.Series, kind: str) -> str:
    if kind == "money":
        return s.map(lambda v: f"${v:,.0f}").to_string()
    if kind == "ratio2":
        return s.map(lambda v: f"{v:,.2f}").to_string()
    if kind == "pct":
        return s.map(lambda v: f"{v:.2%}").to_string()
    return s.round(2).to_string()


def get_dynamic_context(question: str, df: pd.DataFrame) -> str:
    dim = detect_dimension(question)
    metric = detect_metric(question)

    # --- summary path: a question like "summarize country performance" with no
    # specific metric word -> return a small multi-metric table for that dim.
    explicit_metric = any(any(k in question.lower() for k in kws) for _, kws in METRIC_RULES)
    if is_summary_request(question) and not explicit_metric and dim not in ("Month", "Quarter"):
        g = (df.groupby(dim)
               .agg(Revenue=("Revenue", "sum"),
                    NetProfit=("NetProfit", "sum"),
                    GrossMarginPct=("GrossMarginPct", "mean"),
                    ReturnRate=("ReturnRate", "mean"),
                    CustomerSatisfaction=("CustomerSatisfaction", "mean"))
               .sort_values("Revenue", ascending=False))
        out = g.copy()
        out["Revenue"] = out["Revenue"].map(lambda v: f"${v:,.0f}")
        out["NetProfit"] = out["NetProfit"].map(lambda v: f"${v:,.0f}")
        out["GrossMarginPct"] = out["GrossMarginPct"].map(lambda v: f"{v:.1%}")
        out["ReturnRate"] = out["ReturnRate"].map(lambda v: f"{v:.2%}")
        out["CustomerSatisfaction"] = out["CustomerSatisfaction"].round(2)
        return f"Performance summary by {dim}:\n{out.to_string()}"

    # --- time path: monthly/quarterly trend with growth ---
    if dim in ("Month", "Quarter"):
        time_col = "MonthPeriod" if dim == "Month" else "Quarter"
        col = SUM_COLS.get(metric, "Revenue")
        if metric in MEAN_COLS:
            g = df.groupby(time_col)[MEAN_COLS[metric]].mean()
            body = g.round(4).to_string()
        else:
            g = df.groupby(time_col)[col].sum()
            body = g.map(lambda v: f"${v:,.0f}").to_string() if metric in ("revenue", "net_profit", "marketing_spend", "shipping") else g.to_string()
        peak = g.idxmax()
        mom = g.pct_change().dropna().mean()
        return (f"{METRIC_LABEL.get(metric, metric)} by {dim} (time-ordered):\n{body}\n\n"
                f"Peak period: {peak}\nAverage period-over-period change: {mom:+.1%}")

    # --- marketing efficiency: a ratio, computed explicitly ---
    if metric == "marketing_efficiency":
        g = (df.groupby(dim)
               .apply(lambda d: d["Revenue"].sum() / d["MarketingSpend"].sum(), include_groups=False)
               .sort_values(ascending=False))
        return f"Marketing Efficiency by {dim} (revenue per marketing $):\n{_fmt_series(g, 'ratio2')}"

    # --- standard single-metric paths ---
    if metric in SUM_COLS:
        g = df.groupby(dim)[SUM_COLS[metric]].sum().sort_values(ascending=False)
        kind = "money" if metric in ("revenue", "net_profit", "marketing_spend", "shipping") else "plain"
        return f"{METRIC_LABEL[metric]} by {dim}:\n{_fmt_series(g, kind)}"

    if metric in MEAN_COLS:
        g = df.groupby(dim)[MEAN_COLS[metric]].mean().sort_values(ascending=False)
        kind = {"return_rate": "pct", "conversion": "pct", "discount": "pct",
                "margin": "pct", "aov": "money"}.get(metric, "plain")
        if metric == "margin":
            kind = "pct"
        return f"{METRIC_LABEL[metric]} by {dim}:\n{_fmt_series(g, kind)}"

    # fallback
    g = df.groupby(dim)["Revenue"].sum().sort_values(ascending=False)
    return f"Total Revenue by {dim}:\n{_fmt_series(g, 'money')}"


# ---------------------------------------------------------------------------
# Overall business context (always sent to the model alongside dynamic context)
# ---------------------------------------------------------------------------
def build_business_context(df: pd.DataFrame) -> str:
    total_rev = df["Revenue"].sum()
    total_profit = df["NetProfit"].sum()
    months = f'{df["Month"].min()} to {df["Month"].max()}'
    rev_country = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
    rev_cat = df.groupby("ProductCategory")["Revenue"].sum().sort_values(ascending=False)
    margin_cat = df.groupby("ProductCategory")["GrossMarginPct"].mean().sort_values(ascending=False)
    return (
        "Dataset: Simulated APAC e-commerce performance, monthly grain.\n"
        f"Period: {months} | Rows: {len(df):,}\n"
        f"Total Revenue: ${total_rev:,.0f} | Total Net Profit: ${total_profit:,.0f}\n\n"
        f"Revenue by Country:\n{rev_country.map(lambda v: f'${v:,.0f}').to_string()}\n\n"
        f"Revenue by Category:\n{rev_cat.map(lambda v: f'${v:,.0f}').to_string()}\n\n"
        f"Gross Margin % by Category:\n{margin_cat.map(lambda v: f'{v:.1%}').to_string()}\n\n"
        f"Available columns: {', '.join(df.columns)}"
    )


def build_prompt(question: str, df: pd.DataFrame) -> tuple[str, str]:
    """Return (evidence_block, full_prompt). The evidence block is shown to the
    user verbatim so they can see exactly what the AI's answer is grounded in."""
    business_context = build_business_context(df)
    evidence = get_dynamic_context(question, df)
    prompt = f"""You are a business analytics assistant for an APAC e-commerce dataset.
Help business users interpret KPI performance in clear language.

Rules:
1. Use ONLY the Business Context and KPI Evidence below. Do not invent numbers.
2. If the evidence is insufficient, say what additional KPI is needed.
3. Be concise, professional, and decision-oriented.
4. Do not ask casual follow-up questions.

Respond in exactly this format:
Direct Answer:
[1-2 sentences.]

KPI Evidence:
[The key numbers that support the answer.]

Business Interpretation:
[What it means for the business.]

Recommended Next Step:
[One practical action.]

=== Business Context ===
{business_context}

=== KPI Evidence (computed by Python) ===
{evidence}

=== User Question ===
{question}
"""
    return evidence, prompt


# ---------------------------------------------------------------------------
# Dashboard aggregations
# ---------------------------------------------------------------------------
def build_aggregations(df: pd.DataFrame) -> dict:
    def by(dim):
        g = (df.groupby(dim)
               .agg(Revenue=("Revenue", "sum"),
                    Orders=("Orders", "sum"),
                    MarketingSpend=("MarketingSpend", "sum"),
                    NetProfit=("NetProfit", "sum"),
                    GrossMarginPct=("GrossMarginPct", "mean"),
                    ReturnRate=("ReturnRate", "mean"),
                    ConversionRate=("ConversionRate", "mean"),
                    AverageOrderValue=("AverageOrderValue", "mean"),
                    CustomerSatisfaction=("CustomerSatisfaction", "mean"))
               .reset_index())
        g["RevenuePerMarketingDollar"] = (g["Revenue"] / g["MarketingSpend"]).round(2)
        g["NetMarginPct"] = (g["NetProfit"] / g["Revenue"])
        return g

    country = by("Country")
    category = by("ProductCategory")
    channel = by("Channel")
    segment = by("CustomerSegment")

    monthly = (df.groupby("MonthPeriod")
                 .agg(Revenue=("Revenue", "sum"), NetProfit=("NetProfit", "sum"))
                 .reset_index())
    monthly["Month"] = monthly["MonthPeriod"].astype(str)

    headline = {
        "total_revenue": df["Revenue"].sum(),
        "total_orders": df["Orders"].sum(),
        "total_net_profit": df["NetProfit"].sum(),
        "net_margin": df["NetProfit"].sum() / df["Revenue"].sum(),
        "top_country": country.sort_values("Revenue", ascending=False).iloc[0]["Country"],
        "best_efficiency_country": country.sort_values("RevenuePerMarketingDollar", ascending=False).iloc[0]["Country"],
        "lowest_sat_country": country.sort_values("CustomerSatisfaction").iloc[0]["Country"],
        "top_category": category.sort_values("Revenue", ascending=False).iloc[0]["ProductCategory"],
        "most_profitable_category": category.sort_values("NetProfit", ascending=False).iloc[0]["ProductCategory"],
        "highest_return_category": category.sort_values("ReturnRate", ascending=False).iloc[0]["ProductCategory"],
        "n_countries": df["Country"].nunique(),
        "n_channels": df["Channel"].nunique(),
        "n_segments": df["CustomerSegment"].nunique(),
        "n_categories": df["ProductCategory"].nunique(),
        "n_months": df["Month"].nunique(),
    }
    return {"country": country, "category": category, "channel": channel,
            "segment": segment, "monthly": monthly, "headline": headline}
