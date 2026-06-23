import os
import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

try:
    if not api_key and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

client = OpenAI(api_key=api_key)

st.set_page_config(
    page_title="AI Business Analyst Assistant",
    page_icon="📊",
    layout="wide"
)

df = pd.read_csv("data/business_data.csv")


def get_dynamic_context(question, df):
    question_lower = question.lower()

    # Revenue analysis
    if "revenue" in question_lower:
        if "category" in question_lower or "product" in question_lower:
            summary = (
                df.groupby("ProductCategory")["Revenue"]
                .sum()
                .sort_values(ascending=False)
            )

            return f"""
Revenue by Product Category:
{summary.to_string()}
"""

        elif "region" in question_lower or "market" in question_lower:
            summary = (
                df.groupby("Region")["Revenue"]
                .sum()
                .sort_values(ascending=False)
            )

            return f"""
Revenue by Region:
{summary.to_string()}
"""

    # Marketing efficiency / ROI analysis
    if (
        "marketing efficiency" in question_lower
        or "marketing roi" in question_lower
        or "roas" in question_lower
        or "revenue per marketing" in question_lower
    ):
        summary = (
            df.groupby("Region")
            .agg(
                TotalRevenue=("Revenue", "sum"),
                TotalMarketingSpend=("MarketingSpend", "sum")
            )
        )

        summary["RevenuePerMarketingDollar"] = (
            summary["TotalRevenue"] / summary["TotalMarketingSpend"]
        ).round(2)

        summary = summary.sort_values(
            "RevenuePerMarketingDollar",
            ascending=False
        )

        return f"""
Marketing Efficiency by Region:
{summary.to_string()}
"""

    # Marketing spend analysis
    if "marketing" in question_lower or "marketing spend" in question_lower:
        if "category" in question_lower or "product" in question_lower:
            summary = (
                df.groupby("ProductCategory")["MarketingSpend"]
                .sum()
                .sort_values(ascending=False)
            )

            return f"""
Total Marketing Spend by Product Category:
{summary.to_string()}
"""

        else:
            summary = (
                df.groupby("Region")["MarketingSpend"]
                .sum()
                .sort_values(ascending=False)
            )

            return f"""
Total Marketing Spend by Region:
{summary.to_string()}
"""

    # Orders analysis
    if "orders" in question_lower or "order volume" in question_lower:
        if "category" in question_lower or "product" in question_lower:
            summary = (
                df.groupby("ProductCategory")["Orders"]
                .sum()
                .sort_values(ascending=False)
            )

            return f"""
Total Orders by Product Category:
{summary.to_string()}
"""

        else:
            summary = (
                df.groupby("Region")["Orders"]
                .sum()
                .sort_values(ascending=False)
            )

            return f"""
Total Orders by Region:
{summary.to_string()}
"""

    # Average Order Value analysis
    if (
        "aov" in question_lower
        or "average order value" in question_lower
    ):
        if "region" in question_lower or "market" in question_lower:
            summary = (
                df.groupby("Region")["AverageOrderValue"]
                .mean()
                .sort_values(ascending=False)
                .round(2)
            )

            return f"""
Average Order Value by Region:
{summary.to_string()}
"""

        else:
            summary = (
                df.groupby("ProductCategory")["AverageOrderValue"]
                .mean()
                .sort_values(ascending=False)
                .round(2)
            )

            return f"""
Average Order Value by Product Category:
{summary.to_string()}
"""

    # Customer satisfaction analysis
    if (
        "satisfaction" in question_lower
        or "customer satisfaction" in question_lower
        or "cx" in question_lower
    ):
        if "category" in question_lower or "product" in question_lower:
            summary = (
                df.groupby("ProductCategory")["CustomerSatisfaction"]
                .mean()
                .sort_values(ascending=False)
                .round(2)
            )

            return f"""
Average Customer Satisfaction by Product Category:
{summary.to_string()}
"""

        else:
            summary = (
                df.groupby("Region")["CustomerSatisfaction"]
                .mean()
                .sort_values(ascending=False)
                .round(2)
            )

            return f"""
Average Customer Satisfaction by Region:
{summary.to_string()}
"""

    # Return rate analysis
    if "return rate" in question_lower or "returns" in question_lower:
        if "category" in question_lower or "product" in question_lower:
            summary = (
                df.groupby("ProductCategory")["ReturnRate"]
                .mean()
                .sort_values(ascending=False)
                .round(4)
            )

            return f"""
Average Return Rate by Product Category:
{summary.to_string()}
"""

        else:
            summary = (
                df.groupby("Region")["ReturnRate"]
                .mean()
                .sort_values(ascending=False)
                .round(4)
            )

            return f"""
Average Return Rate by Region:
{summary.to_string()}
"""

    # Conversion rate analysis
    if "conversion" in question_lower or "conversion rate" in question_lower:
        if "category" in question_lower or "product" in question_lower:
            summary = (
                df.groupby("ProductCategory")["ConversionRate"]
                .mean()
                .sort_values(ascending=False)
                .round(4)
            )

            return f"""
Average Conversion Rate by Product Category:
{summary.to_string()}
"""

        else:
            summary = (
                df.groupby("Region")["ConversionRate"]
                .mean()
                .sort_values(ascending=False)
                .round(4)
            )

            return f"""
Average Conversion Rate by Region:
{summary.to_string()}
"""

    return "No specific dynamic KPI was triggered. Use the general business context."


def build_business_context(df):
    total_revenue = df["Revenue"].sum()

    region_summary = (
        df.groupby("Region")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    category_summary = (
        df.groupby("ProductCategory")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    satisfaction_summary = (
        df.groupby("Region")["CustomerSatisfaction"]
        .mean()
        .round(2)
    )

    return f"""
Dataset: Retail / E-commerce business performance data

Total Revenue: ${total_revenue:,.0f}

Revenue by Region:
{region_summary.to_string()}

Revenue by Product Category:
{category_summary.to_string()}

Average Customer Satisfaction by Region:
{satisfaction_summary.to_string()}

Available columns:
{", ".join(df.columns)}
"""


def generate_ai_answer(question, df):
    business_context = build_business_context(df)
    dynamic_context = get_dynamic_context(question, df)

    prompt = f"""
You are a ChatGPT-style AI Business Analyst Assistant for an e-commerce analytics dataset.

Your role is to help business users understand KPI performance using natural language.

Rules:
1. Answer only using the Business Context and Dynamic KPI Results provided.
2. Do not invent numbers, columns, or explanations that are not supported by the provided data.
3. If the provided KPI results are not enough to answer the question, clearly say that more KPI context is needed.
4. Keep the answer concise, professional, and business-focused.
5. Do not ask casual follow-up questions at the end.

Use this response format:

Direct Answer:
[Give the answer in 1-2 sentences.]

KPI Evidence:
[Show the key numbers that support the answer.]

Business Interpretation:
[Explain what the result means from a business perspective.]

Recommended Next Step:
[Suggest one practical action.]

Business Context:
{business_context}

Dynamic KPI Results:
{dynamic_context}

User Question:
{question}
"""

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return response.output_text


# -----------------------------
# Streamlit UI
# -----------------------------

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 45%, #f8fafc 100%);
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .hero-container {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.22);
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
        letter-spacing: -0.03em;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        max-width: 900px;
        line-height: 1.55;
    }

    .badge {
        display: inline-block;
        padding: 0.38rem 0.75rem;
        margin-right: 0.45rem;
        margin-top: 0.9rem;
        border-radius: 999px;
        background-color: rgba(255,255,255,0.12);
        color: #e2e8f0;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .kpi-card {
        background: rgba(255, 255, 255, 0.92);
        padding: 1.1rem 1.2rem;
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
        min-height: 125px;
    }

    .kpi-label {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.45rem;
    }

    .kpi-value {
        color: #0f172a;
        font-size: 1.7rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }

    .kpi-note {
        color: #64748b;
        font-size: 0.88rem;
        line-height: 1.4;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.4rem;
        margin-bottom: 0.65rem;
    }

    .insight-card {
        background: rgba(255, 255, 255, 0.92);
        padding: 1.1rem 1.2rem;
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.07);
        height: 100%;
    }

    .insight-title {
        color: #0f172a;
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .insight-text {
        color: #475569;
        font-size: 0.93rem;
        line-height: 1.5;
    }

    .small-muted {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
    }

    div[data-testid="stButton"] > button {
        border-radius: 12px;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def kpi_card(label, value, note):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# Prepare dashboard data
# -----------------------------

total_revenue = df["Revenue"].sum()
total_orders = df["Orders"].sum()
avg_satisfaction = df["CustomerSatisfaction"].mean()

region_perf = (
    df.groupby("Region")
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("Orders", "sum"),
        MarketingSpend=("MarketingSpend", "sum"),
        CustomerSatisfaction=("CustomerSatisfaction", "mean"),
        ReturnRate=("ReturnRate", "mean"),
        ConversionRate=("ConversionRate", "mean"),
    )
    .reset_index()
)

region_perf["RevenuePerMarketingDollar"] = (
    region_perf["Revenue"] / region_perf["MarketingSpend"]
).round(2)

region_perf["CustomerSatisfaction"] = region_perf["CustomerSatisfaction"].round(2)
region_perf["ReturnRate"] = region_perf["ReturnRate"].round(4)
region_perf["ConversionRate"] = region_perf["ConversionRate"].round(4)

category_perf = (
    df.groupby("ProductCategory")
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("Orders", "sum"),
        AverageOrderValue=("AverageOrderValue", "mean"),
        ReturnRate=("ReturnRate", "mean"),
        CustomerSatisfaction=("CustomerSatisfaction", "mean"),
    )
    .reset_index()
)

category_perf["AverageOrderValue"] = category_perf["AverageOrderValue"].round(2)
category_perf["ReturnRate"] = category_perf["ReturnRate"].round(4)
category_perf["CustomerSatisfaction"] = category_perf["CustomerSatisfaction"].round(2)

top_region = (
    region_perf.sort_values("Revenue", ascending=False)
    .iloc[0]["Region"]
)

top_category = (
    category_perf.sort_values("Revenue", ascending=False)
    .iloc[0]["ProductCategory"]
)

best_marketing_region = (
    region_perf.sort_values("RevenuePerMarketingDollar", ascending=False)
    .iloc[0]["Region"]
)

lowest_satisfaction_region = (
    region_perf.sort_values("CustomerSatisfaction", ascending=True)
    .iloc[0]["Region"]
)

highest_return_category = (
    category_perf.sort_values("ReturnRate", ascending=False)
    .iloc[0]["ProductCategory"]
)


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.header("💬 Sample Questions")

    sample_questions = [
        "Which region has the best marketing efficiency?",
        "Which product category has the highest return rate?",
        "Which category has the highest AOV?",
        "Which region generated the highest revenue?",
        "Summarize regional performance.",
    ]

    if "question_input" not in st.session_state:
        st.session_state.question_input = ""

    if "analysis_answer" not in st.session_state:
        st.session_state.analysis_answer = ""

    if "last_question" not in st.session_state:
        st.session_state.last_question = ""

    for sample_question in sample_questions:
        if st.button(sample_question, use_container_width=True):
            st.session_state.question_input = sample_question

    st.divider()

    st.header("📁 Dataset")
    st.write(f"Rows: {len(df)}")
    st.write(f"Regions: {df['Region'].nunique()}")
    st.write(f"Categories: {df['ProductCategory'].nunique()}")

    st.divider()

    st.header("🧠 KPI Engine")
    st.write("Revenue")
    st.write("Orders")
    st.write("AOV")
    st.write("Conversion Rate")
    st.write("Return Rate")
    st.write("Marketing Efficiency")


# -----------------------------
# Hero
# -----------------------------

st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">📊 ChatGPT-Style Business Analytics Assistant</div>
        <div class="hero-subtitle">
            A product-style AI analytics assistant that turns natural language business questions
            into KPI-driven insights, evidence, interpretation, and recommended next actions.
        </div>
        <span class="badge">Python</span>
        <span class="badge">Pandas</span>
        <span class="badge">OpenAI API</span>
        <span class="badge">Streamlit</span>
        <span class="badge">Business Analytics</span>
        <span class="badge">Dynamic KPI Engine</span>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Main Overview: AI Assistant + KPI Snapshot
# -----------------------------

st.markdown('<div class="section-title">Executive KPI Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    kpi_card(
        "Total Revenue",
        f"${total_revenue:,.0f}",
        "Total revenue across all regions and product categories."
    )

with col2:
    kpi_card(
        "Total Orders",
        f"{total_orders:,.0f}",
        "Total order volume in the simulated business dataset."
    )

with col3:
    kpi_card(
        "Top Region",
        top_region,
        "Region with the highest total revenue."
    )

with col4:
    kpi_card(
        "Best Efficiency",
        best_marketing_region,
        "Highest revenue generated per marketing dollar."
    )

st.markdown("")

overview_left, overview_right = st.columns([0.92, 1.08])

with overview_left:
    st.markdown('<div class="section-title">Ask the AI Business Analyst</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="insight-card">
            <div class="insight-title">Natural Language KPI Analysis</div>
            <div class="insight-text">
                Ask a business question about revenue, orders, AOV, conversion rate, return rate,
                customer satisfaction, or marketing efficiency. The assistant calculates relevant KPI context first,
                then generates a structured business recommendation.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")

    question = st.text_area(
        "Business question",
        key="question_input",
        height=115,
        placeholder="Example: Which region has the best marketing efficiency?"
    )

    analyze_button = st.button(
        "Analyze KPI Performance",
        type="primary",
        use_container_width=True
    )

with overview_right:
    st.markdown('<div class="section-title">AI Business Analysis</div>', unsafe_allow_html=True)

    if analyze_button and question:
        with st.spinner("Analyzing KPI data and generating business insight..."):
            st.session_state.analysis_answer = generate_ai_answer(question, df)
            st.session_state.last_question = question

    elif analyze_button and not question:
        st.warning("Please enter a business question first.")

    if st.session_state.analysis_answer:
        st.markdown(f"**Question:** {st.session_state.last_question}")

        with st.container(border=True):
            st.markdown(st.session_state.analysis_answer)
    else:
        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-title">Ready to Analyze</div>
                <div class="insight-text">
                    Choose a sample question from the sidebar or type your own question.
                    The response will include a direct answer, KPI evidence, business interpretation,
                    and a recommended next step.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.divider()

# -----------------------------
# Snapshot charts on landing page
# -----------------------------

snapshot_left, snapshot_right = st.columns([1.1, 1])

with snapshot_left:
    st.markdown('<div class="section-title">Revenue by Region</div>', unsafe_allow_html=True)

    revenue_chart = (
        alt.Chart(region_perf)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Region:N", sort="-y", title=None),
            y=alt.Y("Revenue:Q", title="Revenue", axis=alt.Axis(format="$~s")),
            tooltip=[
                "Region",
                alt.Tooltip("Revenue:Q", format="$,.0f"),
                alt.Tooltip("Orders:Q", format=",.0f"),
                alt.Tooltip("CustomerSatisfaction:Q"),
            ],
            color=alt.value("#2563eb"),
        )
        .properties(height=260)
    )

    st.altair_chart(revenue_chart, use_container_width=True)

with snapshot_right:
    st.markdown('<div class="section-title">Marketing Efficiency by Region</div>', unsafe_allow_html=True)

    efficiency_chart = (
        alt.Chart(region_perf)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("RevenuePerMarketingDollar:Q", title="Revenue per marketing dollar"),
            y=alt.Y("Region:N", sort="-x", title=None),
            tooltip=[
                "Region",
                alt.Tooltip("RevenuePerMarketingDollar:Q", format=".2f"),
                alt.Tooltip("MarketingSpend:Q", format="$,.0f"),
                alt.Tooltip("Revenue:Q", format="$,.0f"),
            ],
            color=alt.value("#0f766e"),
        )
        .properties(height=260)
    )

    st.altair_chart(efficiency_chart, use_container_width=True)

st.markdown('<div class="section-title">Product Category Revenue</div>', unsafe_allow_html=True)

category_chart = (
    alt.Chart(category_perf)
    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
    .encode(
        x=alt.X("Revenue:Q", title="Revenue", axis=alt.Axis(format="$~s")),
        y=alt.Y("ProductCategory:N", sort="-x", title=None),
        tooltip=[
            "ProductCategory",
            alt.Tooltip("Revenue:Q", format="$,.0f"),
            alt.Tooltip("Orders:Q", format=",.0f"),
            alt.Tooltip("AverageOrderValue:Q", format="$,.2f"),
            alt.Tooltip("ReturnRate:Q", format=".2%"),
        ],
        color=alt.value("#7c3aed"),
    )
    .properties(height=280)
)

st.altair_chart(category_chart, use_container_width=True)

st.divider()

# -----------------------------
# Detail Tabs
# -----------------------------

dashboard_tab, data_tab, method_tab = st.tabs(
    ["Detailed Dashboard", "Data Explorer", "How It Works"]
)

with dashboard_tab:
    st.markdown('<div class="section-title">Business Signals</div>', unsafe_allow_html=True)

    insight_col1, insight_col2, insight_col3 = st.columns(3)

    with insight_col1:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">Growth Signal</div>
                <div class="insight-text">
                    <b>{top_category}</b> is the top revenue category, making it a key area
                    for growth analysis and category strategy.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with insight_col2:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">Efficiency Signal</div>
                <div class="insight-text">
                    <b>{best_marketing_region}</b> generates the strongest revenue per marketing dollar,
                    suggesting better marketing efficiency.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with insight_col3:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">Risk Signal</div>
                <div class="insight-text">
                    <b>{lowest_satisfaction_region}</b> has the lowest customer satisfaction,
                    making it a potential customer experience risk area.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown('<div class="section-title">Regional Performance Table</div>', unsafe_allow_html=True)

    st.dataframe(
        region_perf.style.format(
            {
                "Revenue": "${:,.0f}",
                "Orders": "{:,.0f}",
                "MarketingSpend": "${:,.0f}",
                "CustomerSatisfaction": "{:.2f}",
                "ReturnRate": "{:.2%}",
                "ConversionRate": "{:.2%}",
                "RevenuePerMarketingDollar": "${:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


with data_tab:
    st.markdown('<div class="section-title">Regional Performance Table</div>', unsafe_allow_html=True)

    st.dataframe(
        region_perf.style.format(
            {
                "Revenue": "${:,.0f}",
                "Orders": "{:,.0f}",
                "MarketingSpend": "${:,.0f}",
                "CustomerSatisfaction": "{:.2f}",
                "ReturnRate": "{:.2%}",
                "ConversionRate": "{:.2%}",
                "RevenuePerMarketingDollar": "${:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<div class="section-title">Product Category Performance Table</div>', unsafe_allow_html=True)

    st.dataframe(
        category_perf.style.format(
            {
                "Revenue": "${:,.0f}",
                "Orders": "{:,.0f}",
                "AverageOrderValue": "${:.2f}",
                "ReturnRate": "{:.2%}",
                "CustomerSatisfaction": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Raw dataset preview"):
        st.dataframe(df.head(30), use_container_width=True)


with method_tab:
    st.markdown('<div class="section-title">How the Assistant Works</div>', unsafe_allow_html=True)

    flow_col1, flow_col2, flow_col3 = st.columns(3)

    with flow_col1:
        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-title">1. User Question</div>
                <div class="insight-text">
                    A business user asks a natural language question about KPI performance.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with flow_col2:
        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-title">2. Dynamic KPI Engine</div>
                <div class="insight-text">
                    Python and Pandas calculate the relevant KPI context based on question intent.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with flow_col3:
        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-title">3. AI Business Response</div>
                <div class="insight-text">
                    OpenAI generates a structured answer with evidence, interpretation, and action.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown(
        """
        **Project Architecture**

        ```text
        Business Question
        ↓
        Dynamic KPI Selection
        ↓
        Pandas Calculation
        ↓
        Prompt Context
        ↓
        OpenAI Response
        ↓
        Business Insight
        ```

        **Key Design Principle:**  
        The model does not calculate the KPIs by guessing. The Python data layer calculates KPI results first, then the AI explains the results in business language.
        """
    )