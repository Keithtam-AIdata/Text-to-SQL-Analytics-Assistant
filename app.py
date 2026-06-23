import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

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

st.title("📊 ChatGPT-Style Business Analytics Assistant")
st.caption(
    "An AI-powered analytics assistant that helps business users ask natural language "
    "questions about e-commerce KPI performance."
)

# Prepare KPI metrics
total_revenue = df["Revenue"].sum()
total_orders = df["Orders"].sum()

top_region = (
    df.groupby("Region")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .index[0]
)

top_category = (
    df.groupby("ProductCategory")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .index[0]
)

marketing_efficiency = (
    df.groupby("Region")
    .agg(
        TotalRevenue=("Revenue", "sum"),
        TotalMarketingSpend=("MarketingSpend", "sum")
    )
)

marketing_efficiency["RevenuePerMarketingDollar"] = (
    marketing_efficiency["TotalRevenue"]
    / marketing_efficiency["TotalMarketingSpend"]
).round(2)

best_marketing_region = (
    marketing_efficiency["RevenuePerMarketingDollar"]
    .sort_values(ascending=False)
    .index[0]
)

# Sidebar
with st.sidebar:
    st.header("💬 Sample Questions")

    sample_questions = [
        "Which region has the best marketing efficiency?",
        "Which product category has the highest return rate?",
        "Which category has the highest AOV?",
        "Which region generated the highest revenue?",
        "Summarize regional performance."
    ]

    if "question_input" not in st.session_state:
        st.session_state.question_input = ""

    for sample_question in sample_questions:
        if st.button(sample_question, use_container_width=True):
            st.session_state.question_input = sample_question

    st.divider()

    st.header("📁 Dataset Overview")
    st.write(f"Rows: {len(df)}")
    st.write(f"Regions: {df['Region'].nunique()}")
    st.write(f"Product Categories: {df['ProductCategory'].nunique()}")

# KPI cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Revenue", f"${total_revenue:,.0f}")

with col2:
    st.metric("Total Orders", f"{total_orders:,.0f}")

with col3:
    st.metric("Top Revenue Region", top_region)

with col4:
    st.metric("Best Marketing Efficiency", best_marketing_region)

st.divider()

# Main question input
st.subheader("Ask a business question")

question = st.text_input(
    "Type your question below:",
    key="question_input",
    placeholder="Example: Which region has the best marketing efficiency?"
)

analyze_button = st.button("Analyze", type="primary")

if analyze_button and question:
    with st.spinner("Analyzing KPI data and generating business insight..."):
        answer = generate_ai_answer(question, df)

    st.subheader("AI Business Analysis")

    with st.container(border=True):
        st.markdown(answer)

elif analyze_button and not question:
    st.warning("Please enter a business question first.")

st.divider()

# Dataset preview
with st.expander("Preview dataset"):
    st.dataframe(df.head(20), use_container_width=True)