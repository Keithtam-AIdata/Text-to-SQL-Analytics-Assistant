import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

df = pd.read_csv("data/business_data.csv")

def get_dynamic_context(question, df):
    question_lower = question.lower()

    if "return rate" in question_lower:
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

        elif "region" in question_lower:
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

    elif "marketing" in question_lower or "marketing spend" in question_lower:
        summary = (
            df.groupby("Region")["MarketingSpend"]
            .sum()
            .sort_values(ascending=False)
        )

        return f"""
Total Marketing Spend by Region:
{summary.to_string()}
"""

    elif "conversion" in question_lower or "conversion rate" in question_lower:
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

total_revenue = df["Revenue"].sum()
top_region = df.groupby("Region")["Revenue"].sum().idxmax()
top_category = df.groupby("ProductCategory")["Revenue"].sum().idxmax()

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

business_context = f"""
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

print("AI Business Analyst Assistant")
print("Ask a business question about the dataset.")
print("Type 'exit' to quit.\n")

while True:
    question = input("Ask a business question: ")

    if question.lower() == "exit":
        print("Goodbye.")
        break

    dynamic_context = get_dynamic_context(question, df)

    prompt = f"""
You are a ChatGPT-style AI Business Analyst Assistant.

You help business users understand performance data using natural language.

Use the business context and dynamic KPI results below to answer the user's question clearly and professionally.

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

    print("\nAI Answer")
    print(response.output_text)
    print("\n" + "-" * 60 + "\n")