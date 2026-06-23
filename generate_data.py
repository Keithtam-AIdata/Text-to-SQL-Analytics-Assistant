import pandas as pd
import random

random.seed(42)

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

regions = ["Hong Kong", "Singapore", "Japan", "Australia"]

categories = [
    "Electronics",
    "Fashion",
    "Home & Living",
    "Beauty",
    "Sports",
    "Grocery"
]

rows = []

for month in months:
    for region in regions:
        for category in categories:
            base_revenue = random.randint(50000, 180000)
            orders = random.randint(400, 1800)
            customers = random.randint(300, orders)
            marketing_spend = random.randint(5000, 30000)

            conversion_rate = round(random.uniform(0.025, 0.085), 4)
            return_rate = round(random.uniform(0.02, 0.12), 4)
            customer_satisfaction = round(random.uniform(3.4, 4.8), 2)

            rows.append({
                "Month": month,
                "Region": region,
                "ProductCategory": category,
                "Revenue": base_revenue,
                "Orders": orders,
                "Customers": customers,
                "AverageOrderValue": round(base_revenue / orders, 2),
                "ConversionRate": conversion_rate,
                "MarketingSpend": marketing_spend,
                "ReturnRate": return_rate,
                "CustomerSatisfaction": customer_satisfaction
            })

df = pd.DataFrame(rows)

df.to_csv("data/business_data.csv", index=False)

print("Dataset generated successfully.")
print(f"Rows generated: {len(df)}")
print(df.head())