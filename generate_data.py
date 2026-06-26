"""
generate_data.py
=================
Pattern-driven synthetic e-commerce dataset for the AI Business Analyst Assistant.

DESIGN PHILOSOPHY
-----------------
Nothing here is purely random. Every metric is built as:

        value = base  ×  business-pattern multipliers  ×  small noise

Each multiplier below maps to ONE documented, explainable business story
(see docs/data_dictionary.md). The small noise layer (~±10%) is added LAST,
only so the data does not look artificially perfect. This means every number
in the output can be traced back to a deliberate assumption you can defend in
an interview.

GRAIN (one row per combination, per month)
------------------------------------------
    Month (24) × Country (4) × Channel (4) × Segment (3) × Category (6)
    = 41,472 rows

Subcategory and Campaign are attributes attached to each row, not part of the
grain (so they don't multiply the row count). They still carry pattern, so
grouping by them is meaningful.

To shrink the dataset for a lighter demo, reduce N_MONTHS or drop a channel.
"""

import numpy as np
import pandas as pd
from itertools import product

RNG = np.random.default_rng(42)  # fixed seed => reproducible build

# ===========================================================================
# 1. DIMENSIONS
# ===========================================================================
MONTHS = pd.period_range("2024-01", "2025-12", freq="M")  # 24 months -> trend & seasonality
REGION = "APAC"
COUNTRIES = ["Hong Kong", "Singapore", "Japan", "Australia"]
CHANNELS = ["Online Store", "Mobile App", "Marketplace", "Retail"]
SEGMENTS = ["New", "Returning", "VIP"]

# Each category owns subcategories with a relative share (weights sum to 1.0).
# The dominant subcategory per row is drawn using these weights, so e.g.
# Smartphones naturally outnumber Audio within Electronics.
CATEGORY_SUBCATS = {
    "Electronics":   {"Smartphones": 0.45, "Laptops": 0.35, "Audio": 0.20},
    "Fashion":       {"Apparel": 0.50, "Footwear": 0.30, "Accessories": 0.20},
    "Home & Living": {"Furniture": 0.40, "Kitchen": 0.35, "Decor": 0.25},
    "Beauty":        {"Skincare": 0.50, "Makeup": 0.30, "Fragrance": 0.20},
    "Sports":        {"Fitness": 0.45, "Outdoor": 0.35, "Apparel-Sport": 0.20},
    "Grocery":       {"Packaged": 0.55, "Fresh": 0.30, "Beverages": 0.15},
}
CATEGORIES = list(CATEGORY_SUBCATS)

# ===========================================================================
# 2. BUSINESS-PATTERN MULTIPLIERS  (each line = one defensible story)
# ===========================================================================

# --- Country -------------------------------------------------------------
# Story: Hong Kong is the largest market; Australia the smallest.
COUNTRY_REVENUE_MULT   = {"Hong Kong": 1.30, "Singapore": 1.00, "Japan": 1.10, "Australia": 0.85}
# Story: Japan spends marketing dollars most efficiently (precision targeting culture).
COUNTRY_MKT_EFFICIENCY = {"Hong Kong": 0.95, "Singapore": 1.00, "Japan": 1.45, "Australia": 0.90}
# Story: HK has high volume but more service strain -> lower baseline satisfaction.
COUNTRY_SAT_BASE       = {"Hong Kong": 3.9, "Singapore": 4.3, "Japan": 4.5, "Australia": 4.2}

# --- Channel -------------------------------------------------------------
CHANNEL_REVENUE_MULT = {"Online Store": 1.10, "Mobile App": 1.00, "Marketplace": 0.90, "Retail": 1.05}
# Story: online/marketplace returns are higher; in-store (Retail) returns are lowest.
CHANNEL_RETURN_MULT  = {"Online Store": 1.05, "Mobile App": 1.10, "Marketplace": 1.20, "Retail": 0.70}
CHANNEL_CONV_BASE    = {"Online Store": 0.055, "Mobile App": 0.045, "Marketplace": 0.035, "Retail": 0.080}

# --- Segment -------------------------------------------------------------
# Story: VIPs spend far more per order and buy more often; New customers spend least.
SEGMENT_REVENUE_MULT      = {"New": 0.70, "Returning": 1.00, "VIP": 1.70}
SEGMENT_AOV_MULT          = {"New": 0.80, "Returning": 1.00, "VIP": 1.75}
SEGMENT_ORDERS_PER_CUST   = {"New": 1.1, "Returning": 1.6, "VIP": 2.6}

# --- Category ------------------------------------------------------------
# Story: Electronics has the highest AOV, Grocery the lowest.
CATEGORY_AOV_MULT    = {"Electronics": 1.80, "Fashion": 0.85, "Home & Living": 1.30,
                        "Beauty": 0.75, "Sports": 1.00, "Grocery": 0.45}
# Story: Electronics is thin-margin; Beauty/Fashion are high-margin.
CATEGORY_MARGIN      = {"Electronics": 0.32, "Fashion": 0.55, "Home & Living": 0.44,
                        "Beauty": 0.62, "Sports": 0.46, "Grocery": 0.34}
# Story: Fashion has by far the highest return rate (size/fit); Grocery the lowest.
CATEGORY_RETURN_BASE = {"Electronics": 0.06, "Fashion": 0.19, "Home & Living": 0.08,
                        "Beauty": 0.05, "Sports": 0.10, "Grocery": 0.02}

# --- Seasonality ---------------------------------------------------------
# Story: Q4 holiday peak (Nov/Dec); Q1 post-holiday dip.
MONTH_SEASONALITY = {1: 0.85, 2: 0.88, 3: 0.95, 4: 0.98, 5: 1.00, 6: 1.02,
                     7: 1.00, 8: 0.98, 9: 1.02, 10: 1.10, 11: 1.35, 12: 1.55}
# Story (INTERACTION): gifting categories spike MORE in Q4 than staples like Grocery.
CATEGORY_Q4_SENSITIVITY = {"Electronics": 1.45, "Fashion": 1.20, "Home & Living": 1.12,
                           "Beauty": 1.30, "Sports": 1.05, "Grocery": 0.85}

# --- Campaigns -----------------------------------------------------------
# Each (month, country) runs ONE headline campaign.
# Tuple = (order_lift, extra_discount). Story: promotions lift volume but
# the extra discount eats into margin -> the classic revenue/margin trade-off.
CAMPAIGN_EFFECT = {
    "None":          (0.00, 0.00),
    "Brand":         (0.08, 0.02),
    "Performance":   (0.18, 0.05),
    "Influencer":    (0.14, 0.04),
    "Seasonal Sale": (0.35, 0.15),
}

# Return-reason mix differs by category (story-consistent).
CATEGORY_RETURN_REASONS = {
    "Electronics":   {"Defective": 0.55, "Changed Mind": 0.30, "Wrong Item": 0.15},
    "Fashion":       {"Size/Fit": 0.65, "Changed Mind": 0.25, "Defective": 0.10},
    "Home & Living": {"Damaged in Transit": 0.45, "Changed Mind": 0.35, "Wrong Item": 0.20},
    "Beauty":        {"Allergic Reaction": 0.40, "Changed Mind": 0.40, "Wrong Item": 0.20},
    "Sports":        {"Size/Fit": 0.45, "Defective": 0.30, "Changed Mind": 0.25},
    "Grocery":       {"Damaged in Transit": 0.55, "Wrong Item": 0.30, "Expired": 0.15},
}

# ===========================================================================
# 3. HELPERS
# ===========================================================================
def noise(low=0.90, high=1.10):
    """Final ±10% realism layer."""
    return RNG.uniform(low, high)

def pick_weighted(weight_dict):
    keys = list(weight_dict)
    probs = np.array(list(weight_dict.values()), dtype=float)
    probs = probs / probs.sum()
    return RNG.choice(keys, p=probs)

def assign_campaign(month_num):
    """Q4 heavily favours Seasonal Sale; other months a softer mix."""
    if month_num in (11, 12):
        weights = {"Seasonal Sale": 0.60, "Performance": 0.20, "Influencer": 0.12, "Brand": 0.08}
    else:
        weights = {"None": 0.45, "Performance": 0.20, "Brand": 0.18, "Influencer": 0.12, "Seasonal Sale": 0.05}
    return pick_weighted(weights)

# Pre-assign one campaign per (month, country) so all categories in that
# market share the same headline promotion that month.
CAMPAIGN_BY_MONTH_COUNTRY = {
    (m, c): assign_campaign(m.month) for m in MONTHS for c in COUNTRIES
}

# ===========================================================================
# 4. ROW GENERATION
# ===========================================================================
BASE_ORDERS = 320          # baseline orders for a "neutral" cell
BASE_AOV = 90.0            # baseline average order value (USD)
BASE_ROAS = 8.0            # baseline revenue per marketing dollar (before country efficiency)
                           # ~8 keeps marketing spend at a realistic 10-15% of revenue

rows = []
for month, country, channel, segment, category in product(
        MONTHS, COUNTRIES, CHANNELS, SEGMENTS, CATEGORIES):

    m = month.month
    campaign = CAMPAIGN_BY_MONTH_COUNTRY[(month, country)]
    camp_lift, camp_extra_disc = CAMPAIGN_EFFECT[campaign]

    subcategory = pick_weighted(CATEGORY_SUBCATS[category])

    # --- seasonality with category interaction ---------------------------
    season = MONTH_SEASONALITY[m]
    if m in (10, 11, 12):  # Q4: gifting categories react more strongly
        season = 1 + (season - 1) * CATEGORY_Q4_SENSITIVITY[category]

    # --- ORDERS ----------------------------------------------------------
    orders = (BASE_ORDERS
              * COUNTRY_REVENUE_MULT[country]
              * CHANNEL_REVENUE_MULT[channel]
              * SEGMENT_REVENUE_MULT[segment]
              * season
              * (1 + camp_lift))
    # INTERACTION: New customers over-index on the Mobile App (younger audience).
    if channel == "Mobile App" and segment == "New":
        orders *= 1.25
    orders = max(1, int(round(orders * noise())))

    # --- AVERAGE ORDER VALUE --------------------------------------------
    aov = BASE_AOV * CATEGORY_AOV_MULT[category] * SEGMENT_AOV_MULT[segment] * noise()
    aov = round(aov, 2)

    # --- REVENUE (consistent: revenue = orders × AOV) --------------------
    revenue = round(orders * aov, 2)

    # --- CUSTOMERS (derived from orders & repeat behaviour) --------------
    customers = max(1, int(round(orders / SEGMENT_ORDERS_PER_CUST[segment] * noise())))

    # --- MARKETING SPEND (efficiency varies by country) ------------------
    roas = BASE_ROAS * COUNTRY_MKT_EFFICIENCY[country] * noise()
    marketing_spend = round(revenue / roas, 2)

    # --- DISCOUNT & MARGIN (campaign discount eats margin) ---------------
    base_discount = 0.05 + (0.03 if segment == "VIP" else 0.0)  # VIPs get loyalty discounts
    discount_rate = round(min(0.45, base_discount + camp_extra_disc) * noise(0.95, 1.05), 4)
    gross_margin_pct = max(0.05, CATEGORY_MARGIN[category] - discount_rate)
    gross_profit = round(revenue * gross_margin_pct, 2)

    # --- RETURNS (category × channel) ------------------------------------
    return_rate = round(CATEGORY_RETURN_BASE[category] * CHANNEL_RETURN_MULT[channel] * noise(), 4)
    return_rate = min(return_rate, 0.40)
    # Returned revenue is partly recoverable; assume 60% of returned value is lost.
    return_cost = round(revenue * return_rate * 0.60, 2)
    return_reason = pick_weighted(CATEGORY_RETURN_REASONS[category])

    # --- SHIPPING (per order; cheaper in-store) --------------------------
    ship_per_order = 4.5 * (0.4 if channel == "Retail" else 1.0)
    shipping_cost = round(orders * ship_per_order * noise(), 2)

    # --- NET PROFIT (full P&L chain) -------------------------------------
    net_profit = round(gross_profit - marketing_spend - shipping_cost - return_cost, 2)

    # --- CONVERSION RATE (channel-driven, display metric) ----------------
    conversion_rate = round(CHANNEL_CONV_BASE[channel] * noise(), 4)

    # --- SATISFACTION (HK scale penalty + high returns hurt) -------------
    satisfaction = (COUNTRY_SAT_BASE[country]
                    - return_rate * 3.0           # high returns drag satisfaction
                    + RNG.normal(0, 0.08))
    satisfaction = round(float(np.clip(satisfaction, 1.0, 5.0)), 2)

    rows.append({
        "Month": str(month),
        "Year": month.year,
        "Quarter": f"Q{(m - 1) // 3 + 1}",
        "Region": REGION,
        "Country": country,
        "Channel": channel,
        "CustomerSegment": segment,
        "ProductCategory": category,
        "ProductSubcategory": subcategory,
        "CampaignType": campaign,
        "Revenue": revenue,
        "Orders": orders,
        "Customers": customers,
        "AverageOrderValue": aov,
        "ConversionRate": conversion_rate,
        "MarketingSpend": marketing_spend,
        "DiscountRate": discount_rate,
        "GrossMarginPct": round(gross_margin_pct, 4),
        "GrossProfit": gross_profit,
        "ShippingCost": shipping_cost,
        "ReturnRate": return_rate,
        "ReturnCost": return_cost,
        "ReturnReason": return_reason,
        "NetProfit": net_profit,
        "CustomerSatisfaction": satisfaction,
    })

df = pd.DataFrame(rows)
df.to_csv("data/business_data.csv", index=False)
print(f"Generated {len(df):,} rows -> data/business_data.csv")
