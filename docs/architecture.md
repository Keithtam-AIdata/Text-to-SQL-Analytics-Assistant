# Architecture Overview

This document explains the design of the ChatGPT-Style Business Analytics Assistant.

The project is designed as a lightweight AI analytics copilot that combines structured business data, a dynamic KPI engine, and OpenAI-generated business interpretation.

---

## 1. Objective

The goal of this project is to allow business users to ask natural language questions about retail / e-commerce KPI performance and receive structured business insights.

Example questions include:

```text
Which region has the best marketing efficiency?
Which product category has the highest return rate?
Which category has the highest AOV?
Summarize regional performance.
```

The assistant returns answers in a consistent business format:

```text
Direct Answer
KPI Evidence
Business Interpretation
Recommended Next Step
```

---

## 2. High-Level Architecture

```text
User Question
↓
Streamlit Web Interface
↓
Dynamic KPI Engine
↓
Pandas Data Calculation
↓
Prompt Context Construction
↓
OpenAI API
↓
Structured Business Insight
```

---

## 3. Key Design Principle

The language model is not responsible for calculating KPIs from raw data.

Instead, the Python data layer calculates the relevant metrics first using Pandas. The calculated KPI results are then passed to the OpenAI model as context.

This reduces hallucination risk and makes the AI response more grounded in the actual dataset.

---

## 4. Main Components

### 4.1 Streamlit Web Interface

The Streamlit app provides:

* Executive KPI overview
* Dashboard charts
* Natural language question input
* AI-generated business analysis
* Data explorer
* Methodology explanation

The goal of the interface is to make the project feel like a business analytics product prototype rather than a command-line demo.

---

### 4.2 Business Dataset

The dataset is a synthetic retail / e-commerce performance dataset.

It includes:

* Month
* Region
* Product Category
* Revenue
* Orders
* Customers
* Average Order Value
* Conversion Rate
* Marketing Spend
* Return Rate
* Customer Satisfaction

The data is stored in:

```text
data/business_data.csv
```

---

### 4.3 Dynamic KPI Engine

The dynamic KPI engine detects the type of question asked by the user and calculates the relevant KPI summary.

Supported analysis areas include:

* Revenue analysis
* Order volume analysis
* Average Order Value analysis
* Conversion rate analysis
* Return rate analysis
* Customer satisfaction analysis
* Marketing efficiency analysis

Example:

If the user asks:

```text
Which region has the best marketing efficiency?
```

The system calculates:

```text
Revenue / Marketing Spend
```

by region before sending the results to the AI model.

---

### 4.4 Prompt Context Construction

The application builds a prompt containing:

* General business context
* Relevant dynamic KPI results
* User question
* Response rules
* Required answer format

This ensures the model answers in a consistent and business-focused way.

---

### 4.5 OpenAI Response Layer

The OpenAI API is used to generate the final business explanation.

The model is instructed to:

* Use only the provided business context and KPI results
* Avoid inventing unsupported numbers
* Provide concise business interpretation
* Recommend one practical next step

---

## 5. Example Flow

### User Question

```text
Which product category has the highest return rate?
```

### Dynamic KPI Engine Output

```text
Average Return Rate by Product Category
```

### AI Response

```text
Direct Answer:
Home & Living has the highest return rate.

KPI Evidence:
Home & Living has an average return rate of 8.00%.

Business Interpretation:
The high return rate may indicate product quality, fulfillment, or expectation mismatch issues.

Recommended Next Step:
Perform a return reason analysis for Home & Living products.
```

---

## 6. Why This Architecture Matters

A basic chatbot can answer general questions, but this project is designed to act as a business analytics assistant.

The key difference is that the assistant combines:

```text
Structured business data
+
KPI calculation logic
+
AI-generated interpretation
```

This makes the project relevant to analytics, BI, and AI solution roles.

---

## 7. Future Architecture Enhancements

Potential future improvements include:

* Replace keyword-based KPI detection with LLM-based intent classification
* Add user-uploaded CSV support
* Add monthly trend analysis
* Add conversation memory
* Add document-based RAG for business reports
* Add vector database support
* Deploy with secure secrets management
* Add authentication and usage limits for public demos
