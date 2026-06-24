"""
text_to_sql.py
==============
Turn a natural-language business question into a DuckDB SQL query (Phase 1, Step 2).

The model never touches the database here — it only writes a query string.
Step 3 will validate that string with guardrails; Step 4 will execute it.

Run `python text_to_sql.py` (with OPENAI_API_KEY set and duckdb installed) to see
the model write SQL for a few sample questions.
"""

import re

# The instruction block. The strict "ONLY SQL" rules are what keep the output
# parseable; the DuckDB-dialect and exact-value rules are what keep it correct.
SYSTEM_RULES = """You are a precise text-to-SQL generator for a DuckDB database.
Given the schema and a business question, return ONE DuckDB SQL query that answers it.

Rules:
- Output ONLY the SQL query. No explanation, no commentary, no markdown code fences.
- Use only the table and columns shown in the schema.
- Use DuckDB SQL syntax.
- Read only: use SELECT. Never write, modify, or delete data.
- For "which / best / highest / top" questions, select the grouping column(s) and
  the metric, ORDER BY the metric, and add a sensible LIMIT.
- Use categorical values exactly as written in the schema (e.g. 'Hong Kong', not 'HK').
- For ratio metrics (e.g. marketing efficiency = revenue per marketing dollar),
  compute SUM(numerator) / SUM(denominator), not the average of per-row ratios.
"""


def build_sql_prompt(question: str, schema: str) -> str:
    """Assemble the full prompt: rules + schema + question."""
    return (f"{SYSTEM_RULES}\n"
            f"=== Schema ===\n{schema}\n\n"
            f"=== Question ===\n{question}\n\n"
            f"=== SQL ===")


def clean_sql(raw: str) -> str:
    """Strip the wrappers the model sometimes adds, returning bare SQL.

    Handles: ```sql ... ``` fences, plain ``` fences, a leading 'SQL:' label,
    and trailing semicolons/whitespace.
    """
    s = raw.strip()
    s = re.sub(r"^```(?:sql)?\s*", "", s, flags=re.I)  # opening fence
    s = re.sub(r"\s*```$", "", s)                       # closing fence
    s = re.sub(r"^\s*sql\s*:\s*", "", s, flags=re.I)    # 'SQL:' label
    return s.strip().rstrip(";").strip()


def generate_sql(question: str, schema: str, client, model: str = "gpt-5-mini") -> str:
    """Call the LLM and return a cleaned SQL string (not yet validated/executed)."""
    prompt = build_sql_prompt(question, schema)
    resp = client.responses.create(model=model, input=prompt)
    return clean_sql(resp.output_text)


if __name__ == "__main__":
    import db
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI()
    con = db.get_connection()
    schema = db.get_schema(con)

    samples = [
        "Which country has the best marketing efficiency?",
        "Which product category has the highest return rate?",
        "Show total revenue by month over time.",
    ]
    for q in samples:
        print(f"Q:   {q}")
        print(f"SQL: {generate_sql(q, schema, client)}\n")