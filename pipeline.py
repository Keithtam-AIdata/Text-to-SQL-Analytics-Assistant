"""
pipeline.py
===========
The full text-to-SQL pipeline with self-correction (Phase 1, Step 4).

Flow:
    question
      -> generate SQL            (text_to_sql)
      -> validate                (sql_guard)        guardrail failure = hard stop
      -> execute                 (DuckDB)
      -> on execution error: feed the error back to the model, retry ONCE
      -> return (sql, dataframe) or a clean error

Why distinguish the two failure types:
  * A guardrail failure is a SAFETY problem (the model tried something unsafe).
    Retrying would just waste tokens — we stop.
  * An execution error is a CAPABILITY problem (wrong column, syntax slip). The
    model can often fix it if shown the error — so we give it one chance.

Run `python pipeline.py` (duckdb + openai + OPENAI_API_KEY) for a live demo.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd

import text_to_sql as t2s
import sql_guard


@dataclass
class QueryResult:
    question: str
    sql: Optional[str]
    data: Optional[pd.DataFrame]
    error: Optional[str] = None
    repaired: bool = False   # True if it succeeded only after self-correction
    blocked: bool = False    # True if stopped by a guardrail


def _build_repair_prompt(question: str, schema: str, bad_sql: str, error: str) -> str:
    """Prompt for the retry: show the failed SQL and the error, ask for a fix."""
    return (f"{t2s.SYSTEM_RULES}\n"
            f"=== Schema ===\n{schema}\n\n"
            f"Your previous SQL failed to execute.\n"
            f"Previous SQL:\n{bad_sql}\n\n"
            f"Database error:\n{error}\n\n"
            f"Return a corrected DuckDB SQL query. Output ONLY the SQL.\n"
            f"=== Question ===\n{question}\n\n=== SQL ===")


def run_question(question: str, con, schema: str, client, model: str = "gpt-5-mini") -> QueryResult:
    # 1. generate
    sql = t2s.generate_sql(question, schema, client, model)

    # 2. validate — safety failures are NOT retried
    try:
        safe_sql = sql_guard.validate_sql(sql)
    except sql_guard.UnsafeSQLError as e:
        return QueryResult(question, sql, None, error=f"Blocked by guardrail: {e}", blocked=True)

    # 3. execute
    try:
        data = con.execute(safe_sql).fetchdf()
        return QueryResult(question, safe_sql, data)
    except Exception as exec_err:
        # 4. self-correction: one retry with the error fed back
        repair_prompt = _build_repair_prompt(question, schema, safe_sql, str(exec_err))
        resp = client.responses.create(model=model, input=repair_prompt)
        sql2 = t2s.clean_sql(resp.output_text)
        try:
            safe_sql2 = sql_guard.validate_sql(sql2)
        except sql_guard.UnsafeSQLError as e:
            return QueryResult(question, sql2, None, error=f"Repair blocked by guardrail: {e}", blocked=True)
        try:
            data = con.execute(safe_sql2).fetchdf()
            return QueryResult(question, safe_sql2, data, repaired=True)
        except Exception as exec_err2:
            return QueryResult(question, safe_sql2, None,
                               error=f"Failed after one self-correction: {exec_err2}")


def interpret_result(question: str, sql: str, df, client, model: str = "gpt-5-mini") -> str:
    """Explain an executed query result in business language (4-part format).

    The model interprets; it never computes. It is given the exact rows Python
    produced and told to use only those numbers.
    """
    table_text = df.head(20).to_string(index=False)
    prompt = (
        "You are a senior business analyst explaining a query result to executives.\n"
        "Python already ran SQL and produced the result below. Use ONLY these numbers —\n"
        "invent nothing.\n\n"
        "Writing discipline (important):\n"
        "1. Plain English, never raw column names. Translate field names into natural\n"
        "   wording: 'revenue_per_marketing' -> 'revenue per marketing dollar',\n"
        "   'weighted_customer_satisfaction' -> 'customer satisfaction'. No underscores,\n"
        "   no code-style names anywhere in your answer.\n"
        "2. Be selective, not exhaustive. Cite only the few numbers that actually support\n"
        "   the conclusion (2-4 of them). Ignore metrics that are nearly identical across\n"
        "   rows or irrelevant to the question — a good analyst chooses, never dumps.\n"
        "3. Format every number for a human: 11.58 not 11.584695; $9.23M not 9230791.08;\n"
        "   4.24 / 5 for ratings; 8.4% for rates. Round sensibly.\n\n"
        "Respond in exactly this format, keeping each part tight:\n"
        "Direct Answer:\n[One or two sentences. The conclusion only.]\n\n"
        "KPI Evidence:\n[Only the 2-4 formatted numbers that support the answer.]\n\n"
        "Business Interpretation:\n[2-3 sentences on what it means.]\n\n"
        "Recommended Next Step:\n[One practical action.]\n\n"
        f"=== Question ===\n{question}\n\n"
        f"=== SQL executed ===\n{sql}\n\n"
        f"=== Query result ===\n{table_text}\n"
    )
    resp = client.responses.create(model=model, input=prompt)
    return resp.output_text


if __name__ == "__main__":
    import db
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI()
    con = db.get_connection()
    schema = db.get_schema(con)

    for q in [
        "Which country has the best marketing efficiency?",
        "Which product category is most profitable?",
        "What is the average order value by customer segment?",
    ]:
        r = run_question(q, con, schema, client)
        print(f"Q: {q}")
        if r.error:
            print(f"   ERROR: {r.error}")
        else:
            print(f"   SQL: {r.sql}")
            print(f"   repaired: {r.repaired}")
            print(f"   result: {r.data.head(3).to_dict('records')}")
        print()