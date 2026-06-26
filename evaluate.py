"""
evaluate.py
===========
A text-to-SQL evaluation harness (Phase 1, Step 5).

Generating answers is easy; knowing how often they're RIGHT is the senior part.
This runs a fixed set of questions whose correct answers are locked from the real
data, scores the pipeline against them, and reports an accuracy number.

How "correct" is judged:
  The pipeline returns a DataFrame whose columns/format vary by run, so we can't
  string-match the whole thing. Each question here has a single-label answer
  (e.g. 'Japan'). We count it correct if that label appears in the FIRST ROW of
  the result — which also catches ordering mistakes (ask "lowest", get the
  highest, and the first row is wrong).

Run `python evaluate.py` (duckdb + openai + OPENAI_API_KEY) to score the pipeline.
"""

from pipeline import run_question

# Each answer was verified against the real data with canonical SQL.
EVAL_SET = [
    {"q": "Which country has the best marketing efficiency?",            "expect": "Japan"},
    {"q": "Which product category has the highest return rate?",         "expect": "Fashion"},
    {"q": "Which product category is most profitable?",                  "expect": "Beauty"},
    {"q": "Which product category has the highest average order value?", "expect": "Electronics"},
    {"q": "Which country has the highest revenue?",                      "expect": "Hong Kong"},
    {"q": "Which country has the lowest customer satisfaction?",         "expect": "Hong Kong"},
    {"q": "Which customer segment has the highest average order value?", "expect": "VIP"},
    {"q": "Which channel has the lowest return rate?",                   "expect": "Retail"},
    {"q": "Which quarter has the highest revenue?",                      "expect": "Q4"},
    {"q": "Which product category has the lowest return rate?",          "expect": "Grocery"},
]


def hit(result_df, expect) -> bool:
    """True if `expect` appears (case-insensitive) in the first row of results."""
    if result_df is None or len(result_df) == 0:
        return False
    cells = [str(v).strip().lower() for v in result_df.iloc[0].values]
    return str(expect).strip().lower() in cells


def evaluate(con, schema, client, model: str = "gpt-5-mini"):
    """Run every eval question through the pipeline and score it.
    Returns (rows, passed) where rows = list of (question, expect, ok, detail)."""
    rows = []
    passed = 0
    for case in EVAL_SET:
        r = run_question(case["q"], con, schema, client, model)
        if r.error:
            ok, detail = False, r.error
        else:
            ok = hit(r.data, case["expect"])
            detail = r.data.iloc[0].to_dict() if len(r.data) else "empty"
        passed += ok
        rows.append((case["q"], case["expect"], ok, detail))
    return rows, passed


if __name__ == "__main__":
    import db
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI()
    con = db.get_connection()
    schema = db.get_schema(con)

    rows, passed = evaluate(con, schema, client)
    print("text-to-SQL evaluation\n" + "-" * 60)
    for q, expect, ok, detail in rows:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {q}")
        print(f"        expect={expect} | got={detail}")
    total = len(rows)
    print("-" * 60)
    print(f"Accuracy: {passed}/{total} ({passed/total:.0%})")