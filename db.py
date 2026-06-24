"""
db.py
=====
DuckDB warehouse layer for the AI Business Analyst Assistant (Phase 1, Step 1).

Why DuckDB (and not Postgres or raw CSV)?
  * Stronger than CSV: a CSV is not a database — you can't run SQL against it.
    text-to-SQL needs a real SQL engine for the model to target.
  * Simpler than Postgres: DuckDB is in-process (runs inside Python, no server,
    no credentials, no managed host). It deploys on Streamlit Cloud with zero
    infrastructure, unlike Postgres.
  * Built for analytics: columnar storage makes GROUP BY / aggregation fast on
    our 6,912 rows and far beyond, using standard SQL.
The CSV stays the source of truth; the warehouse is rebuilt from it on launch.

Run `python db.py` to load the data and print the schema description.
"""

import duckdb

CSV_PATH = "data/business_data.csv"
TABLE = "sales"


def get_connection(csv_path: str = CSV_PATH):
    """Return an in-memory DuckDB connection with the sales table loaded.

    In-memory keeps deployment trivial: nothing to persist or clean up. In the
    Streamlit app this is wrapped in @st.cache_resource so the load runs once.
    """
    con = duckdb.connect(":memory:")
    con.execute(f"CREATE TABLE {TABLE} AS SELECT * FROM read_csv_auto('{csv_path}')")
    return con


def get_schema(con, table: str = TABLE, max_distinct: int = 30) -> str:
    """Build an LLM-friendly schema description.

    For categorical (VARCHAR) columns we list the distinct values when there are
    few enough. This is the single most important input to text-to-SQL accuracy:
    it lets the model write  WHERE Country = 'Hong Kong'  instead of guessing
    'HK'. Numeric columns just show their type.
    """
    cols = con.execute(f"DESCRIBE {table}").fetchdf()
    lines = [f"Table: {table}", "Columns:"]
    for _, row in cols.iterrows():
        name = row["column_name"]
        dtype = str(row["column_type"])
        line = f"  - {name} ({dtype})"
        if "VARCHAR" in dtype.upper():
            n = con.execute(f'SELECT COUNT(DISTINCT "{name}") FROM {table}').fetchone()[0]
            if n <= max_distinct:
                vals = [r[0] for r in con.execute(
                    f'SELECT DISTINCT "{name}" FROM {table} ORDER BY 1').fetchall()]
                line += ": " + ", ".join(map(str, vals))
            else:
                line += f": {n} distinct values"
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    con = get_connection()
    rows = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    print(f"Loaded {rows:,} rows into table '{TABLE}'.\n")
    print(get_schema(con))