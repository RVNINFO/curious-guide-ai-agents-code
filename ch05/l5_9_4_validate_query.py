"""Listing 5.9.4 - Validate a model-written SQL query before it runs (sqlglot).

The read-only role plus statement_timeout is the real control; the validator is a second layer.
Needs `pip install sqlglot` (listed in requirements.txt).  Run:  python -m ch05.l5_9_4_validate_query
"""
# using the sqlglot library (pip install sqlglot)
import sqlglot
from sqlglot import exp

ALLOWED = {"invoices", "purchase_orders", "goods_receipts"}
ALLOWED_FUNCTIONS = {"count", "sum", "avg", "min", "max",
                     "coalesce", "round"}
MAX_ROWS = 200


def validate(sql: str) -> tuple[bool, str]:
    try:
        trees = sqlglot.parse(sql, read="postgres")
    except sqlglot.errors.ParseError:
        return False, "unparseable"
    if len(trees) != 1 or trees[0] is None:
        return False, "exactly one statement"
    tree = trees[0]
    if not isinstance(tree, exp.Select):
        return False, "SELECT only"
    tables = set()
    for t in tree.find_all(exp.Table):
        if t.db or t.catalog:
            return False, "schema names not allowed"
        tables.add(t.name.lower())
    tables -= {c.alias.lower() for c in tree.find_all(exp.CTE)}
    if not tables <= ALLOWED:
        return False, f"table not allowed: {tables - ALLOWED}"
    for f in tree.find_all(exp.Func):
        name = (f.name if isinstance(f, exp.Anonymous)
                else f.sql_name()).lower()
        if name not in ALLOWED_FUNCTIONS:
            return False, f"function not allowed: {name}"
    limit = tree.args.get("limit")
    n = None
    if (isinstance(limit, exp.Limit)
            and isinstance(limit.expression, exp.Literal)
            and limit.expression.is_int):
        n = int(limit.expression.name)
    if n is None or n > MAX_ROWS:
        tree = tree.limit(MAX_ROWS)
    return True, tree.sql(dialect="postgres")


if __name__ == "__main__":
    for q in ["SELECT * FROM invoices", "DROP TABLE invoices", "SELECT * FROM users",
              "SELECT 1; SELECT 2", "SELECT * FROM invoices LIMIT 5",
              "SELECT * FROM invoices LIMIT 1000000",
              "SELECT * FROM other_schema.invoices",
              "SELECT pg_sleep(5) FROM invoices"]:
        print(q, "->", validate(q))
