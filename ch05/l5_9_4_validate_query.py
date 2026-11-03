"""Listing 5.9.4 - Validate a model-written SQL query before it runs (sqlglot).

The read-only role plus statement_timeout is the real control; the validator is a second layer.
Needs `pip install sqlglot` (listed in requirements.txt).  Run:  python -m ch05.l5_9_4_validate_query
"""
import sqlglot
from sqlglot import exp

ALLOWED = {"invoices", "purchase_orders", "goods_receipts"}
ALLOWED_FUNCTIONS = {
    "count", "sum", "avg", "min", "max", "date_trunc", "coalesce",
    "lower", "upper", "round", "concat",
}
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

    for t in tree.find_all(exp.Table):
        if t.db or t.catalog:
            return False, f"schema qualification not allowed: {t.sql()}"

    tables = {t.name.lower() for t in tree.find_all(exp.Table)}
    tables -= {c.alias.lower() for c in tree.find_all(exp.CTE)}
    if not tables <= ALLOWED:
        return False, f"table not allowed: {tables - ALLOWED}"

    for f in tree.find_all((exp.Func, exp.Anonymous)):
        fn_name = (f.name if isinstance(f, exp.Anonymous) else f.sql_name()).lower()
        if fn_name not in ALLOWED_FUNCTIONS:
            return False, f"function not allowed: {fn_name}"

    lim = tree.args.get("limit")
    n = int(lim.expression.name) if (lim and hasattr(lim.expression, "name") and lim.expression.name.isdigit()) else None
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
