"""Listing 5.9.4 - Validate a model-written SQL query before it runs (sqlglot).

Needs `pip install sqlglot` (listed in requirements.txt).  Run:  python -m ch05.l5_9_4_validate_query
"""
import sqlglot
from sqlglot import exp

ALLOWED = {"invoices", "purchase_orders", "goods_receipts"}
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
    tables = {t.name.lower() for t in tree.find_all(exp.Table)}
    tables -= {c.alias.lower() for c in tree.find_all(exp.CTE)}
    if not tables <= ALLOWED:
        return False, f"table not allowed: {tables - ALLOWED}"
    if tree.args.get("limit") is None:
        tree = tree.limit(MAX_ROWS)
    return True, tree.sql(dialect="postgres")


if __name__ == "__main__":
    for q in ["SELECT * FROM invoices", "DROP TABLE invoices", "SELECT * FROM users",
              "SELECT 1; SELECT 2", "SELECT * FROM invoices LIMIT 5"]:
        print(q, "->", validate(q))
