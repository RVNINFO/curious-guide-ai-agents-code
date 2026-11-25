"""Listing 5.9.4 - Validate a model-written SQL query before it runs (sqlglot).

The read-only role plus statement_timeout is the real control; the validator is a second layer.
It rejects SELECT INTO and locking clauses (FOR UPDATE) as well as writes, even inside a CTE.
Needs `pip install sqlglot` (listed in requirements.txt).  Run:  python -m ch05.l5_9_4_validate_query
"""
# using the sqlglot library (pip install sqlglot)
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
    if tree.find(exp.Into, exp.Lock):
        return False, "SELECT INTO / locking not allowed"
    if tree.find(exp.Insert, exp.Update,
                 exp.Delete, exp.Merge):
        return False, "writes not allowed"
    tables = {t.name.lower() for t in tree.find_all(exp.Table)}
    tables -= {c.alias.lower() for c in tree.find_all(exp.CTE)}
    if not tables <= ALLOWED:
        return False, f"table not allowed: {tables - ALLOWED}"
    if tree.args.get("limit") is None:
        tree = tree.limit(MAX_ROWS)
    return True, tree.sql(dialect="postgres")


if __name__ == "__main__":
    for q in ["SELECT * FROM invoices", "DROP TABLE invoices", "SELECT * FROM users",
              "SELECT 1; SELECT 2", "SELECT * FROM invoices LIMIT 5",
              "SELECT * INTO invoices_copy FROM goods_receipts",
              "SELECT * FROM invoices FOR UPDATE"]:
        print(q, "->", validate(q))
