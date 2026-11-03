"""In-memory stand-in for the `db` object the book's listings assume ("your own database client").

Mimics the small slice of a MongoDB-style API the listings use: find_one(query), and
find(query).sort(field, direction).limit(n).
"""


class _Cursor:
    def __init__(self, rows):
        self.rows = list(rows)

    def sort(self, field, direction=1):
        self.rows.sort(key=lambda r: r[field], reverse=direction < 0)
        return self

    def limit(self, n):
        self.rows = self.rows[:n]
        return self

    def __iter__(self):
        return iter(self.rows)


class _Collection:
    def __init__(self, rows):
        self.rows = rows

    def find_one(self, query: dict):
        return next((r for r in self.rows if all(r.get(k) == v for k, v in query.items())), None)

    def find(self, query: dict) -> _Cursor:
        return _Cursor(r for r in self.rows if all(r.get(k) == v for k, v in query.items()))


class FakeDB:
    def __init__(self):
        self.orders = _Collection([
            {"id": "ORD-1234", "email": "ana@example.com", "status": "shipped",
             "carrier": "BlueDart", "tracking_number": "BD123", "estimated_delivery": "2026-11-02",
             "created_at": "2026-10-25"},
            {"id": "ORD-1235", "email": "ana@example.com", "status": "delivered",
             "carrier": "DTDC", "tracking_number": "DT777", "estimated_delivery": "2026-10-20",
             "created_at": "2026-10-10"},
        ])


db = FakeDB()
