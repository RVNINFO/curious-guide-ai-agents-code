"""In-memory stand-in for the `db` object the book's listings assume ("your own database client")."""


class _Collection:
    def __init__(self, rows):
        self.rows = rows

    def find_one(self, query: dict):
        return next((r for r in self.rows if all(r.get(k) == v for k, v in query.items())), None)

    def find(self, query: dict):
        return [r for r in self.rows if all(r.get(k) == v for k, v in query.items())]


class FakeDB:
    def __init__(self):
        self.orders = _Collection([
            {"id": "ORD-1234", "email": "ana@example.com", "status": "shipped",
             "carrier": "BlueDart", "tracking_number": "BD123", "estimated_delivery": "2026-11-02"},
            {"id": "ORD-1235", "email": "ana@example.com", "status": "delivered",
             "carrier": "DTDC", "tracking_number": "DT777", "estimated_delivery": "2026-10-20"},
        ])


db = FakeDB()
