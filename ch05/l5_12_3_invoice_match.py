"""Listing 5.12.3 - A deterministic three-way matcher (purchase order, goods receipt, invoice).

The model never does this arithmetic: it receives the structured result and explains it.
Run:  python -m ch05.l5_12_3_invoice_match
"""
from decimal import Decimal

PO = {"po": "45000987", "item": 10,
      "qty": Decimal(10), "price": Decimal(1000)}
GR = [{"gr": "50003456", "po": "45000987", "item": 10,
       "qty": Decimal(8), "reversed": False}]
INV = {"id": "INV-78421", "po": "45000987", "item": 10,
       "qty": Decimal(10), "price": Decimal(1000)}


def match(po, receipts, inv, qty_tolerance=Decimal(0)):
    if po is None or inv is None:
        return {"status": "insufficient_evidence",
                "requires_human_review": True}
    keys = {(po["po"], po["item"])}
    keys |= {(r["po"], r["item"]) for r in receipts}
    if keys != {(inv["po"], inv["item"])}:
        return {"status": "conflict",
                "requires_human_review": True}
    received = sum((r["qty"] for r in receipts
                    if not r["reversed"]), Decimal(0))
    extra = max(inv["qty"] - received, Decimal(0))
    ok = extra <= qty_tolerance
    return {"status": "matched" if ok else "mismatch",
            "ordered": po["qty"], "received": received,
            "invoiced": inv["qty"],
            "calculated_difference": extra,
            "indicative_value": extra * po["price"],
            "source_refs": [inv["id"], po["po"]]
                           + [r["gr"] for r in receipts],
            "requires_human_review": not ok}


if __name__ == "__main__":
    # Read the tolerance from the company code's settings,
    # never hard-code it here.
    print(match(PO, GR, INV, qty_tolerance=Decimal(0)))
