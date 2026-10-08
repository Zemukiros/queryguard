"""The metric glossary: business definitions every model in the pipeline shares.

Generation, the second-opinion query and the alignment judge all read this one
text. If they disagree about what revenue is, a correct query loses points for
being correct: in refund-fix-2026-10-08 the generator applied the revenue
status rule and the judge, which had never seen it, called the status filter
a discrepancy (refund_04, alignment 0.6).

The same rule is stated in the schema comment on orders.total_amount
(db/init/04_comments.sql), which back-translation reads through the schema.
"""

from __future__ import annotations

GLOSSARY = """\
Glossary (fixed business definitions; they override any reading of your own):
- Revenue, also called gross revenue: sum(orders.total_amount) over orders
  with status IN ('paid', 'shipped', 'delivered', 'refunded'). Pending orders
  are unpaid and cancelled orders were voided: neither is revenue, whatever
  date range or other filter the question adds. Refunded orders still count
  toward gross revenue.
- Net revenue: gross revenue minus refunds.amount on those same orders."""
