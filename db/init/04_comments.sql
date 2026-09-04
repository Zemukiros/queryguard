-- Column and table semantics, stated as facts about what each column holds.
--
-- These are not documentation for humans: queryguard.schema.introspect reads them
-- straight out of pg_description and renders them into the schema block that goes
-- into every LLM call. A model that cannot tell gross from net, or a cancelled
-- order from a refunded one, writes plausible SQL that returns the wrong number.
-- So every business term that could be read two ways is pinned down here.
--
-- COMMENT ON is idempotent, so this file can be re-applied to a running database
-- without a full rebuild:  psql -U queryguard -d queryguard -f db/init/04_comments.sql

BEGIN;

-- ---------------------------------------------------------------- categories
COMMENT ON TABLE categories IS
    'Product category tree; 6 top-level plus 6 children. Two rows are inactive.';
COMMENT ON COLUMN categories.description IS
    'Free-text category summary. NULL when not written.';
COMMENT ON COLUMN categories.parent_category_id IS
    'Parent category. NULL means a top-level category.';
COMMENT ON COLUMN categories.is_active IS
    'False means retired, but products still reference it.';
COMMENT ON COLUMN categories.created_at IS
    'When the category row was created.';

-- ----------------------------------------------------------------- customers
COMMENT ON TABLE customers IS
    'One row per registered customer. Customers 481-500 have no orders.';
COMMENT ON COLUMN customers.phone IS
    'Contact phone. NULL means not on file.';
COMMENT ON COLUMN customers.city IS
    'City name. NULL means not on file.';
COMMENT ON COLUMN customers.signup_date IS
    'When the customer registered. Unrelated to their first order date.';
COMMENT ON COLUMN customers.is_active IS
    'False means deactivated. Past orders are retained either way.';
COMMENT ON COLUMN customers.marketing_opt_in IS
    'Marketing consent. NULL means never asked, which is distinct from false.';
COMMENT ON COLUMN customers.lifetime_value IS
    'Sum of total_amount for paid, shipped, delivered, refunded orders; NULL means unknown.';

-- ------------------------------------------------------------------ products
COMMENT ON TABLE products IS
    'Product catalogue. Products 186-200 never appear in any order.';
COMMENT ON COLUMN products.description IS
    'Marketing blurb. NULL when not written.';
COMMENT ON COLUMN products.price IS
    'Current list price. The price actually paid is order_items.unit_price.';
COMMENT ON COLUMN products.cost IS
    'Unit cost to the business. NULL means not recorded.';
COMMENT ON COLUMN products.weight_kg IS
    'Shipping weight in kilograms. NULL means not measured.';
COMMENT ON COLUMN products.in_stock IS
    'Mirrors stock_quantity > 0. Not an independent availability flag.';
COMMENT ON COLUMN products.stock_quantity IS
    'Units currently on hand. Zero when out of stock.';
COMMENT ON COLUMN products.discontinued_at IS
    'When the product was withdrawn. NULL means still sold.';
COMMENT ON COLUMN products.created_at IS
    'When the product was added to the catalogue.';

-- -------------------------------------------------------------------- orders
COMMENT ON TABLE orders IS
    'One row per order, cancelled ones included. Exactly three line items each.';
COMMENT ON COLUMN orders.order_date IS
    'When the order was placed. Spans roughly three years.';
COMMENT ON COLUMN orders.status IS
    'pending=unpaid, paid=awaiting shipment, shipped=in transit, delivered=received, cancelled=voided before payment, refunded=money returned.';
COMMENT ON COLUMN orders.total_amount IS
    'Sum of line items after discount. Zero for cancelled. Not reduced by refunds.';
COMMENT ON COLUMN orders.currency IS
    'ISO currency code for total_amount. Always USD in this dataset.';
COMMENT ON COLUMN orders.shipping_address IS
    'Street address. NULL for cancelled orders and some others.';
COMMENT ON COLUMN orders.shipped_at IS
    'When despatched. NULL unless status is shipped, delivered or refunded.';
COMMENT ON COLUMN orders.is_gift IS
    'True when the customer marked the order as a gift.';
COMMENT ON COLUMN orders.discount_code IS
    'Promo code recorded on the order. Does not affect total_amount.';

-- --------------------------------------------------------------- order_items
COMMENT ON TABLE order_items IS
    'Three lines per order. Line total is quantity * unit_price - discount * quantity.';
COMMENT ON COLUMN order_items.quantity IS
    'Units ordered. Zero occurs deliberately in a few rows.';
COMMENT ON COLUMN order_items.unit_price IS
    'Price per unit at order time, within 5% of products.price.';
COMMENT ON COLUMN order_items.discount IS
    'Per-unit discount amount, not the line total. NULL means none.';
COMMENT ON COLUMN order_items.is_gift_wrap IS
    'True when this line was gift wrapped.';

-- ------------------------------------------------------------------- refunds
COMMENT ON TABLE refunds IS
    'Refunds against delivered or refunded orders only. At most one per order.';
COMMENT ON COLUMN refunds.order_item_id IS
    'The refunded line item. NULL means a whole-order refund.';
COMMENT ON COLUMN refunds.amount IS
    'Equals orders.total_amount when is_partial is false, else 10-60% of it.';
COMMENT ON COLUMN refunds.reason IS
    'Customer-stated reason. NULL means not recorded.';
COMMENT ON COLUMN refunds.refunded_at IS
    '1-31 days after shipping or ordering. May be a future date.';
COMMENT ON COLUMN refunds.is_partial IS
    'True when only part of the order was refunded.';
COMMENT ON COLUMN refunds.approved_by IS
    'Staff username. NULL means auto-approved without review.';

COMMIT;
