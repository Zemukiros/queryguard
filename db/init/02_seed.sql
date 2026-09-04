-- QueryGuard seed data. Pure SQL so it runs inside the container's init hook.
-- setseed() makes the dataset reproducible across rebuilds.
--
-- Pattern note: every random() lives in a derived table that selects FROM the
-- row source (generate_series / orders). An *uncorrelated* subquery — including
-- an uncorrelated LATERAL — is evaluated once and its value reused for every
-- row, which silently collapses the whole dataset onto a single draw.

BEGIN;

-- Deterministic clock. setseed() fixes the random draws; the DATE '2026-09-03'
-- anchor below fixes the dates, so a rebuild on any future day reproduces
-- byte-identical data -- which is what the eval suite's expected answers rely on.
-- TimeZone is pinned too, so the anchor means the same instant on any host.
SET TimeZone = 'UTC';

SELECT setseed(0.42);

-- ---------------------------------------------------------------- categories
INSERT INTO categories (name, slug, description, parent_category_id, is_active) VALUES
    ('Electronics',      'electronics',      'Consumer electronics and gadgets',   NULL, true),
    ('Home & Kitchen',   'home-kitchen',     'Everything for the home',            NULL, true),
    ('Apparel',          'apparel',          NULL,                                 NULL, true),
    ('Sports & Outdoors','sports-outdoors',  'Gear for the outdoors',              NULL, true),
    ('Books',            'books',            NULL,                                 NULL, true),
    ('Toys & Games',     'toys-games',       'Fun for all ages',                   NULL, false),
    ('Laptops',          'laptops',          'Portable computers',                    1, true),
    ('Audio',            'audio',            NULL,                                    1, true),
    ('Cookware',         'cookware',         'Pots, pans and bakeware',               2, true),
    ('Footwear',         'footwear',         NULL,                                    3, true),
    ('Camping',          'camping',          'Tents, packs and sleeping bags',        4, true),
    ('Board Games',      'board-games',      NULL,                                    6, false);

-- The INSERT above omits created_at, which would take DEFAULT now() and reintroduce
-- the wall clock. Pin it to the same anchor as every other generated date.
UPDATE categories SET created_at = DATE '2026-09-03';

-- ----------------------------------------------------------------- customers
-- 500 customers. #481-500 deliberately never place an order.
WITH words AS (
    SELECT
        ARRAY['Ava','Liam','Noah','Mia','Ethan','Zoe','Owen','Isla','Lucas','Nora',
              'Kai','Priya','Omar','Sofia','Jonas','Amara','Diego','Yuki','Elena','Theo']::text[] AS first_names,
        ARRAY['Nguyen','Patel','Garcia','Smith','Okafor','Kim','Rossi','Dubois','Haddad','Silva',
              'Novak','Tanaka','Weber','Costa','Ivanov','Ali','Fischer','Moreau','Berg','Chen']::text[] AS last_names,
        ARRAY['example.com','mailbox.io','testmail.org','inbox.dev']::text[] AS domains,
        -- Parallel arrays share one index: country[i] always matches city[i].
        ARRAY['USA','USA','USA','Canada','UK','Germany','France','Japan','Brazil','India']::text[] AS countries,
        ARRAY['Seattle','Austin','Chicago','Toronto','London','Berlin','Lyon','Osaka','Recife','Pune']::text[] AS cities
)
INSERT INTO customers
    (first_name, last_name, email, phone, country, city, signup_date,
     is_active, marketing_opt_in, lifetime_value)
SELECT
    w.first_names[s.fn_i],
    w.last_names[s.ln_i],
    lower(w.first_names[s.fn_i]) || '.' || lower(w.last_names[s.ln_i])
        || s.g || '@' || w.domains[s.dom_i],
    CASE WHEN s.r_phone < 0.15 THEN NULL                       -- ~15% unknown phone
         ELSE '+1-' || (200 + floor(s.r_area * 700))::int || '-'
              || lpad(floor(s.r_line * 10000)::int::text, 4, '0') END,
    w.countries[s.geo_i],
    CASE WHEN s.r_city < 0.20 THEN NULL ELSE w.cities[s.geo_i] END,
    DATE '2026-09-03' - (s.r_signup * 1460)::int * interval '1 day',
    s.r_active >= 0.05,                                        -- ~5% deactivated
    CASE WHEN s.r_optin < 0.25 THEN NULL ELSE s.r_optin2 < 0.6 END,
    NULL                                  -- backfilled from real order totals below
FROM (
    SELECT g,
           1 + floor(random() * 20)::int AS fn_i,
           1 + floor(random() * 20)::int AS ln_i,
           1 + floor(random() *  4)::int AS dom_i,
           1 + floor(random() * 10)::int AS geo_i,
           random() AS r_phone,  random() AS r_area,   random() AS r_line,
           random() AS r_city,   random() AS r_signup, random() AS r_active,
           random() AS r_optin,  random() AS r_optin2
    FROM generate_series(1, 500) AS g
) AS s
CROSS JOIN words AS w;

-- ------------------------------------------------------------------ products
-- 200 products. #186-200 deliberately never appear in an order.
WITH words AS (
    SELECT
        ARRAY['Nordic','Compact','Pro','Vintage','Ultra','Everyday','Rugged','Silent',
              'Modular','Featherweight','Classic','Solar']::text[] AS adjectives,
        ARRAY['Headphones','Blender','Backpack','Notebook','Sneakers','Kettle','Lamp',
              'Tripod','Jacket','Mouse','Skillet','Tent','Speaker','Monitor','Puzzle']::text[] AS nouns
)
INSERT INTO products
    (sku, name, description, category_id, price, cost, weight_kg,
     in_stock, stock_quantity, discontinued_at, created_at)
SELECT
    'SKU-' || lpad(s.g::text, 5, '0'),
    w.adjectives[s.adj_i] || ' ' || w.nouns[s.noun_i],
    CASE WHEN s.r_desc < 0.20 THEN NULL
         ELSE 'A ' || lower(w.adjectives[s.adj_i]) || ' '
              || lower(w.nouns[s.noun_i]) || ' built to last.' END,
    s.category_id,
    s.price,
    CASE WHEN s.r_cost < 0.30 THEN NULL                        -- ~30% unknown cost
         ELSE round((s.price * (0.35 + s.r_margin * 0.3))::numeric, 2) END,
    CASE WHEN s.r_weight < 0.25 THEN NULL
         ELSE round((0.1 + s.r_weight2 * 12)::numeric, 3) END,
    s.stock > 0,
    s.stock,
    CASE WHEN s.r_disc < 0.08                                  -- ~8% discontinued
         THEN DATE '2026-09-03' - (s.r_disc_age * 400)::int * interval '1 day' END,
    DATE '2026-09-03' - (s.r_created * 1500)::int * interval '1 day'
FROM (
    SELECT g,
           1 + floor(random() * 12)::int AS adj_i,
           1 + floor(random() * 15)::int AS noun_i,
           1 + floor(random() * 12)::int AS category_id,
           round((4.99 + random() * 895)::numeric, 2) AS price,
           CASE WHEN random() < 0.12 THEN 0                    -- ~12% out of stock
                ELSE floor(random() * 400)::int END AS stock,
           random() AS r_desc,   random() AS r_cost,     random() AS r_margin,
           random() AS r_weight, random() AS r_weight2,
           random() AS r_disc,   random() AS r_disc_age, random() AS r_created
    FROM generate_series(1, 200) AS g
) AS s
CROSS JOIN words AS w;

-- -------------------------------------------------------------------- orders
-- 5000 orders spread across the last 3 years (1095 days).
WITH words AS (
    SELECT ARRAY['Maple','Oak','Cedar','Birch','Willow']::text[] AS streets,
           ARRAY['WELCOME10','SAVE5','FREESHIP','BLACKFRIDAY','LOYAL15']::text[] AS promos
)
INSERT INTO orders
    (customer_id, order_date, status, total_amount, currency,
     shipping_address, shipped_at, is_gift, discount_code)
SELECT
    s.customer_id,
    s.order_date,
    s.status,
    0,                                    -- recomputed from order_items below
    'USD',
    CASE WHEN s.status = 'cancelled' OR s.r_addr < 0.08 THEN NULL
         ELSE (100 + floor(s.r_house * 9000))::int || ' '
              || w.streets[s.street_i] || ' St' END,
    CASE WHEN s.status IN ('shipped','delivered','refunded')
         THEN s.order_date + (1 + s.r_ship * 6)::int * interval '1 day' END,
    s.r_gift < 0.12,
    CASE WHEN s.r_promo < 0.70 THEN NULL ELSE w.promos[s.promo_i] END
FROM (
    SELECT
        x.*,
        CASE
            WHEN x.r_status < 0.04 THEN 'cancelled'
            WHEN x.r_status < 0.10 THEN 'pending'
            WHEN x.r_status < 0.22 THEN 'paid'
            WHEN x.r_status < 0.38 THEN 'shipped'
            WHEN x.r_status < 0.94 THEN 'delivered'
            ELSE 'refunded'
        END AS status
    FROM (
        SELECT g,
               1 + floor(random() * 480)::int AS customer_id,  -- #481-500 get none
               DATE '2026-09-03' - (random() * 1095)::int * interval '1 day'
                     - (random() * 86400)::int * interval '1 second' AS order_date,
               random() AS r_status, random() AS r_addr,  random() AS r_house,
               random() AS r_ship,   random() AS r_gift,  random() AS r_promo,
               1 + floor(random() * 5)::int AS street_i,
               1 + floor(random() * 5)::int AS promo_i
        FROM generate_series(1, 5000) AS g
    ) AS x
) AS s
CROSS JOIN words AS w;

-- --------------------------------------------------------------- order_items
-- 3 items per order = 15000 rows. Products 186-200 are never referenced.
INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount, is_gift_wrap)
SELECT
    s.order_id,
    p.product_id,
    CASE WHEN s.r_zero < 0.002 THEN 0                     -- ~30 zero-quantity rows
         ELSE 1 + floor(s.r_qty * 5)::int END,
    round((p.price * (0.95 + s.r_jitter * 0.1))::numeric, 2),  -- price snapshot
    CASE WHEN s.r_disc < 0.60 THEN NULL
         ELSE round((p.price * (0.05 + s.r_disc2 * 0.15))::numeric, 2) END,
    s.r_wrap < 0.07
FROM (
    SELECT o.order_id,
           1 + floor(random() * 185)::int AS product_id,
           random() AS r_zero, random() AS r_qty,   random() AS r_jitter,
           random() AS r_disc, random() AS r_disc2, random() AS r_wrap
    FROM orders AS o
    CROSS JOIN generate_series(1, 3) AS n
) AS s
JOIN products AS p ON p.product_id = s.product_id;

-- Edge case: one order where every line is zero-quantity.
UPDATE order_items
SET quantity = 0
WHERE order_id = (SELECT min(order_id) FROM orders WHERE status = 'delivered');

-- Keep order totals consistent with their line items; cancelled orders bill nothing.
UPDATE orders AS o
SET total_amount = CASE
        WHEN o.status = 'cancelled' THEN 0
        ELSE GREATEST(COALESCE(t.line_total, 0), 0)
    END
FROM (
    SELECT order_id,
           round(sum(quantity * unit_price - COALESCE(discount, 0) * quantity), 2) AS line_total
    FROM order_items
    GROUP BY order_id
) AS t
WHERE t.order_id = o.order_id;

-- ------------------------------------------------------------------- refunds
-- 300 refunds, only against orders that actually completed.
WITH words AS (
    SELECT ARRAY['Damaged in transit','Wrong item shipped','Item not as described',
                 'Arrived late','Changed mind','Defective on arrival']::text[] AS reasons,
           ARRAY['s.morgan','r.patel','j.kim','support-bot']::text[] AS approvers
)
INSERT INTO refunds (order_id, order_item_id, amount, reason, refunded_at, is_partial, approved_by)
SELECT
    o.order_id,
    CASE WHEN o.r_item < 0.60 THEN item.order_item_id END,   -- NULL = whole-order refund
    CASE WHEN o.is_partial
         THEN round((o.total_amount * (0.1 + o.r_amount * 0.5))::numeric, 2)
         ELSE o.total_amount END,
    CASE WHEN o.r_reason < 0.40 THEN NULL ELSE w.reasons[o.reason_i] END,
    COALESCE(o.shipped_at, o.order_date) + (1 + o.r_days * 30)::int * interval '1 day',
    o.is_partial,
    CASE WHEN o.r_approver < 0.35 THEN NULL ELSE w.approvers[o.approver_i] END
FROM (
    -- Two levels on purpose: `ORDER BY random()` is reused as the target-list
    -- random(), and taking the smallest 300 of ~3000 would bias every derived
    -- value toward zero. So sample first, then draw attributes on the sample.
    SELECT pick.*,
           random() < 0.55 AS is_partial,
           random() AS r_item,   random() AS r_amount, random() AS r_reason,
           random() AS r_days,   random() AS r_approver,
           1 + floor(random() * 6)::int AS reason_i,
           1 + floor(random() * 4)::int AS approver_i
    FROM (
        SELECT order_id, total_amount, order_date, shipped_at
        FROM orders
        WHERE status IN ('delivered','refunded') AND total_amount > 0
        ORDER BY random()
        LIMIT 300
    ) AS pick
) AS o
CROSS JOIN words AS w
-- Correlated (references o.order_id), so this LATERAL really does run per row.
LEFT JOIN LATERAL (
    SELECT order_item_id FROM order_items WHERE order_id = o.order_id LIMIT 1
) AS item ON true;

-- Backfill customer lifetime value from real revenue; leave ~10% NULL (unknown).
UPDATE customers AS c
SET lifetime_value = CASE WHEN random() < 0.10 THEN NULL ELSE r.revenue END
FROM (
    SELECT customer_id, round(sum(total_amount), 2) AS revenue
    FROM orders
    WHERE status IN ('paid','shipped','delivered','refunded')
    GROUP BY customer_id
) AS r
WHERE r.customer_id = c.customer_id;

COMMIT;

ANALYZE;
