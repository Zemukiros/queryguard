-- QueryGuard e-commerce schema.
-- Runs automatically on first container start (empty data directory).

BEGIN;

CREATE TABLE categories (
    category_id        serial       PRIMARY KEY,
    name               text         NOT NULL UNIQUE,
    slug               text         NOT NULL UNIQUE,
    description        text,                                  -- nullable on purpose
    parent_category_id integer      REFERENCES categories (category_id),
    is_active          boolean      NOT NULL DEFAULT true,
    created_at         timestamptz  NOT NULL DEFAULT now()
);

CREATE TABLE customers (
    customer_id      serial        PRIMARY KEY,
    first_name       text          NOT NULL,
    last_name        text          NOT NULL,
    email            text          NOT NULL UNIQUE,
    phone            text,                                    -- nullable on purpose
    country          text          NOT NULL,
    city             text,                                    -- nullable on purpose
    signup_date      timestamptz   NOT NULL DEFAULT now(),
    is_active        boolean       NOT NULL DEFAULT true,
    marketing_opt_in boolean,                                 -- nullable tri-state
    lifetime_value   numeric(12,2) CHECK (lifetime_value IS NULL OR lifetime_value >= 0)
);

CREATE TABLE products (
    product_id      serial        PRIMARY KEY,
    sku             text          NOT NULL UNIQUE,
    name            text          NOT NULL,
    description     text,                                     -- nullable on purpose
    category_id     integer       NOT NULL REFERENCES categories (category_id),
    price           numeric(10,2) NOT NULL CHECK (price >= 0),
    cost            numeric(10,2) CHECK (cost IS NULL OR cost >= 0),
    weight_kg       numeric(6,3),                             -- nullable on purpose
    in_stock        boolean       NOT NULL DEFAULT true,
    stock_quantity  integer       NOT NULL DEFAULT 0 CHECK (stock_quantity >= 0),
    discontinued_at timestamptz,                              -- NULL = still sold
    created_at      timestamptz   NOT NULL DEFAULT now()
);

CREATE TABLE orders (
    order_id         serial        PRIMARY KEY,
    customer_id      integer       NOT NULL REFERENCES customers (customer_id),
    order_date       timestamptz   NOT NULL,
    status           text          NOT NULL CHECK (status IN
                         ('pending','paid','shipped','delivered','cancelled','refunded')),
    total_amount     numeric(12,2) NOT NULL DEFAULT 0 CHECK (total_amount >= 0),
    currency         text          NOT NULL DEFAULT 'USD',
    shipping_address text,                                    -- NULL for digital / cancelled
    shipped_at       timestamptz,                             -- NULL until shipped
    is_gift          boolean       NOT NULL DEFAULT false,
    discount_code    text                                     -- NULL when no promo used
);

CREATE TABLE order_items (
    order_item_id bigserial     PRIMARY KEY,
    order_id      integer       NOT NULL REFERENCES orders (order_id) ON DELETE CASCADE,
    product_id    integer       NOT NULL REFERENCES products (product_id),
    quantity      integer       NOT NULL CHECK (quantity >= 0),  -- 0 allowed: edge case
    unit_price    numeric(10,2) NOT NULL CHECK (unit_price >= 0),
    discount      numeric(10,2) CHECK (discount IS NULL OR discount >= 0),
    is_gift_wrap  boolean       NOT NULL DEFAULT false
);

CREATE TABLE refunds (
    refund_id     serial        PRIMARY KEY,
    order_id      integer       NOT NULL REFERENCES orders (order_id),
    order_item_id bigint        REFERENCES order_items (order_item_id),  -- NULL = whole order
    amount        numeric(12,2) NOT NULL CHECK (amount >= 0),
    reason        text,                                       -- nullable on purpose
    refunded_at   timestamptz   NOT NULL DEFAULT now(),
    is_partial    boolean       NOT NULL DEFAULT false,
    approved_by   text                                        -- NULL = auto-approved
);

-- Foreign-key and common-filter indexes.
CREATE INDEX idx_categories_parent      ON categories  (parent_category_id);
CREATE INDEX idx_products_category      ON products    (category_id);
CREATE INDEX idx_orders_customer        ON orders      (customer_id);
CREATE INDEX idx_orders_order_date      ON orders      (order_date);
CREATE INDEX idx_orders_status          ON orders      (status);
CREATE INDEX idx_order_items_order      ON order_items (order_id);
CREATE INDEX idx_order_items_product    ON order_items (product_id);
CREATE INDEX idx_refunds_order          ON refunds     (order_id);
CREATE INDEX idx_refunds_order_item     ON refunds     (order_item_id);

COMMIT;
