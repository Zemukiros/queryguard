-- Read-only role used by QueryGuard to execute LLM-generated SQL.
-- SELECT only: no INSERT/UPDATE/DELETE/TRUNCATE, no CREATE, now or in the future.

-- Password comes from the POSTGRES_RO_PASSWORD environment variable so that no
-- credential is committed to the repository. (psql \getenv, PostgreSQL 16+.)
\getenv ro_password POSTGRES_RO_PASSWORD
\if :{?ro_password}
\else
\echo 'FATAL: POSTGRES_RO_PASSWORD is not set'
\quit 1
\endif

CREATE ROLE queryguard_ro LOGIN PASSWORD :'ro_password';

-- Database level: connect, nothing else. (Also strips CREATE inherited via PUBLIC.)
REVOKE ALL ON DATABASE :"DBNAME" FROM queryguard_ro;
REVOKE CREATE ON DATABASE :"DBNAME" FROM PUBLIC;
GRANT CONNECT ON DATABASE :"DBNAME" TO queryguard_ro;

-- Schema level: look, don't create. PUBLIC's implicit CREATE on public is removed
-- as well, otherwise every role could still make tables here.
REVOKE ALL ON SCHEMA public FROM queryguard_ro;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO queryguard_ro;

-- Table level: SELECT and only SELECT on everything that exists today.
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM queryguard_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO queryguard_ro;

-- Sequences: SELECT lets the role read currval/last_value; USAGE/UPDATE would
-- let it advance them, so those stay revoked.
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM queryguard_ro;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO queryguard_ro;

-- No EXECUTE on functions/procedures by default.
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA public FROM queryguard_ro;
REVOKE ALL ON ALL PROCEDURES IN SCHEMA public FROM queryguard_ro;

-- Future objects created by the owner role land read-only too. Default privileges
-- are per-creating-role, so they are pinned to the connected owner role (:USER).
ALTER DEFAULT PRIVILEGES FOR ROLE :"USER" IN SCHEMA public
    GRANT SELECT ON TABLES TO queryguard_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE :"USER" IN SCHEMA public
    REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON TABLES FROM queryguard_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE :"USER" IN SCHEMA public
    GRANT SELECT ON SEQUENCES TO queryguard_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE :"USER" IN SCHEMA public
    REVOKE USAGE, UPDATE ON SEQUENCES FROM queryguard_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE :"USER" IN SCHEMA public
    REVOKE EXECUTE ON FUNCTIONS FROM queryguard_ro;

-- The role can neither create databases/roles nor bypass RLS.
ALTER ROLE queryguard_ro NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;

-- Belt and braces: every session this role opens starts read-only, so even a
-- privilege misconfiguration cannot result in a write.
ALTER ROLE queryguard_ro SET default_transaction_read_only = on;
