-- Atomic check-and-reserve against the daily spend ceiling.
--
-- A question's cost is only known after its LLM calls finish, so while it runs
-- it holds a reservation worth ARGV[2] (more than any question has cost). A
-- new question is admitted only if
--
--     spent today + (live reservations + 1) x reserve  <=  ceiling
--
-- Run as one script, the read and the write cannot interleave with another
-- instance's: two instances can never both take the last slot. (With GET and
-- ZADD as separate commands, both could read "one slot left" and both add.)
--
-- KEYS[1]  today's spend in USD, a float stored as a string (spend:<UTC date>)
-- KEYS[2]  sorted set of live reservations: member = reservation id,
--          score = the time it expires, in ms
-- ARGV[1]  ceiling in USD   ARGV[2]  reserve per question in USD
-- ARGV[3]  reservation lifetime in ms   ARGV[4]  this reservation's id
--
-- Returns {1, spent} if reserved, {0, spent} if refused. Spent goes back as a
-- string: Redis would truncate a Lua number to an integer (0.42 -> 0).

local ceiling = tonumber(ARGV[1])
local reserve = tonumber(ARGV[2])
local ttl_ms = tonumber(ARGV[3])

local t = redis.call('TIME')                               -- Redis's clock: every instance
local now = t[1] * 1000 + math.floor(t[2] / 1000)          -- agrees on what "expired" means

-- A reservation whose holder crashed is never released. Its expiry time has
-- passed, so it stops counting here: a dead instance cannot pin the budget.
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now)

local spent = tonumber(redis.call('GET', KEYS[1]) or '0')  -- GET returns false when missing
local held = redis.call('ZCARD', KEYS[2])                  -- live reservations right now

if spent + (held + 1) * reserve > ceiling then             -- this one would not fit
  return {0, tostring(spent)}
end

redis.call('ZADD', KEYS[2], now + ttl_ms, ARGV[4])         -- take the slot, expiring at now + ttl
redis.call('PEXPIRE', KEYS[2], ttl_ms)                     -- the set itself goes once all expire
return {1, tostring(spent)}
