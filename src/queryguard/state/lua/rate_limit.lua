-- Sliding-window rate limit for one client: at most ARGV[1] questions in any
-- 60 seconds and ARGV[2] in any 24 hours.
--
-- Why a Lua script: Redis runs a script start to finish with no other command
-- in between. "Count the recent hits, then add this one" as two separate
-- commands would let two instances both see 9 hits and both add a 10th --
-- that race is exactly what a per-client limit across instances must not have.
--
-- KEYS[1]  this client's sorted set: one member per admitted question,
--          scored by its time in milliseconds
-- ARGV[1]  per-minute limit      ARGV[2]  per-day limit
-- ARGV[3]  a unique id for this request (a sorted set needs distinct members)
--
-- Returns -1 if admitted, otherwise the milliseconds until a slot frees up.

local key = KEYS[1]                                    -- the client's hit log
local per_minute = tonumber(ARGV[1])                   -- arguments arrive as strings
local per_day = tonumber(ARGV[2])
local minute_ms = 60000
local day_ms = 86400000

local t = redis.call('TIME')                           -- Redis's clock, not the caller's:
local now = t[1] * 1000 + math.floor(t[2] / 1000)      -- {seconds, microseconds} -> ms

redis.call('ZREMRANGEBYSCORE', key, '-inf', now - day_ms)   -- forget hits older than 24 h

-- Hits strictly inside the last minute: '(' makes the lower bound exclusive.
local in_minute = redis.call('ZCOUNT', key, '(' .. (now - minute_ms), '+inf')
if in_minute >= per_minute then
  -- The oldest hit inside the window decides when the window has room again.
  local oldest = redis.call('ZRANGEBYSCORE', key, '(' .. (now - minute_ms), '+inf', 'WITHSCORES', 'LIMIT', 0, 1)
  return minute_ms - (now - tonumber(oldest[2]))       -- oldest = {member, score}
end

local in_day = redis.call('ZCARD', key)                 -- everything left is within 24 h
if in_day >= per_day then
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')   -- lowest score = oldest hit
  return day_ms - (now - tonumber(oldest[2]))
end

redis.call('ZADD', key, now, ARGV[3])                   -- admit: record this hit
redis.call('PEXPIRE', key, day_ms)                      -- an idle client's log deletes itself
return -1
