-- The shared daily cap on LLM calls: a hard backstop under the spend ceiling,
-- counted across every instance. Runs before each call, so a refused call is
-- never made.
--
-- KEYS[1]  today's call count (calls:<UTC date>)
-- ARGV[1]  the cap
--
-- Returns 1 if the call may go ahead, 0 if today's cap is reached.

local n = redis.call('INCR', KEYS[1])        -- count this call; INCR is atomic, and a missing
                                             -- key starts at 0, so the first call of the day gets 1
if n == 1 then
  redis.call('EXPIRE', KEYS[1], 172800)      -- first call today: let the counter die in 48 h
end
if n > tonumber(ARGV[1]) then
  redis.call('DECR', KEYS[1])                -- over the cap: undo the count, so the stored
  return 0                                   -- number stays "calls made", not "calls tried"
end
return 1
