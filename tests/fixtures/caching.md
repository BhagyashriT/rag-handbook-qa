# Caching strategies

## The four patterns

Cache-aside, read-through, write-through, and write-behind. Cache-aside is
the most common: the application checks the cache first and falls back to
the database, then populates the cache on a miss.

## Thundering herd

When a hot key expires, many requests can stampede the database. Mitigate
with probabilistic early refresh or a single-flight lock per key.
