# Concurrency notes

## Deadlock checklist

A deadlock needs four conditions: mutual exclusion, hold and wait, no
preemption, and circular wait. Break any one of them: acquire locks in a
global order, use tryLock with timeouts, and keep critical sections tiny.

## Thread pools and backpressure

Size pools with Little's law in mind. Prefer a bounded queue plus a
CallerRunsPolicy so producers feel backpressure instead of OOMing.
