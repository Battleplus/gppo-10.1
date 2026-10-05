# Label-history time-boundary root cause

## Evidence scope

This diagnosis uses the sealed attempt
`w1-action-outcome-learning-loop-launch-repair-v1-once`, its SQLite ledger and
worker traceback, the frozen launch-repair source, and the frozen public
controller source. No hidden environment state was reconstructed and no real
environment was instantiated during this analysis.

## First failing chain

The first unit was `train-1534 / repeat-0`. The ledger records one reset, seven
Hungarian prefix decisions, a snapshot, one candidate branch, four Hungarian
continuation decisions, and twelve environment steps before the worker stopped.
No label row was persisted.

1. `AuthorizedRuntimeBackend._collect_unit()` calls the public adapter's
   `prepare(observation)` once per prefix step. The adapter takes a public-only
   copy and `PublicMemory.observe()` appends a deep copy to `memory.history`.
2. A decision window is fixed by the first step at or after step 4 with at
   least two non-NOOP candidates and nonidentical transparent scores. The
   decision time is `float(public_obs["time"])` from that prepared public
   observation.
3. The old collector captured `(env, adapter, public_obs, ...)` and deep-copied
   the whole capture for each candidate. This isolated candidate branches from
   the parent and from one another; existing evidence does not show shared
   mutable branch state.
4. During Hungarian continuation, the copied adapter repeatedly calls
   `prepare()`, which correctly appends later public observations to that
   branch's memory.
5. The defect occurred after continuation: the old `_run_label_branch()` read
   `adapter.memory.history[:-1]` from the already-advanced branch, then called
   `history_features(..., public_obs["time"])` using the captured decision
   time. Removing only the last history row did not remove all continuation
   observations. With four continuation decisions recorded in the ledger,
   later observations remained and triggered the frozen future-observation
   check.

The failing comparison was between each public observation's top-level `time`
and the decision observation's top-level `time`. It was not comparing telemetry
measurement time or message receipt time. Field measurement time remains
represented by `measured_at = observation time - age` inside public memory.

## Root cause classification

The first deviation was **feature construction after branch execution using
branch-mutated public history**. The future-time check behaved correctly. The
evidence does not support weakening that check, changing timestamps, or deleting
future rows after the fact.

The sealed attempt proves the exception and dynamic call sequence. It did not
persist the exact offending history rows, so their individual contents and
receipt events are unknown. The source path and four recorded continuation
decisions are sufficient to explain how more than one post-decision row could
remain after `history[:-1]`.

## Repair

The repaired collector constructs an immutable `DecisionInputSnapshot` before
the first candidate branch. It contains decision time, current 770-value public
flat input, history features computed from pre-decision history, legal actions,
and both frozen baseline score maps. A canonical audit payload records the
current public observation and prior public history and supplies a SHA-256
identity.

Each branch still deep-copies the environment and adapter for label generation,
but reads its model input only from the immutable snapshot. Branch observations
remain available to the labeler and communication audit and cannot flow back
into features. Candidate action encoding remains separate in
`materialize_features()`.

## Historical preservation

The old attempt, its 12 environment steps, 1 reset, 1 branch, 11 public rule
decisions, 36 settled calls, native evidence, and verified export remain
unchanged. This repair does not retrospectively turn that stopped branch into a
label.
