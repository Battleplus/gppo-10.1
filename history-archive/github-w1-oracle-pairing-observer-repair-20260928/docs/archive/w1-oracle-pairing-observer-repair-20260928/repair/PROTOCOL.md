# Oracle Primitive Observer Single-Unit Gate

Status: NOT APPROVED

Proposed attempt: `w1-action-consequence-oracle-observer-single-unit-gate-v1-once`

## Question

Does the repaired observer faithfully record the existing frozen communication primitive calls, distinguish keyed exogenous fate from later execution outcomes, and close branch, label, and accounting evidence for one fixed unit?

This is a technical wiring gate. It does not estimate an oracle ceiling, perform a materiality decision, validate a predictor, or update method rankings.

## Frozen unit

- Parent: `validation-0000`.
- Repeat: `repeat-0`.
- Runtime: packet-identity-repaired W1-light runtime used by the completed repaired fair rerun.
- Window: first public state satisfying the original oracle qualification rule.
- Prefix and continuation: frozen Hungarian controller.
- Candidate enumeration: recomputed from the frozen public state and masks. The seven actions seen in the stopped attempt are not hard coded.
- At most 24 legal non-NOOP assignments and one legal NOOP, for at most 25 branches.
- Native terminal or 18 steps.

All communication, faults, masks, rules, rewards, labels, completion semantics, tie handling, and exogenous-key construction remain unchanged.

## Observer contract

The observer wraps the existing `telemetry`, `command_delivered`, `ack_delivered`, and `renewal` methods. For each existing call it records branch, ordinal, callsite, method, complete keyword arguments including full identity, return value, or raised exception.

The wrapper must call its delegate exactly once, return the identical result object/value, preserve call order, consume no randomness, and make no communication call of its own. An exception is recorded and re-raised.

At branch start, evidence is sliced after the captured prefix. Pairing uses `(method, full identity, complete parameters)` and preserves repeated calls as ordered lists. A call present in only one branch is action-specific. A common call with different returned fate is a technical stop. Parameter conflicts sharing `(method, identity)` are reported and fail the gate. Empty common or action-specific sets cannot pass.

High-level `communication_delta`, delivery, expiration, public-memory acceptance, command acceptance, lease results, labels, and task outcomes remain fully recorded. They are execution outcomes and are not compared as exogenous random fate.

## Execution order

1. Verify the sealed package, runtime hashes, Linux CPU stack, native filesystem, exact external token hash, and absence of prior attempt directories.
2. Execute only `validation-0000/repeat-0`.
3. Recompute the first qualifying window and candidates from frozen rules.
4. Deep-copy the captured state for every legal candidate and record primitive calls after branch start.
5. Verify observer evidence, common fate, action-specific coverage, labels, source-snapshot isolation, and ledger settlement.
6. Settle and export once after worker exit.

Any identity, observer, pairing, public-boundary, isolation, label, ledger, or resource anomaly stops the run. Environment, branch, rule, and communication calls are never retried.

## Acceptance

The gate passes only if:

- the fixed unit reaches an opportunity under the frozen selector;
- at least two branches complete with full labels;
- common primitive calls and action-specific primitive calls are both nonempty;
- all common primitive return values match;
- no `(method, identity)` parameter conflict exists;
- the source snapshot remains unchanged;
- the ledger has no pending or unknown calls and all resource caps hold.

A pass authorizes nothing else. The remaining 23 units are not run. Whether the gate data can later be reused and how it counts toward a full oracle run must be frozen in a separate authorization before any continuation.

## History retained

The stopped oracle attempt remains a technical stop with 90 environment steps, one reset, 84 public rule decisions, and seven branches. Its 149 reported mismatches were explained as 135 downstream-stage comparisons plus 14 wrong command/renewal/ACK groupings, but missing command fields were not reconstructed and the gate was not retroactively passed.

The completed repaired fair rerun remains valid and negative under its frozen practical criteria. All prior technical failures, costs, unknown calls, and the 10 ms CPU cost failure remain cumulative and receive no budget credit.

## Authorization boundary

The resource request is `NOT_APPROVED`. Preparation, tests, and GitHub archival do not authorize creation of the proposed attempt, environment reset/step, model use, training, or the remaining oracle matrix.
