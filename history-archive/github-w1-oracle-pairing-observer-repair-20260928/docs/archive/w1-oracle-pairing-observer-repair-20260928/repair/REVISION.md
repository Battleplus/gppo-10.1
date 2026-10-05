# Revision: Primitive-call pairing observer

## Confirmed failure

The stopped runner grouped communication events by `link + identity/message_id/command_id + ordinal + attempt` and compared complete high-level event sequences. This merged distinct allocation and lease-renewal calls and treated branch-dependent receipt records as keyed random fate.

The offline audit reproduced all counts: 983 common coarse identities and 149 mismatches, consisting of 135 telemetry downstream-stage differences and 14 command/ACK grouping collisions. No true telemetry primitive-fate mismatch was found. Allocation command identities lacked enough saved fields for retrospective validation.

## Minimal change

- Add a transparent wrapper at the actual frozen communication primitive boundary.
- Save exact primitive input and output once per existing call.
- Compare only common exact primitive calls.
- Keep execution outcomes outside the random-fate equality gate.
- Restrict the prepared runner to one fixed dynamic technical unit.

No shared runtime behavior or research mechanism changes.

## Historical impact

- Old seven oracle branches: remain technically stopped and unusable for ceiling analysis.
- Two-parent packet-identity validation: unchanged; it already compared exact primitive identities and results.
- Completed repaired fair rerun: unchanged; the defect is in the oracle-specific verifier.
- Historical costs and the 10 ms failure: retained.
