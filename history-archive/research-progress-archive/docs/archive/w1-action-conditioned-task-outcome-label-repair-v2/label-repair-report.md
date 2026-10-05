# W1 Action-Conditioned Task Outcome Label Repair

## Finding

The immediate defect was a mismatch between the EAWM collector's actual
effective configuration and a reused source-run `environment.json`, plus a
one-step branch export that ended before task lifecycle outcomes could be
observed. The collector used continuous service under `M10Config` defaults;
the source-run environment file described arrival-to-region completion. That
file was valid for the fair-rerun that loaded it, but was not the effective
configuration of the default-constructed EAWM collector.

The new contract names `continuous_service_until_deadline / physical_service`
explicitly, verifies the complete config before environment construction, and
records the exact runtime config object. It extends each candidate's label-side
trajectory under frozen Hungarian continuation through native termination or
the 18-second horizon. It extracts physical completion from
`TaskLifecycle.completed_at` and expiry from explicit `TaskLifecycle.state`.
Host confirmation remains unavailable and masked.

## Label ownership

The label belongs to the legal first candidate plus the complete frozen
Hungarian continuation. It does not assert the first command directly caused
the terminal result. First-command status, command ID, target task/UAV, any
initial execution identity, later completion execution identity, public input
hash, deadline, configuration hash, and continuation ID remain separately
auditable. A later reassignment or execution already in progress can yield a
valid sequence result while leaving first-action-specific causal identity
false or unknown.

Decision-time public input is frozen before branches. Lifecycle records and
future telemetry appear only in the label-side trajectory and audit fields.
NOOP task heads are masked; missing lifecycle records, early endings and
unobserved deadline outcomes remain unknown.

## Reuse decision

The 219/68/111 train/model-selection/prediction records from the 88-window
historical attempt are retained as development evidence but not admitted to
the v3 labels. They use arrival-to-region semantics and omit an auditable
candidate-to-internal-task identity and full task lifecycle trajectory. The
144 oracle branches are likewise arrival-contract evidence. The separate
56-window/771-candidate EAWM integration export is synthetic, contains no
terminal task lifecycle, and contributes zero real labels. See
`data-reuse-audit.json` and `configuration-identity-audit.md`.

## Head decision and scope

Keep physical on-time completion and explicit expiry as separately masked
sequence-outcome targets. Keep host confirmation disabled under the frozen
continuous-service mode. Keep completion-by-first-command audit-only. This
proves that production code can construct and persist labels from a controlled
lifecycle fixture; real-world label coverage is not yet measured. The attached
request for eight parent scenarios is `NOT_APPROVED`, label-only, and cannot
be interpreted as training or prediction evaluation.

The old `prediction_gate_stop` and all prior utility, oracle, cost, and
technical-stop evidence remain unchanged. This preparation does not establish
whether correctly supervised task outcomes are learnable or useful.
