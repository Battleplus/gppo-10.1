# W1 Action-Conditioned Task Outcome Label Repair

## Finding

The source-provenance review found that the v2 reuse audit's `TEST_FIXTURE_ONLY`
classification for the 56-window/771-candidate export conflicts with frozen
formal-entry and run records. The formal package has `integration_test=false`,
uses `W1RuntimeBackend` and `ProductionDataCollector`, and verified its source
tape. The export is production collector output from generated W1 M10
simulator scenarios, not field UAV telemetry. The E2E fake-worker fixture is a
different, much smaller artifact with a different hash. Why the older audit
mislabeled the formal export is not established.

The old EAWM collector used the native default
`continuous_service_until_deadline / physical_service`; fair-rerun loaded an
explicit `arrival_to_region / physical_arrival` configuration. The finite
label contract uses arrival semantics because that is the verified effective
fair-rerun configuration and matches the stated meaning that arrival at the
target region completes the task. These are different task contracts;
historical rows are not pooled.

The new contract names `arrival_to_region / physical_arrival` explicitly,
verifies the complete config before environment construction, and
records the exact runtime config object. It extends each candidate's label-side
trajectory under frozen Hungarian continuation through native termination or
the 18-second horizon. It extracts physical completion from
`TaskLifecycle.completed_at` and expiry from explicit `TaskLifecycle.state`.
Host confirmation is separate from physical arrival: an observed notice
receipt by deadline is true, explicit expiry makes it false, and missing
receipt without terminal expiry remains unknown.

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
the v4 labels. They use arrival-to-region semantics and omit an auditable
candidate-to-internal-task identity and full task lifecycle trajectory. The
144 oracle branches are likewise arrival-contract evidence. The 56-window/
771-candidate formal export has no usable terminal task lifecycle supervision
for v4 and contributes zero rows to the qualification set. It is not the
E2E fake-worker artifact. See
`data-reuse-audit.json` and `configuration-identity-audit.md`.

## Head decision and scope

Keep physical on-time completion and explicit expiry as separately masked
sequence-outcome targets. Keep host confirmation separate from physical
arrival; use actual notice receipt or explicit expiry and preserve unknown.
Keep completion-by-first-command audit-only. Controlled lifecycle fixtures
exercise production code; real label coverage is not yet measured. The attached
request for eight parent scenarios is `NOT_APPROVED`, label-only, and cannot
be interpreted as training or prediction evaluation.

The old `prediction_gate_stop` and all prior utility, oracle, cost, and
technical-stop evidence remain unchanged. This preparation does not establish
whether correctly supervised task outcomes are learnable or useful.
