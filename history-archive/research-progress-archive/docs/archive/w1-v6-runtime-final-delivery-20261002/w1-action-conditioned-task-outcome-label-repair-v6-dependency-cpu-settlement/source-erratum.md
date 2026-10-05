# Source Erratum: Prior EAWM/Jepa Export

## Finding

The v2 reuse audit's `TEST_FIXTURE_ONLY` label for the 56-window, 771-candidate
export is contradicted by the frozen launch package and its exported run
records. The export was produced through the formal `W1RuntimeBackend` and
`ProductionDataCollector` path with `integration_test=false`; its run summary
reports verified source-tape identity. The evidence supports generated W1 M10
simulator data collected by the production software path. It does not support
calling this field UAV telemetry. The specific origin of the v2
misclassification remains undetermined.

The exported window rows have a null `environment_config_identity`, and the
attempt output has no runtime `environment.json`. The effective mode for that
collector is therefore derived from the frozen source path, not claimed as a
separately persisted runtime record: the formal worker invokes
`W1RuntimeBackend`, which invokes `ProductionDataCollector`; its frozen
`production_data.py` constructs bare `M10Config()` and the pinned
`m10_environment.py` defaults to
`continuous_service_until_deadline / physical_service`. The source tape came
from the fair-rerun whose separate `environment.json` selected
`arrival_to_region / physical_arrival`; that file is not evidence that the
formal collector inherited those settings.

The export's arithmetic and procedural stop state are preserved. Its one-step
utility/event/state metrics do not establish prediction of terminal physical
task completion, expiry, host confirmation, or real-world UAV outcomes. The
run had zero branches with a terminal or truncated task-lifecycle record,
zero valid physical-completion labels, and zero valid host-confirmation
labels. `prediction_gate_stop` remains the historical process status, while
the scientific interpretation of these metrics as evidence for real W1 task
outcome prediction is withdrawn. No task comparison was executed.

## Identity Evidence

| Item | Recorded identity |
| --- | --- |
| Formal package attempt | `w1-eawm-jepa-world-model-production-repair-v4-e2e-integration-auth-repair-v1-once` |
| Frozen execution manifest SHA-256 | `db17669cbeb472415b550015d5272f04574e7daeebb923a6f39f001613e39c6a` |
| Frozen hashes SHA-256 | `24da3ecd62baea3340d38c9b27db599fb074d275beda952c405202277761253b` |
| Frozen resource request SHA-256 | `69c9252598567832cd07640f3420a9ff07e523648fce8aa3fcd9db73119b9ce5` |
| Controlled export manifest SHA-256 | `2c78ad80e4a90e5578b8deb15bc964f086f1ba53468f04d06693ce4243f61d05` |
| Controlled export completion marker SHA-256 | `b8f69683fd11f6b34f5c0adfa00ed29308ec43b6e6083903829982c52c0554a9` |
| Source run manifest SHA-256 | `961016499ba5229112c116313ff80a87305eda48342c869212b696f280833a12` |
| Source train-tape SHA-256 | `f2e6f7b2d7590989e305be5d7c6faf685369e15563a7b2fefb18c0468e7aa510` |
| Fair-rerun `environment.json` | `runs/w1-light-repaired-fair-rerun-v2-nativefs-once/environment.json`, SHA-256 `95b8b6e293fd7f500c3e89114424ff850717914cd3950a01e5cfbfb426f625ae`, `arrival_to_region / physical_arrival` |
| Pinned old collector source | `production_data.py` calls `M10Config()` with native defaults; `environment_config_identity` is null in exported rows |
| Pinned old environment source | `runs/w1-light-repaired-fair-rerun-v2-nativefs-once/native/gppo_world/m10_environment.py`, SHA-256 `0f4615d11604a7cd1cc74c94f0258dfc6cb59748f64a075276ec90941db094b1` |
| Frozen formal collector | `runner.py` dispatches `W1RuntimeBackend`; frozen `runtime_backend.py` calls the production collector (source hashes below) |
| Exported windows | `run-once/world-model-windows.jsonl`, 110,433,034 bytes, SHA-256 `932b9f6638dbe332359301b26b098db411d5f75cee93ec239114fb0ac421bb51` |
| Recorded window/candidate split | 56 windows / 771 candidate rows; 24 train, 8 model-selection, 24 prediction-confirmation windows |
| Run status | `prediction_gate_stop`; policy training false; task comparison false |

The separate E2E fake-worker fixture has a different path, size, and digest;
it is not the 110,433,034-byte windows export. The formal package identity and
the source export identity are distinct: the package manifest freezes
pre-run code/configuration, while the controlled export manifest binds the
post-run records.

## Affected Outputs and Claims

The files below are retained without alteration. Their metrics remain
recomputable from their traces, but are limited to generated-simulator,
one-step development evidence. They cannot support a claim about terminal
task outcomes or field-UAV performance.

| Exported output | SHA-256 | Interpretation affected |
| --- | --- | --- |
| `run-once/world-model-training-summary.json` (38,275 bytes) | `d29b185434300f754f4bb5fd95ce23b03a917770edd8fb919ea665dddce6fd8d` | G1/G2 fit and model-selection history on generated W1 M10 simulator windows |
| `run-once/prediction-trace.jsonl` (11,407,408 bytes) | `0fe0c9e1d99272338a67637c93615d44ca4e6c477c55655cb0f6c96967b991db` | Per-candidate predictions, labels, selected actions and one-step continuation identity |
| `run-once/prediction-metrics.json` (20,920 bytes) | `4b68f201abfc690a599afe77290a5c0625501f0e6c4d416b88e2d9285c9dd1bc` | Outcome MAE/RMSE, public-state MAE/RMSE, event Brier/ECE/NLL, pairwise direction accuracy, regret and Top-1 |
| `run-once/status.json` (165 bytes) | `773b02e52f21f6a96ab1bc4294a54b96abed23a0bddcc32f2aa13bbddf0756b0` | Procedural `prediction_gate_stop`; task comparison not run |

Six saved G1/G2 checkpoints are also retained as historical artifacts. No
checkpoint was loaded for this erratum:

| Checkpoint | SHA-256 |
| --- | --- |
| `world-model-checkpoints/G1/seed-8201.pt` | `453aba3f68ae829f98559f649b5321557863eaf13ecd1feeb10dd0eabf4723f2` |
| `world-model-checkpoints/G1/seed-8202.pt` | `bbc0a29a59a1223c58d8303d648c26360a7f24b6595b1a9892afe9d07ec9eb4c` |
| `world-model-checkpoints/G1/seed-8203.pt` | `3b3e9487098a46e1eb8a3d07724fe54355714675d45c0cf581c35879af2fec83` |
| `world-model-checkpoints/G2/seed-8201.pt` | `883120bfb9bfe35db7bb5c81622215ae8759985d72e4f3ab3c6da97ab9901ce9` |
| `world-model-checkpoints/G2/seed-8202.pt` | `07a0a4125c21db1ecbc9c793ba0678b02c66d13564a8c5525542ce6c74c14016` |
| `world-model-checkpoints/G2/seed-8203.pt` | `c1c6ec3cd4c8134bfcb8183b062df2f74fe5cdb52e3747db9bc11264c53a0259` |

The event Brier/ECE/NLL values concern the saved one-step event targets. They
are not physical-arrival or host-confirmation metrics. Utility MAE/RMSE,
pairwise accuracy, regret, and Top-1 use the saved one-step utility target and
candidate set; they do not estimate the terminal task outcome under a fixed
continuation. Public-state errors remain simulator transition diagnostics.

The related earlier files remain unchanged and need this scope when cited:

| Existing file | SHA-256 | Correction scope |
| --- | --- | --- |
| `research-plans/w1-action-conditioned-task-outcome-label-repair-v2/data-reuse-audit.json` (4,240 bytes) | `d984afc5fef93ca010321cc741cb2ded61decd41183a9e5acae65ad3685650a1` | Its `TEST_FIXTURE_ONLY` source class is superseded by the provenance evidence above; the reason for its misclassification remains unknown |
| `research-plans/w1-action-conditioned-task-outcome-label-repair-v1/data-availability-audit.json` (4,783 bytes) | `e9030c9ed2a4e1a23a93ee2b2a8cd0c2b677c6e6c3d973f69fe5776d59e633d7` | The missing lifecycle counts and no-reconstruction finding remain valid; the producer class is production collector on generated simulator scenarios |
| `research-plans/w1-action-conditioned-task-outcome-label-repair-v1/report.md` (2,669 bytes) | `115b571f8134564b9d0d79fc4035765a2e77e81b1fda4cb342742e2852cdc442` | The 19/24 NOOP and zero one-step regret calculation remains narrowly valid for the saved utility labels; it is not evidence about terminal task consequences |
| `research-plans/w1-eawm-jepa-world-model-production-repair-v4-e2e-integration-auth-repair-v1/report.md` (5,569 bytes) | `b2d6f4b14bd4a4a1b1638aaabe0cb784e06fae8af33fcd710d264d9ac8799bf3` | Engineering E2E fixture evidence is separate from this formal export and is not a model-effect report |

The old frozen package pins `runner.py` SHA-256
`0d6eef1ef7f757aec7693d5c1182fcdc98da9d825028fa878088f171f2ec6547`,
`runtime_backend.py` SHA-256
`9489b5091493737d2d96473571fd0d4a189d8a7cbdc58f5298ffdd61717e85b9`, and
`production_data.py` SHA-256
`371cead307b345d250ed0a294aec0204d55761725158360d7e7f66742b7d9c4c`.
Those identities tie the dispatch path to the collector source used by the
formal attempt. They do not substitute for the missing runtime configuration
record.

## Task Semantics

The historical EAWM collector constructed the native default
`continuous_service_until_deadline / physical_service`; the fair-rerun's
effective `environment.json` explicitly selects
`arrival_to_region / physical_arrival`. These task contracts are not pooled or
compared as if interchangeable. This label-qualification package freezes
`arrival_to_region / physical_arrival`, based on the verified fair-rerun
configuration and the task meaning that reaching the target region constitutes
completion. It records the exact instantiated environment configuration and
rejects any mismatch before reset. It does not claim that historical
continuous-service results are arrival-contract results.

The eight proposed label-qualification parents are fixed as `train-0056`
through `train-0063` from the verified train tape, before outcomes are
observed. They have no overlap with the inspected registered split matrix;
available records do not establish that these parents were unused by every
historical policy.

This erratum is not a retrospective relabeling, model re-evaluation, or change
to any historical gate. The new resource request remains `NOT_APPROVED` and
this package has not started its real collector.

## v4 Preparation Corrections

The independent v4 preparation copy resolves three execution-contract issues
without changing the task data, parent split, label scope, or dynamic-call
limits. In the frozen arrival mode, `TaskLifecycle.arrive()` calls `advance()`
first; `advance(now >= deadline)` expires the task, so a physical arrival at
exact equality is not accepted. The executable label validator already rejects
`completed_at >= deadline`; v4 corrects the prose contract and adds an exact
boundary regression test. This is not a historical relabeling.

`arrival_radius=0.0` is retained from the verified fair-rerun configuration.
Because the simulator uses a distance `<= radius` predicate, this means an
exact point-target objective; it does not model an area around a target.

Environment construction is now recorded as paired attempted/constructed
events around the actual production constructor call. The final runner status
is derived from those events, never from a reserved reset budget. A dangling
attempt or malformed evidence is reported as indeterminate/unavailable.

The v4 Windows entry measures its process CPU and full elapsed wall time across
the WSL call, reads the controlled export's Linux-side accounting, adds the
Windows post-preflight CPU and verifies both named reserves. The Linux export
accounting now includes the native-launcher child CPU. The global ceilings
remain 753 seconds wall and 551 seconds complete-process CPU. The unchanged
ceilings are reallocated to stage limits of 20/693/38 seconds plus 2 seconds
wall reserve, and 20/491/38 seconds plus 1 second each for native-launcher and
Windows post-preflight CPU. If the measured cross-system terms exceed those
reserves, the launcher returns a technical stop; it does not retry.
