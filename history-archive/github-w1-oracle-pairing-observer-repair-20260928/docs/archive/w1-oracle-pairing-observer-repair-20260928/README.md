# W1 Oracle Pairing Observer Repair Archive

Date: 2026-09-28

This archive records the stopped oracle run, the offline root-cause audit, the zero-call observer repair, and the completed single-unit dynamic observer gate.

## Evidence status

- The stopped oracle run did not answer whether accurate action-consequence prediction is useful. Its seven branches remain a technical stop.
- The audit reproduced 983 coarse common identities and 149 mismatches. All 149 were explained: 135 telemetry cases compared branch-dependent downstream receipt records, while 8 command and 6 ACK cases were merged by coarse `command_id` grouping.
- Saved telemetry primitive fate was consistent. Missing command/ACK primitive addresses were not reconstructed, so the stopped gate was not retroactively passed.
- The completed repaired-runtime fair rerun is unaffected. The defect was in the oracle-specific verifier, not in the shared environment or communication primitive.

## Repair

`repair/communication_observer.py` wraps the frozen communication primitive at its real call boundary. It records branch, ordinal, source callsite, method, complete identity and parameters, and the primitive return value or exception. It delegates exactly once and preserves returned objects and exceptions.

The repaired verifier compares only exact common primitive calls. Branch-specific calls remain separate, duplicate calls retain multiplicity, parameter conflicts are explicit, and downstream delivery, expiration, acceptance, and lease results are not treated as random fate.

`repair/runner.py` is restricted to `validation-0000 / repeat-0`. It recomputes the qualifying window and actions from the frozen public state, performs no oracle materiality analysis, and cannot continue to the other 23 units.

## Verification

On WSL native ext4 with Python 3.11.16 and the frozen runtime dependencies, 34 zero-call tests passed. The final package manifest verified all 38 files byte-for-byte.

- Execution manifest SHA-256: `99643974f96681ebd1d70a24bfd1190218caf7868a08bb80fc5fdb38e98b9353`
- Package hashes SHA-256: `694e7fefabf7ac1139741b4cb742d520ac7ce6a5540ad3872b366a39e443bed4`
- Root-cause audit hashes SHA-256: `24a12e8cf7582258e85d4a1c8764588450897754e7f505df0e2ba0f941eef911`

## Dynamic observer gate

The authorized one-shot gate ran only `validation-0000 / repeat-0`. The frozen rules recomputed seven branches (six non-NOOP assignments and one legal NOOP); no old action list was hard-coded. It did not run the other 23 units and did not calculate oracle materiality.

The observer recorded 6,157 calls at the real communication primitive boundary: 5,778 telemetry, 89 command, 166 ACK, and 124 renewal calls. Exact semantic matching found 1,006 common call keys and 70 branch-specific keys. There were zero random-fate mismatches, parameter conflicts, multiplicity conflicts, or missing identities. This establishes that the repaired observer is connected to actual primitive calls and distinguishes exogenous fate from downstream execution outcomes for this unit.

All seven branches terminated normally. Seven `task-5` records remain explicitly `unknown` for host confirmation because native termination occurred after physical completion and before confirmation arrived. These labels were not filled or dropped and limit any future reuse of the unit.

The gate used 90 environment steps, one reset, 84 public-rule decisions, and seven branches. Model initialization, model forward calls, and training updates were all zero. The verified export contains 54 files and 26,389,006 bytes; its manifest SHA-256 is `c1124832a421c46283e9f45ed2c53c0569b53b552c5358f5aa606464e6d1f86b`.

- Dynamic gate report hashes SHA-256: `403dd7d0eb62a84c4505eb390e9f8f4825d84ddae8bf9013e3322b78f0f56f60`
- Native export status SHA-256: `9f27f06e83dba7902e7dfd7cf8312cbf427fc10c40ae6d22a58af97a07bec761`

## Remaining 23-unit preparation

The completed gate unit is now frozen for one-sample reuse in a future 24-unit joint analysis. A zero-call audit independently reconstructed all seven branch utilities from the complete saved step sequence. Every saved scalar environment reward is finite, every branch ended at native termination without truncation, and the maximum utility recomputation error is `1.1102230246251565e-16`. The seven unknown host-confirmation labels remain unknown and do not enter the frozen utility formula.

The prepared runner explicitly excludes `validation-0000 / repeat-0` and enumerates exactly the remaining 23 units. Joint analysis requires exactly one reused unit plus 23 new units; duplicates, missing units, incomplete labels, or reuse identity differences stop the run. Forty-one zero-call tests passed on the frozen WSL Python 3.11.16 stack, including a fail-before-environment-construction test for damaged reuse evidence.

The independent request remains `NOT_APPROVED`: at most 10,350 environment steps, 23 resets, 9,798 public-rule decisions, and 575 branches, with all model and training counters zero. Preparation does not authorize an attempt or training.

- Remaining-23 execution manifest SHA-256: `2bf84e49d151c0fa3be4a723df97eb5b41f1571bbd88def0c04c196c2e401843`
- Remaining-23 package hashes SHA-256: `d0e0388174dce35787b18cc377e5455f6fc8874673b73b5e79a973147fb97ac1`

## Completed 24-unit oracle result

The separately authorized one-shot attempt executed only the remaining 23 units and reused `validation-0000/repeat-0` exactly once. The resulting matrix is complete: 24 of 24 units and 8 of 8 development parents had a qualifying opportunity. The run completed normally with no retry, worker relaunch, model call, or training update.

The frozen materiality gate failed. The eight-parent macro mean hindsight gain was `0.013226693424422117`, but only `validation-0002` and `validation-0007` passed the parent gate. The protocol required at least 4 of 8 passing parents. A macro mean above `0.01` does not override that breadth requirement.

The combined matrix contains 144 candidate branches and 864 candidate-branch task labels. There are 182 unknown host-confirmation labels. They remain unknown, are not physical failures or accounting unknowns, and do not enter the frozen utility calculation.

This closes only the first qualifying public window with frozen Hungarian continuation on these development parents. It does not establish that every later window or every action-consequence target lacks value. The hindsight maximum may include unpredictable random luck and is not predictor performance. The result does not authorize predictor training or another experiment.

The new attempt consumed 1,598 environment steps, 23 resets, 1,484 public-rule decisions, and 137 branches. All 3,586 accounted calls were verified. The controlled export listed 68 files, and an independent post-run check found zero missing or mismatched files.

- Result reports and selected evidence: `remaining-23-result/`
- Export manifest SHA-256: `efa772e6b26f1c764cd7f4c9063fa0ce352722220eb7f8061efba929310779b1`
- Post-run report hashes SHA-256: `3401fbc6d4c5dc0cac14f1c6a2d3e34ab3d8845dd2e9424d471ea424c5f31736`

## Authorization boundary

The single-unit gate and remaining-23 requests were separately authorized and have both been consumed by their completed one-shot attempts. This archive does not authorize a retry, model use, training, expanded sampling, threshold changes, or a new experiment.

## Archive scope

This Git archive contains the root-cause reports, evidence indexes, minimal observer and runner changes, zero-call tests, frozen protocol and budget, dynamic gate summaries, selected small execution evidence, and final hashes. It is not a full backup or deployable runtime. Large raw step and branch logs, SQLite ledgers, checkpoints, credentials, and the raw one-shot authorization token are not included. Their local paths, sizes, and hashes are listed in `artifact-index.json`.

Historical 90 environment steps from the stopped oracle attempt, all earlier costs and negative results, and the 10 ms cost failure remain in force.
