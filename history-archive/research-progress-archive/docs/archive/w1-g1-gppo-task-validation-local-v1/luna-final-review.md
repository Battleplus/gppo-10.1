# Independent Final Review

Review date: 2026-10-04  
Target: frozen G1 + GPPO task-validation package

## Disposition

No blocking source or frozen-identity issue was found in the audited one-shot entry, authorization checks, recurrent replay and hidden-state refresh, H timing boundary, frozen-model binding, or request arithmetic. The frozen package is suitable for submitting a formal resource request for separate approval. This review does not authorize a run: the package request and external authorization remain `NOT_APPROVED`, no token is present or validated, and no formal attempt, staging, or worker was started.

`runner_ready=true` records engineering readiness to request execution. It does not mean approved, full resource acceptance, or research success. Real task efficacy remains unevaluated.

## Frozen Identity

The current `package` and `frozen-archive/package` each contain 209 files with identical inventories and SHA-256 bytes. All 209 file digests match the frozen delivery record and archived copies. The ordinary package hash map contains 207 content entries; `hashes.json` and `execution-manifest.json` are checked through the outer delivery record to avoid self-referential hashes. Verified identities:

- Execution manifest: `219cd82a046ebffd3feb3f1e2f9bc5b158f7ab040273d6baef5b866b01ff2895`
- Hashes file: `aea59645b889fda413fa9ef110453622d402019016132b0daf59d5779d6d465f`
- Resource request: `85939a3d6211dc86c5db2a56a368b657c42975468b48dadc141e99b7a7e57e8a`

Retained synthetic provenance binds 151 fixture Python files and 11 loaded production modules to the final package bytes. Final Windows and WSL read-only preflights both exited 0; their scope was structure, runtime/source identity, and dependencies, not authorization-token validation. No package test suite or model/environment run was performed for this review. Two distinct retained preparation records report an earlier focused unittest run with 35 tests passed plus syntax compilation, and a later `validate_preparation.py` run with 34 tests, zero errors, and zero failures. These are separate runs and are not summed.

## Synthetic Pipeline

The synthetic orchestration completed nine policy routes, 18 optimizer updates, and 240 task episodes, then settled and verified its export. The ledger records nine policy checkpoint writes, 15 checkpoint loads, 24 model initialization/load entries, and six in-memory world-model substitutions. It used synthetic environment boundaries and reduced route budgets; it did not deserialize real frozen model checkpoints or construct a real research environment. These results validate software/accounting paths, not research efficacy or full-budget completion.

Keep the two attempts and their exit records separate. The first non-isolated attempt exited 1; its verifier listed `hidden_replay_observations_match_ledger_encode_charges` and `reduced_route_budgets`, and its WSL `/tmp` tree was not retained. The final isolated attempt also has a recorded driver exit 1, but its retained `controlled-run-evidence.json` shows only `hidden_replay_observations_match_ledger_encode_charges` failed. Its separate read-only post-run audit exited 0. That audit confirms, for each of G0, T, and G1, three replay routes totaling 36 observations, 36 ledger calls, and 36 encoder charges. The per-method aggregation is the right comparison; a route subtotal must not be compared directly with the method-wide total. The successful later audit does not change the final isolated driver's exit 1.

Both registered synthetic gates failed. The task gate and `research_success` are false. For G1, the cost gate evaluated 1,296 samples and measured 56.548 ms mean CPU and 159.318 ms p95 wall against limits of 10 ms and 50 ms; the cost gate failed. This is a material engineering measurement and must not be presented as a cost pass or as evidence for a positive research result.

## Remaining Limits

The resource request lists unaccepted accounting gaps for Windows/WSL bridge CPU, the final controller write/exit tail, and shared CPU load/cross-clock differences. Full resource acceptance is false. The frozen task split contains eight confirmation parents, while project-wide historical non-use is unproven. Prior G1 improvement was measured against a transparent reference that selected NOOP in all eight windows under its optimistic free-continuation score; it is not evidence of superiority over a generally competent transparent policy. The G1 residual prior uses a fixed-H continuation target while GPPO's later actions can differ, so its transfer to GPPO remains a hypothesis.

The exact-deadline audit does not establish a reachable inclusive-label defect: equality expires before an arrival is logged. The native inclusive flag is therefore not a completed-arrival counterexample on that path.

## Recommendation

Engineering readiness to submit the formal request is supported, with the synthetic task and cost failures and the accounting gaps disclosed. Approval, full resource acceptance, and real task-efficacy evidence remain separate gates.
