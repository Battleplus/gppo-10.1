# Independent Local Package Review

**Target:** `research-plans/w1-task-outcome-g1-g2-joint-local-v1/package`

## Conclusion

Static review found no remaining deterministic startup blocker in the frozen package. This is a bounded readiness finding, not execution approval or full resource acceptance. The package manifest and resource request remain `NOT_APPROVED`, and `runner_ready` is false.

The formal entry is bound to `run_local_research.py`. The controller stages the frozen inputs, verifies their identities, then runs the native read-only preflight before starting the research worker. That native preflight is pending; its designed position in the launch path must not be reported as a completed check. The native controller records `full_resource_acceptance=false`.

## Evidence Reviewed

- The package manifest binds `run_local_research.py`; the manifest and hash-file SHA-256 values are `e08328eeeab23d4cf5d64456f5b7b12d5604b793e56a44963ae233f71f0c590c` and `4303b91ac78e056a0d4fa187a6ab358e77d215df9fb646b4693f9ec314c8217d`. These match the supplied frozen-preflight record.
- Per that supplied record, the Windows read-only structural preflight exited 0 and the before/after package identities matched. This review did not rerun it.
- Offline preparation evidence reports source-input verification passed, production research code byte-identical to the smoke production copy, and zero environment/model/CUDA calls. I independently checked the matching hashes for `production_data.py`, `production_world.py`, `joint_pipeline.py`, `runner.py`, and `task_outcome_contract.py`. These checks do not establish full formal-run behavior.
- The formal design binds 40 parent-disjoint rows, split 24 train / 8 model-selection / 8 prediction-confirmation, from source tape indices 64–103. The split is fixed and hash-bound. Historical global novelty is explicitly unproven; the source is generated W1 simulator data, not field UAV data.
- The planned matrix runs G1 and G2 across seeds 8201–8203, with up to 20 epochs and 480 optimizer updates per route. The resource request retains the existing stage budgets and call ceilings and caps PyTorch allocated/reserved memory at 4 GiB. The completed production smoke covered only 12 optimizer updates and 84 batch forwards, so it demonstrates an engineering path, not capacity or completion of the formal workload.
- Accounting charges the native controller's own CPU plus waited descendants once; nested supervisor/worker values are diagnostics, and Windows controller CPU is reported separately. The disclosed gaps remain WSL bridge CPU, WDDM process GPU memory/exclusivity, the native controller's final write/exit tail, and hard enforcement of sampled CPU limits. These gaps support the package's `full_resource_acceptance=false` status.

## Review Scope

This was a read-only static review. I did not run package code, construct the environment, install dependencies, initialize CUDA, execute a model, or load a checkpoint. No package files were changed. Preparation and smoke evidence above are attributed to their records; they are not represented as work performed during this review.

No findings remain that would make the current bound entry deterministically fail before the native preflight. The pending native preflight and the disclosed resource gaps remain the next evidence gates; neither should be described as already passed.
