# v6 Dependency and CPU Settlement Repair

The v5 frozen package, sealed attempt, stop records, copied payload and failed
export acceptance evidence are unchanged. A payload copy is not a successfully
accepted export. The new proposed attempt is
`w1-action-conditioned-task-outcome-label-qualification-v6-dependency-cpu-settlement-once`.
This preparation does not create or approve that attempt.

## Dependency root cause

The v5 production collector dynamically imported
`production_policy.transparent_utility_components` near the former line 373.
The operation converts transparent utility into fixed numerical components;
it does not require a policy or model. The policy module was absent from the
collection payload. The import failed before environment construction.

`transparent_utility.py` retains this numerical operation as a collection-only
module. Its provenance is represented separately from unchanged historical
source modules: a new derived file cannot be claimed to exist in the old
source manifest. The production collector imports package-staged project code.
Recursive package identities and staged dependency checks must include the
native source files, rather than only top-level Python files.

## CPU root cause and repair

The v5 outer process-tree CPU was 2.811616 seconds; the nested supervisor
reported 2.536691 seconds. The 0.274925-second difference cannot be split into
specific startup/exit processes from retained v5 evidence. Those snapshots have
different inclusion/sampling scopes; adding them would double count children.

v6 preserves the 0.05-second existing consistency tolerance. The authoritative
outer Linux RUSAGE_CHILDREN measurement is charged once. Supervisor and native
launcher snapshots carry timestamps and self/children components for
reconciliation; they are not additional CPU charges. Windows and Linux
launcher work, settlement and export remain charged under the existing bounds.
Budget closure and failed/incomplete reconciliation stop acceptance.

## Scope and acceptance

The task semantics remain arrival_to_region/physical_arrival, zero arrival
radius, fixed train-0056 through train-0063, legal public candidates including
NOOP, and fixed Hungarian continuation. Input freezing and future-label
isolation are retained. Unknown is not imputed. The coverage rule and all
dynamic/resource limits are unchanged; this package has no model or policy
training stage.

Synthetic environment boundaries, OS workload fixtures and tensor dependency
witnesses are preparation evidence. They do not establish real label coverage,
world-model prediction, GPPO utility or cost performance. Final test and
read-only preflight evidence belongs outside the frozen package. Read-only
preflight verifies identity and dependencies, not a real collection run.

The separate server GPU runtime is a future training candidate. This label
package keeps its current frozen WSL CPU interpreter. The server's existing
training processes and non-exclusive resources were not changed. Only a new
explicit label-collection authorization may consume this proposed attempt.
