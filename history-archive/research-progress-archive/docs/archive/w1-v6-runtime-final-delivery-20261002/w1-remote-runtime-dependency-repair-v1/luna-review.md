# Luna independent runtime review

Reviewer `/root/runtime_identity_review` used the source files and saved
artifacts only, without SSH, package installation, tensors or model operations.

The reviewer confirmed the synthetic counts (3 tensor checks, 1 backward,
1 optimizer initialization and 1 update), default-runtime failure, separate
venv success, pinned dependencies and lack of proven exclusive GPU allocation.
The reviewer required stronger process/search-path checks, same-byte outer
digest parsing and installed-payload verification. These were implemented and
the latest eight regression tests pass.

Stricter payload validation revealed a NumPy cached-bytecode/RECORD discrepancy;
the raw failure was retained. The source payload matches RECORD. The new v2
runtime contract prevents reading that cache with a fixed absent pycache_prefix
and disabled writes. Source and native payload checks remain active.

The follow-up reviewer confirmed that the enforced absent pycache_prefix makes
ordinary __pycache__ files unreachable. The reviewer identified the overly broad
exemption of all .pyc files; it was narrowed to __pycache__ entries, and directly
loadable bytecode now requires a checked digest or is rejected. This case has a
dedicated regression. CLI isolation is enforced; the internal verify() helper
alone is not a launch entry. These changes address the review's conditional
blocker; the source-based review is not full training acceptance.

Neither the review nor the synthetic probe proves production training readiness,
exclusive resource allocation or target-workload memory fit. Real training and
formal attempts were not run.
