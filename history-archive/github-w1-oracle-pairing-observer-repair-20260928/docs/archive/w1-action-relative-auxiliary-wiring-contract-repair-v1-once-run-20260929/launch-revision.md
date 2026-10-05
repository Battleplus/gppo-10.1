# Launch wiring revision

The prior package stopped before execution because its entrypoint required an
in-place `APPROVED` edit and then raised `RUNNER_NOT_ATTACHED`. This package
keeps the frozen request at `NOT_APPROVED` and binds authorization to the new
attempt, manifest, hashes, and an external one-shot token. The Windows entry
only validates and delegates through `wsl.exe`; Linux performs staging, fsync,
supervision, settlement, and controlled export on native WSL storage.

The research contract is unchanged except for one explicit execution-time
condition: the A/B effect gate must pass and B must also be strictly better
than the transparent-history baseline before the four-arm task stage is
called. Passing only the A/B gate produces a settled report with zero task
calls.

The integration entry is test-only and requires `integration_test=true` in a
temporary sealed clone. The production `native_launch.py` rejects that mode,
so a fake backend cannot enter the formal runner accidentally.
