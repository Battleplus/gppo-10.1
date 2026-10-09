# v4 Code Differences

- `package/event_signal_contract.py` defines the frozen pre-action value, validity and unknown rejection rules.
- `package/public_controller.py` carries `event_signal_valid`, validates the field, and supports an explicit restored-prefix override.
- `package/consequence_contract.py` includes the validity field in the public whitelist and validates it before task-slot construction.
- `package/consequence_collection.py` preserves the saved pre-action value during branch probing and records an event-signal contract audit.
- `package/native/gppo_world/m10_environment.py` emits `event_signal_valid=true` with every observed public state.
- `package/native/gppo_world/graph5.py` rejects missing or unknown event signals before graph construction.
- `acceptance/candidate_contract_acceptance.py` adds event false/true/unknown, missing, type, pre/post mix and graph-boundary cases to the real Driver/BudgetLedger/StageServer fixture.

No learned tensor is changed and no constant `1.0` bypass is present. Valid false remains `0.0`; unknown remains invalid.
