# Runtime Probe and Audit Binding

The frozen Ubuntu-24.04 label runtime remains CPU-only. Its read-only
`dependency_probe.probe(expected_root=...)` validates the Python/torch/NumPy
identity, package and native project-module closure, and the remote-runtime
metadata binding. It never opens SSH connections or remote files.

The remote GPU identity is bound only for a future audit. The v2 identity,
outer delivery hash list and final remote read-only preflight summary are
checked against their local SHA-256 values by `freeze_contract.py`. The
binding explicitly records that it does not authorize remote execution or
training and does not alter the v6 CPU environment or resource request.

Each successful dependency probe performs exactly one synthetic 4x4 CPU
matrix multiplication as a runtime kernel witness. It performs no backward
pass, optimizer update, research-model forward, checkpoint operation, or
training. The launcher and Linux preflight outputs preserve these counts per
probe. Collector Graph5 feature tensor construction remains a separate
synthetic collector operation and is not a model forward.
