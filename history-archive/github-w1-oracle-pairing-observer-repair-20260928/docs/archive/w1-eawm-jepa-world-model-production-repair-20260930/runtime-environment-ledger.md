# Runtime environment ledger

Provider: local `Ubuntu-24.04` WSL2, native ext4 virtual environment.

- Interpreter: `/home/asus/w1-action-relative-auxiliary-runtime-dependency-repair-v1-runtime/bin/python`
- Python: 3.12.3
- PyTorch: 2.8.0+cpu, installed from the PyTorch CPU wheel index
- NumPy: 1.26.4, inherited through the venv's declared system-site-packages setting
- CUDA: not required; `torch.version.cuda` is null and `USE_CUDA=0`
- Dependency lock SHA-256: `725b14265d729fb8ac33cde1c9e5686e4638966d0d35244fed0d703c59d60e56`
- Environment spec SHA-256: `1ddb80b2055b7631ef6a691a05276b586f0e09703fae0832d8b524c9c398abe0`
- Validation date: 2026-09-29
- Validation witness: seeded CPU tensor matrix multiply, finite 4x4 output
- Prior failure: the old formal interpreter lacked `torch`; the repaired entry binds all three launch layers to this venv.

The environment was built before freezing. Formal launch paths contain no
package installation or network download operation.
