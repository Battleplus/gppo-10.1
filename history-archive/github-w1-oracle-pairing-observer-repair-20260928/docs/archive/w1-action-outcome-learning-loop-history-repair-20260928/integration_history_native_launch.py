"""Linux integration entry for the production collector and controlled fake environment."""

from pathlib import Path

from native_launch import verify_native_launch
from supervise import main as supervise_main


ROOT = Path(__file__).resolve().parent


def main() -> int:
    verify_native_launch(ROOT)
    return supervise_main(ROOT, "integration_history_worker.py", sample_interval=0.05)


if __name__ == "__main__":
    raise SystemExit(main())
