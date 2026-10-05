"""Integration-only Linux entry: real supervisor with a zero-call fake worker."""
from pathlib import Path

from native_launch import verify_native_launch
from supervise import main as supervise_main


ROOT = Path(__file__).resolve().parent


def main() -> int:
    verify_native_launch(ROOT)
    return supervise_main(ROOT, "integration_fake_worker.py", sample_interval=0.05)


if __name__ == "__main__":
    raise SystemExit(main())
