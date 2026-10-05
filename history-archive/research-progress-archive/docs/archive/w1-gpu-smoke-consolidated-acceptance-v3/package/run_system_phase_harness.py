"""Run real Linux phase-accounting and supervisor lifecycle checks only."""
import json
import platform
import sys
import unittest


def main():
    if not sys.platform.startswith("linux"):
        print(json.dumps({"status": "unsupported", "platform": platform.platform(),
                          "reason": "native Linux /proc and AF_UNIX credentials required"}))
        return 2
    suite = unittest.defaultTestLoader.loadTestsFromNames((
        "test_phase_handshake",
        "test_supervised_phase_integration",
    ))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {
        "status": "pass" if result.wasSuccessful() else "fail",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "cuda_requested": False,
        "model_calls": 0,
        "real_system_interfaces": ["filesystem", "AF_UNIX", "SO_PEERCRED", "subprocess",
                                    "procfs", "CPU affinity", "resource guards", "durable export"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
