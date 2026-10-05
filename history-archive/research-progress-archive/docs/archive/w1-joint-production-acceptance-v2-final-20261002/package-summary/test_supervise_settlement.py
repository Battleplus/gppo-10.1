import sys
from types import ModuleType
import unittest

try:
    import resource  # noqa: F401
except ModuleNotFoundError:
    resource_stub = ModuleType("resource")
    resource_stub.RUSAGE_SELF = 0
    resource_stub.RUSAGE_CHILDREN = 1
    sys.modules["resource"] = resource_stub

try:
    import fcntl  # noqa: F401
except ModuleNotFoundError:
    fcntl_stub = ModuleType("fcntl")
    fcntl_stub.LOCK_EX = 2
    fcntl_stub.LOCK_NB = 4
    fcntl_stub.flock = lambda *args: None
    sys.modules["fcntl"] = fcntl_stub

from supervise import _apply_final_resource_status


class SuperviseSettlementTests(unittest.TestCase):
    def test_final_resource_overrun_cannot_retain_complete_status(self):
        state = {
            "status": "complete",
            "stop_reason": None,
            "final_resource_pass": False,
        }
        result = _apply_final_resource_status(state)
        self.assertIs(result, state)
        self.assertEqual(state["status"], "stopped")
        self.assertEqual(state["stop_reason"], "FINAL_RESOURCE_LIMIT_EXCEEDED")

    def test_existing_failure_reason_is_preserved(self):
        state = {
            "status": "complete",
            "stop_reason": "Worker exited 1",
            "final_resource_pass": False,
        }
        _apply_final_resource_status(state)
        self.assertEqual(state["status"], "stopped")
        self.assertEqual(state["stop_reason"], "Worker exited 1")


if __name__ == "__main__":
    unittest.main()
