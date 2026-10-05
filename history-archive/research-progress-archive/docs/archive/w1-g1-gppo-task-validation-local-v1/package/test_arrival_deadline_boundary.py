"""Pure component fixtures; no M10 environment or model is constructed."""
import unittest

from gppo_world.service_clock import ServiceClock, ServiceResource
from gppo_world.task_lifecycle import TaskLifecycle, TaskState


class ArrivalDeadlineBoundaryTests(unittest.TestCase):
    def clock(self, distance):
        task = TaskLifecycle("t", 0.0, 1.0, 1.0,
                             completion_mode="arrival_to_region", completion_radius=0.0)
        clock = ServiceClock({"t": task}, {"u": ServiceResource(10.0, 1.0, 1.0)},
                             [], disconnect_interrupts=False,
                             task_positions={"t": (distance, 0.0)})
        clock.assign("t", "u")
        return clock, task

    def test_arrival_before_deadline_is_completed_and_logged(self):
        clock, task = self.clock(0.5)
        clock.advance(1.0)
        self.assertEqual(task.state, TaskState.COMPLETED)
        self.assertEqual(task.completed_at, 0.5)
        self.assertEqual([r["time"] for r in clock.log if r["kind"] == "arrival"], [0.5])

    def test_arrival_at_deadline_expires_before_completion_log(self):
        clock, task = self.clock(1.0)
        with self.assertRaisesRegex(ValueError, "Arrival requires the assigned UAV"):
            clock.advance(1.0)
        self.assertEqual(task.state, TaskState.EXPIRED)
        self.assertIsNone(task.completed_at)
        self.assertFalse(any(r["kind"] == "arrival" for r in clock.log))

    def test_deadline_before_arrival_expires_without_completion(self):
        clock, task = self.clock(2.0)
        clock.advance(1.0)
        self.assertEqual(task.state, TaskState.EXPIRED)
        self.assertIsNone(task.completed_at)
        self.assertFalse(any(r["kind"] == "arrival" for r in clock.log))


if __name__ == "__main__":
    unittest.main()
