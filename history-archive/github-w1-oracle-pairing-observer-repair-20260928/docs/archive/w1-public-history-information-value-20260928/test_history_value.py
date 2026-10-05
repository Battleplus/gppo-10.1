import importlib.util
import pathlib
import unittest


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("history_value", HERE / "analyze_history_value.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def obs(time, action=0):
    return {
        "time": float(time),
        "continuation_actions": [action],
        "public_entity_ids": {"uavs": ["u0"], "tasks": ["t0"]},
    }


class HistoryValueTests(unittest.TestCase):
    def test_field_preserves_unknown_zero_distinction(self):
        unknown = MODULE.public_field([0, 0, 0, 0], 0, 2)
        known_zero = MODULE.public_field([0, 1, 1, 1], 0, 2)
        self.assertFalse(unknown["known"])
        self.assertTrue(known_zero["known"])
        self.assertEqual(known_zero["value"], 0)
        self.assertEqual(known_zero["measured_at"], 1)

    def test_future_measurement_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.public_field([1, 1, 1, -0.1], 0, 2)

    def test_projection_uses_frozen_speed_and_caps_at_target(self):
        self.assertEqual(MODULE.project_toward((0, 0), (3, 0), 2), (2, 0))
        self.assertEqual(MODULE.project_toward((0, 0), (3, 0), 4), (3, 0))

    def test_continuation_requires_identity_and_no_gap(self):
        rows = [{"observation": obs(1)}, {"observation": obs(2)}]
        self.assertTrue(MODULE.continuation_is_continuous(rows, 1, 0, "u0", "t0")[0])
        wrong = [{"observation": obs(1)}, {"observation": obs(2, action=6)}]
        with self.assertRaises(ValueError):
            MODULE.continuation_is_continuous(wrong, 1, 0, "u0", "t0")
        gap = [{"observation": obs(1)}, {"observation": obs(3)}]
        self.assertFalse(MODULE.continuation_is_continuous(gap, 1, 0, "u0", "t0")[0])

    def test_exact_label_never_substitutes_future_time(self):
        index = {("uavs", "u0", "x", MODULE.time_key(3)): {"value": 7}}
        self.assertIsNone(MODULE.exact_label(index, "uavs", "u0", "x", 2))
        self.assertEqual(MODULE.exact_label(index, "uavs", "u0", "x", 3)["value"], 7)

    def test_analysis_source_has_no_runtime_import_or_call_entry(self):
        source = (HERE / "analyze_history_value.py").read_text(encoding="utf-8")
        forbidden = ("import torch", "import rl_adapters", "import runner", ".reset(", ".step(", "load_state_dict")
        for token in forbidden:
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
