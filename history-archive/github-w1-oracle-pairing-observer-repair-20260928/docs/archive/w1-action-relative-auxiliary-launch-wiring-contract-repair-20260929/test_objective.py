"""Zero-call tests for the relative-utility preparation contract."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from objective import CandidateRow, Window, aggregate_window_loss, group_rows, window_loss
from trace_schema import PredictionTraceRow, group_trace, read_trace, stable_input_hash, write_trace


def make_window(decision: str, targets: tuple[float, ...], *, parent: str = "p0") -> Window:
    actions = tuple(range(len(targets)))
    rows = tuple(CandidateRow(decision, parent, 0, action, actions, target, noop=action == 24) for action, target in zip(actions, targets, strict=True))
    window = Window(decision, rows)
    window.validate()
    return window


class ObjectiveTests(unittest.TestCase):
    def test_constant_shift_does_not_change_centered_term(self):
        base = window_loss([0.1, 0.7, 1.2], [0.0, 1.0, 1.0], variant="centered_auxiliary", auxiliary_weight=0.25)
        shifted_predictions = window_loss([4.1, 4.7, 5.2], [0.0, 1.0, 1.0], variant="centered_auxiliary", auxiliary_weight=0.25)
        self.assertAlmostEqual(base["centered"], shifted_predictions["centered"])
        self.assertNotEqual(base["absolute"], shifted_predictions["absolute"])

    def test_centered_term_sees_ordering_error(self):
        correct = window_loss([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], variant="centered_auxiliary", auxiliary_weight=1.0)
        reversed_scores = window_loss([2.0, 1.0, 0.0], [0.0, 1.0, 2.0], variant="centered_auxiliary", auxiliary_weight=1.0)
        self.assertLess(correct["centered"], reversed_scores["centered"])

    def test_zero_auxiliary_recovers_absolute_variant(self):
        absolute = window_loss([0.0, 2.0], [1.0, 1.0], variant="absolute", auxiliary_weight=0.0)
        zero_b = window_loss([0.0, 2.0], [1.0, 1.0], variant="centered_auxiliary", auxiliary_weight=0.0)
        self.assertEqual(absolute["total"], zero_b["total"])
        self.assertEqual(absolute["absolute"], zero_b["absolute"])

    def test_windows_are_equally_weighted(self):
        first = make_window("d1", (0.0, 1.0), parent="p1")
        second = make_window("d2", (0.0, 0.0, 0.0, 0.0, 0.0), parent="p2")
        values = {"d1": [0.0, 0.0], "d2": [1.0, 1.0, 1.0, 1.0, 1.0]}
        result = aggregate_window_loss((first, second), values, variant="absolute")
        expected = (window_loss(values["d1"], [0.0, 1.0], variant="absolute")["total"] + window_loss(values["d2"], [0.0] * 5, variant="absolute")["total"]) / 2
        self.assertAlmostEqual(result["total"], expected)

    def test_noop_and_unknown_are_explicit(self):
        window = make_window("d-noop", (0.0, 0.5, 0.25), parent="p3")
        self.assertTrue(window.rows[-1].noop is False)
        noop = CandidateRow("d-noop-only", "p3", 0, 24, (24,), 0.0, noop=True)
        Window("d-noop-only", (noop,)).validate()
        unknown = CandidateRow("d-unknown", "p3", 0, 0, (0,), 0.0, target_valid=False, noop=False)
        with self.assertRaises(ValueError):
            Window("d-unknown", (unknown,)).validate()
        nonfinite = CandidateRow("d-nan", "p3", 0, 0, (0,), math.nan, noop=False)
        with self.assertRaises(ValueError):
            nonfinite.validate()

    def test_group_requires_complete_candidate_set(self):
        row = CandidateRow("d", "p", 0, 0, (0, 1), 0.0, noop=False)
        with self.assertRaises(ValueError):
            group_rows((row,))


class TraceTests(unittest.TestCase):
    def make_rows(self):
        digest = stable_input_hash({"decision": "d", "public": [1, 2]})
        return [PredictionTraceRow("p", 0, "d", action, (0, 1), "hungarian-v1-fixed", digest, float(action), True, 0.1, 0.2, (float(action),) * 3, float(action)) for action in (0, 1)]

    def test_trace_round_trip_and_ensemble(self):
        rows = self.make_rows()
        self.assertEqual(len(group_trace(rows)[("A", "d")]), 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prediction-trace.jsonl"
            write_trace(path, rows)
            self.assertEqual(read_trace(path), rows)

    def test_trace_rejects_incomplete_window(self):
        rows = self.make_rows()[:1]
        with self.assertRaises(ValueError):
            group_trace(rows)


if __name__ == "__main__":
    unittest.main()
