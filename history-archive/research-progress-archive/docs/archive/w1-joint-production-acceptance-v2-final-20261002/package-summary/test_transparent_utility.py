from __future__ import annotations

import unittest

from transparent_utility import transparent_utility_components


class TransparentUtilityTests(unittest.TestCase):
    def test_pure_helper_matches_the_frozen_policy_formula(self):
        diagnostic = {"score": 42.0, "energy_cost": 12.0}
        actual = transparent_utility_components(
            diagnostic, task_capacity=6, initial_total_energy=360.0,
        )
        expected = ((42.0 + 0.1 * 12.0) / (100.0 * 6), -12.0 / 360.0)
        self.assertAlmostEqual(actual[0], expected[0])
        self.assertAlmostEqual(actual[1], expected[1])


if __name__ == "__main__":
    unittest.main()
