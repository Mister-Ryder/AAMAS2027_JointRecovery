"""Pure contract checks; these tests never instantiate STK or fabricate a dataset."""
import json
from pathlib import Path
import unittest

from stk_geometry import classify_interval, compare_intervals, decimal_endpoint, fixed_spot_check_pairs, utcg, validate_parameters


class GeometryContractTests(unittest.TestCase):
    def test_containment_does_not_round_outside_contact_inward(self):
        self.assertEqual(classify_interval(-0.0000001, 1.0, 10.0), "left")
        self.assertEqual(classify_interval(9.0, 10.0000001, 10.0), "right")
        self.assertEqual(classify_interval(0.0, 10.0, 10.0), "none")
        self.assertEqual(classify_interval(-1.0, 11.0, 10.0), "both")

    def test_endpoint_serialization_preserves_submicrosecond(self):
        value = 123456.789012345
        self.assertLessEqual(abs(float(decimal_endpoint(value)) - value), 0.5e-9)
        self.assertGreaterEqual(len(decimal_endpoint(value).split(".")[1]), 6)

    def test_bad_spot_check_cannot_be_marked_consistent(self):
        result = compare_intervals([(1.0, 2.0)], [(1.0, 2.0), (3.0, 4.0)])
        self.assertFalse(result["within_comparison_tolerance"])

    def test_input_and_fixed_spots(self):
        source = Path(__file__).resolve().parent.parent / "protocol" / "JointRecovery_parameters.json"
        parameters = json.loads(source.read_text(encoding="utf-8-sig"))
        selected = validate_parameters(parameters, ["r000", "r001"])
        self.assertEqual(len(selected), 2)
        for specification in selected:
            self.assertEqual(len(fixed_spot_check_pairs(specification, parameters["stations"])), 12)
        self.assertEqual(utcg(selected[0]["epoch_utc"]), "01 Nov 2026 00:00:00.000000")


if __name__ == "__main__":
    unittest.main()
