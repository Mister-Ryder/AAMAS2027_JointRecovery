"""One scanner regression guard; no dataset, model, native or fitting."""
import unittest
from p1_actual_policy import scan_request_admission


class AdmissionScannerTest(unittest.TestCase):
    def test_skipped_top_four_long_requests_leave_quota_for_lower_short_requests(self):
        order = list(range(8))
        budgets = [1000, 1000, 1000, 1000, 10, 10, 10, 10]
        current = [0.]
        events = []
        for event in scan_request_admission(order, budgets, {"1000": 1.1, "10": .02},
                                             deadline=.1, max_requests=2, clock=lambda: current[0]):
            events.append(event)
            if event["admitted"]:
                current[0] += .02  # Fake elapsed execution only, no native call.
        self.assertEqual([e["index"] for e in events if not e["admitted"]], [0, 1, 2, 3])
        self.assertEqual([e["index"] for e in events if e["admitted"]], [4, 5])
        self.assertAlmostEqual(events[-1]["actual_remaining_seconds"], .08)


if __name__ == "__main__":
    unittest.main()
