"""Small tests for the contract's strict interval and dual-resource semantics."""
import unittest
import csv
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
from decimal import Decimal

import numpy as np

from build_graphs import build_edges, build_view, decimal_ns, grouped_edges, microsecond_ticks, parse_utc, read_contacts, REQUIRED_COLUMNS


class ConflictContractTests(unittest.TestCase):
    def edges(self, starts, ends, antenna=None, owner=None, ground_gap=0, satellite_gap=0):
        n = len(starts)
        edges, types, _, _ = build_edges(np.asarray([decimal_ns(x) for x in starts]),
                                        np.asarray([decimal_ns(x) for x in ends]),
                                        antenna or list(range(n)), owner or list(range(n)),
                                        ground_gap, satellite_gap)
        return {tuple(x): int(t) for x, t in zip(edges.tolist(), types.tolist())}

    def test_gap_equality_compatible(self):
        self.assertEqual(self.edges([0, 15], [5, 20], ["g", "g"], ground_gap=10), {})
        self.assertEqual(self.edges([0, "14.999999999"], [5, 20], ["g", "g"], ground_gap=10), {(0, 1): 1})

    def test_containment_and_identical_start_conflict(self):
        self.assertEqual(self.edges([0, 3, 0], [10, 4, 2], ["g"] * 3), {(0, 1): 1, (0, 2): 1})

    def test_satellite_can_reuse_channel_later(self):
        self.assertEqual(self.edges([0, 160], [10, 170], owner=["s", "s"], satellite_gap=150), {})
        self.assertEqual(self.edges([0, 159], [10, 170], owner=["s", "s"], satellite_gap=150), {(0, 1): 2})

    def test_dual_resource_union_counts_one_edge(self):
        self.assertEqual(self.edges([0, 2], [4, 6], ["g", "g"], ["s", "s"]), {(0, 1): 3})

    def test_tick_rounding_does_not_coarsen_edge(self):
        x = Decimal("14.999999999")
        self.assertEqual(microsecond_ticks(x), 15_000_000)
        self.assertEqual(self.edges([0, x], [5, 20], ["g", "g"], ground_gap=10), {(0, 1): 1})

    def test_long_enclosing_window_does_not_skip_later_overlap(self):
        self.assertEqual(self.edges([0, 2, 20], [100, 3, 21], ["g"] * 3), {(0, 1): 1, (0, 2): 1})

    def test_ground_gap_monotonicity(self):
        small = self.edges([0, 10, 14, 22], [5, 11, 19, 23], ["g"] * 4, ground_gap=5)
        large = self.edges([0, 10, 14, 22], [5, 11, 19, 23], ["g"] * 4, ground_gap=10)
        self.assertTrue(set(small).issubset(set(large)))

    def test_stk_utcg_and_iso_utc_refer_to_same_time(self):
        self.assertEqual(parse_utc("01 Nov 2026 00:00:12.123456789"),
                         parse_utc("2026-11-01T00:00:12.123456789Z"))

    def test_csv_decimal_rewards_owner_and_whole_window_subset(self):
        scene = {"source_group": "source-test", "geometry_id": "test", "replicate_id": "test",
                 "epoch_utc": "2026-11-01T00:00:00Z", "split": "train",
                 "satellites": [{"satellite_id": "sat-A"}, {"satellite_id": "sat-B"}]}
        stations = [{"site_id": "GS01", "antenna_id": "GS01-A1"},
                    {"site_id": "GS02", "antenna_id": "GS02-A1"}]
        epoch = datetime(2026, 11, 1, tzinfo=timezone.utc)
        rows = []
        for i, (start, end, sat, site) in enumerate([
            ("0.123456789", "10.345678901", "sat-A", "GS01"),
            ("30.123456789", "40.345678901", "sat-A", "GS01"),
            ("100.123456789", "120.345678901", "sat-B", "GS02")]):
            rows.append({"source_group": "source-test", "geometry_id": "test", "replicate_id": "test",
                         "epoch_utc": scene["epoch_utc"], "contact_id": f"contact-{i}", "pass_id": str(i),
                         "satellite_id": sat, "site_id": site, "antenna_id": site + "-A1",
                         "start_rel_seconds": start, "end_rel_seconds": end,
                         "duration_seconds": str(Decimal(end) - Decimal(start)),
                         "start_utc": (epoch + timedelta(seconds=float(start))).isoformat(),
                         "end_utc": (epoch + timedelta(seconds=float(end))).isoformat(),
                         "boundary_crossing": "none", "access_settings_hash": "test-only"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contacts.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted(REQUIRED_COLUMNS))
                writer.writeheader(); writer.writerows(rows)
            actual_rows, mother, validation = read_contacts(path, scene, stations, 259200)
            self.assertEqual(mother["owner"].tolist(), [0, 0, 1])
            self.assertEqual(mother["weights"][0], float(Decimal("10.345678901") - Decimal("0.123456789")))
            view, codes, stats = build_view(actual_rows, mother, {"GS02"}, stations, ["sat-A", "sat-B"],
                                            scene, {"ground_gap_seconds": 170, "satellite_gap_seconds": 150}, 259200)
            self.assertEqual(view["mother_contact_index"].tolist(), [2])
            self.assertEqual(view["owner"].tolist(), [1])
            self.assertEqual(view["start_ns"][0], mother["start_ns"][2])
            self.assertEqual(view["weights"][0], mother["weights"][2])
            self.assertTrue(stats["R8_unchanged_subset"])
            self.assertEqual(view["factor_vertices"].tolist(), [0, 0])
            self.assertEqual(len(codes), 0)


if __name__ == "__main__":
    unittest.main()
