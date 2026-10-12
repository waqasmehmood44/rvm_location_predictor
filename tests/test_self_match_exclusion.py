"""
Unit tests for Self-Match Exclusion in Spatial Catchment & Benchmarking.
Verifies that evaluating an already-installed Coripet RVM (e.g., Esselunga Bergamo):
1. Correctly excludes the self-machine from neighboring cannibalization calculations.
2. Identifies the nearest other machine in the network.
3. Leaves genuine new store predictions and competitor installations completely unaffected.
"""
import unittest
from src.spatial.catchment import SpatialCatchmentAnalyzer


class TestSelfMatchExclusion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.analyzer = SpatialCatchmentAnalyzer()

    def test_installed_store_self_match_detected(self):
        """Esselunga Bergamo (Via San Bernardino) should exclude itself and measure distance to next nearest machine."""
        if not self.analyzer.network_loaded:
            self.skipTest("Network parquet file not available.")

        # Exact location of MCPE5B7FG2AT
        lat, lon = 45.684709, 9.663194
        res = self.analyzer.analyze_coripet_network(
            lat=lat,
            lon=lon,
            cap="24122",
            comune="Bergamo",
            address="Via San Bernardino Angolo Via Spino",
            store_name="ESSELUNGA",
            exclude_self_benchmark=True
        )

        self.assertTrue(res["is_benchmark_self_match"], "Installed machine should be identified as a self-match.")
        self.assertEqual(res["benchmark_pf_id"], "MCPE5B7FG2AT")
        self.assertEqual(res["benchmark_pf_name"], "ESSELUNGA")
        # Nearest other machine in Bergamo is ~3.28 km away, NOT 0.0m
        self.assertGreater(res["nearest_coripet_km"], 1.0)
        self.assertNotEqual(res["network_position"], "CANNIBALIZATION_ALERT")

    def test_new_store_does_not_trigger_self_match(self):
        """A brand new store at an address with no installed machine should not be flagged as a self-match."""
        if not self.analyzer.network_loaded:
            self.skipTest("Network parquet file not available.")

        res = self.analyzer.analyze_coripet_network(
            lat=41.940346,
            lon=12.506457,
            cap="00185",
            comune="Roma",
            address="Via Roma 1",
            store_name="NEW CONAD SUPERMARKET",
            exclude_self_benchmark=True
        )

        self.assertFalse(res["is_benchmark_self_match"], "Brand new store should not trigger self-match.")
        self.assertIsNone(res["benchmark_pf_id"])

    def test_competitor_adjacent_preserves_cannibalization(self):
        """A competing retailer (e.g. Lidl) co-located near an existing Esselunga machine should NOT be self-excluded."""
        if not self.analyzer.network_loaded:
            self.skipTest("Network parquet file not available.")

        # Co-located right next to Esselunga Bergamo
        lat, lon = 45.684709, 9.663194
        res = self.analyzer.analyze_coripet_network(
            lat=lat,
            lon=lon,
            cap="24122",
            comune="Bergamo",
            address="Via San Bernardino Angolo Via Spino",
            store_name="LIDL SUPERSTORE",
            exclude_self_benchmark=True
        )

        self.assertFalse(res["is_benchmark_self_match"], "Different retail banner should not self-exclude.")
        self.assertLess(res["nearest_coripet_km"], 0.1)
        self.assertEqual(res["network_position"], "CANNIBALIZATION_ALERT")


if __name__ == "__main__":
    unittest.main()
