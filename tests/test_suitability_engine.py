"""
Unit tests for the Location Suitability & Disqualification Engine.
Tests the 4 operational/economic clearance gates (Cannibalization, Economic Breakeven, Store Format, Demographics).
"""

import unittest
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.model.scoring import (
    evaluate_location_suitability,
    determine_recommendation_tier,
    calculate_catchment_realization_factor
)


class TestLocationSuitabilityEngine(unittest.TestCase):

    def test_greenfield_high_volume_suitable(self):
        """Test a clear greenfield prospective store with strong volume and zero cannibalization."""
        suit = evaluate_location_suitability(
            realized_daily_bottles=710.2,
            unconstrained_daily_bottles=710.2,
            nearest_coripet_km=3.7,
            nearest_pf_name="CONAD",
            dilution_percentage=0.0,
            network_position="EXPANSION_NODE",
            sales_area_sqm=1200.0,
            numero_casse=6.0,
            population=199015.0,
            population_density=2338.3,
            total_grocery_stores=6
        )

        self.assertTrue(suit["is_suitable"])
        self.assertIn(suit["suitability_status"], ["SUITABLE", "HIGHLY_SUITABLE"])
        self.assertIn("GO", suit["verdict"])
        self.assertGreaterEqual(suit["suitability_score"], 75)
        
        # All gates must PASS
        for gate_name, gate in suit["gate_results"].items():
            self.assertEqual(gate["status"], "PASS", f"{gate_name} did not pass")

        rec = determine_recommendation_tier(
            potential_score=72,
            daily_bottles=710.2,
            cannibalization_alert=False,
            suitability_status=suit["suitability_status"]
        )
        self.assertIn("PROCEED", rec["decision"])

    def test_severe_cannibalization_rejection(self):
        """Test that an existing RVM at 0m trips the Cannibalization Gate and yields NO-GO / REJECT."""
        suit = evaluate_location_suitability(
            realized_daily_bottles=173.3,
            unconstrained_daily_bottles=495.0,
            nearest_coripet_km=0.0,
            nearest_pf_name="Deco",
            dilution_percentage=65.0,
            network_position="CANNIBALIZATION_ALERT",
            sales_area_sqm=780.0,
            numero_casse=6.0,
            population=73000.0,
            population_density=1350.0,
            total_grocery_stores=39
        )

        self.assertFalse(suit["is_suitable"])
        self.assertEqual(suit["suitability_status"], "NOT_SUITABLE")
        self.assertEqual(suit["verdict"], "NO-GO [REJECT]")
        self.assertEqual(suit["gate_results"]["cannibalization_gate"]["status"], "FAIL")
        self.assertTrue(len(suit["disqualifying_reasons"]) > 0)
        self.assertIn("DO NOT install", suit["recommended_action"])

        rec = determine_recommendation_tier(
            potential_score=50,
            daily_bottles=173.3,
            cannibalization_alert=True,
            suitability_status=suit["suitability_status"]
        )
        self.assertEqual(rec["tier_name"], "TIER 5 - NOT SUITABLE / UNVIABLE")
        self.assertEqual(rec["decision"], "NOT RECOMMENDED (DO NOT INSTALL)")

    def test_micro_format_insufficient_scale_rejection(self):
        """Test that a micro-store (60 m2, 1 checkout) trips the Format Gate and low volume trips the Economic Gate."""
        suit = evaluate_location_suitability(
            realized_daily_bottles=85.0,
            unconstrained_daily_bottles=250.0,
            nearest_coripet_km=2.5,
            nearest_pf_name="Conad",
            dilution_percentage=0.0,
            network_position="EXPANSION_NODE",
            sales_area_sqm=60.0,
            numero_casse=1.0,
            population=50000.0,
            population_density=1000.0,
            total_grocery_stores=5
        )

        self.assertFalse(suit["is_suitable"])
        self.assertEqual(suit["suitability_status"], "NOT_SUITABLE")
        self.assertEqual(suit["verdict"], "NO-GO [REJECT]")
        self.assertEqual(suit["gate_results"]["store_format_gate"]["status"], "FAIL")
        self.assertEqual(suit["gate_results"]["economic_volume_gate"]["status"], "FAIL")

        rec = determine_recommendation_tier(
            potential_score=20,
            daily_bottles=85.0,
            cannibalization_alert=False,
            suitability_status=suit["suitability_status"]
        )
        self.assertEqual(rec["tier_name"], "TIER 5 - NOT SUITABLE / UNVIABLE")
        self.assertEqual(rec["decision"], "NOT RECOMMENDED (DO NOT INSTALL)")

    def test_conditional_pilot_marginal_store(self):
        """Test that a store with marginal volume near 250 b/d breakeven gets CONDITIONAL status without being rejected."""
        suit = evaluate_location_suitability(
            realized_daily_bottles=275.0,
            unconstrained_daily_bottles=300.0,
            nearest_coripet_km=1.2,
            nearest_pf_name="Pam",
            dilution_percentage=8.0,
            network_position="EXPANSION_NODE",
            sales_area_sqm=350.0,
            numero_casse=3.0,
            population=30000.0,
            population_density=800.0,
            total_grocery_stores=4
        )

        self.assertTrue(suit["is_suitable"])
        self.assertEqual(suit["suitability_status"], "CONDITIONAL_PILOT")
        self.assertEqual(suit["verdict"], "CONDITIONAL [REVIEW REQUIRED]")
        self.assertEqual(suit["gate_results"]["economic_volume_gate"]["status"], "WARN")

    def test_rural_monopoly_catchment_uplift(self):
        """Test that a solo rural supermarket in Area 4 receives sole monopoly + rural bulk boosts with zero walk-in penalty."""
        real = calculate_catchment_realization_factor(
            unconstrained_daily_bottles=159.4,
            unconstrained_daily_presenze=5.4,
            nearest_coripet_km=0.0,
            nearest_pf_name="Deco",
            store_name="Deco",
            coripet_count_1km=1,
            coripet_count_3km=1,
            coripet_count_5km=6,
            grocery_total_stores=1,
            network_position="CANNIBALIZATION_ALERT",
            sales_area_sqm=387.0,
            numero_casse=2.0,
            is_independent_banner=False,
            nielsen_area="Area 4.0",
            is_proximity_market=False,
            population_density=47.8
        )
        self.assertEqual(real["realization_factor"], 1.65)
        self.assertEqual(real["uplift_percentage"], 65.0)
        self.assertEqual(real["dilution_percentage"], 0.0)
        self.assertAlmostEqual(real["realized_daily_bottles"], 263.0, delta=1.0)
        self.assertGreaterEqual(real["realized_daily_presenze"], 6.7)
        self.assertEqual(len(real["dilution_reasons"]), 2)
        self.assertTrue(any("Monopoly" in r for r in real["dilution_reasons"]))
        self.assertTrue(any("Rural Bulk" in r for r in real["dilution_reasons"]))

    def test_urban_competitive_catchment_dilution(self):
        """Test that an urban store with multiple competitors receives standard corridor dilution and zero monopoly uplift."""
        real = calculate_catchment_realization_factor(
            unconstrained_daily_bottles=223.5,
            unconstrained_daily_presenze=7.8,
            nearest_coripet_km=4.4,
            nearest_pf_name="Conad",
            store_name="CONAD_ITALIA",
            coripet_count_1km=0,
            coripet_count_3km=1,
            coripet_count_5km=3,
            grocery_total_stores=27,
            network_position="DENSE_CLUSTER",
            sales_area_sqm=420.0,
            numero_casse=4.0,
            is_independent_banner=False,
            nielsen_area="Area 3.0",
            is_proximity_market=False,
            population_density=491.8
        )
        self.assertEqual(real["realization_factor"], 0.95)
        self.assertEqual(real["uplift_percentage"], 0.0)
        self.assertEqual(real["dilution_percentage"], 5.0)
        self.assertAlmostEqual(real["realized_daily_bottles"], 212.3, delta=1.0)
        self.assertTrue(any("Competitive Food Retail Corridor" in r for r in real["dilution_reasons"]))

    def test_hardware_oversized_warning(self):
        """Test that requesting a heavy-duty machine on low volume (<250 b/d) triggers a sizing warning."""
        suit = evaluate_location_suitability(
            realized_daily_bottles=145.3,
            unconstrained_daily_bottles=150.0,
            nearest_coripet_km=2.0,
            nearest_pf_name="Conad",
            sales_area_sqm=400.0,
            numero_casse=3.0,
            population=40000.0,
            population_density=800.0,
            total_grocery_stores=5,
            requested_machine="DAINT_A-bis"
        )
        self.assertEqual(suit["gate_results"]["hardware_sizing_gate"]["status"], "WARN")
        self.assertEqual(suit["gate_results"]["hardware_sizing_gate"]["flag"], "OVERSIZED HARDWARE")
        self.assertTrue(any("oversized" in w.lower() for w in suit["warning_flags"]))

    def test_hardware_undersized_warning(self):
        """Test that requesting a compact machine on high volume (>600 b/d) triggers an undercapacity warning."""
        suit = evaluate_location_suitability(
            realized_daily_bottles=650.0,
            unconstrained_daily_bottles=650.0,
            nearest_coripet_km=3.0,
            nearest_pf_name="Conad",
            sales_area_sqm=1500.0,
            numero_casse=8.0,
            population=80000.0,
            population_density=1200.0,
            total_grocery_stores=10,
            requested_machine="Ripet_V2"
        )
        self.assertEqual(suit["gate_results"]["hardware_sizing_gate"]["status"], "WARN")
        self.assertEqual(suit["gate_results"]["hardware_sizing_gate"]["flag"], "UNDERCAPACITY RISK")
        self.assertTrue(any("undersized" in w.lower() for w in suit["warning_flags"]))

    def test_commercial_anchor_foot_traffic(self):
        """Test commercial retail anchor site evaluation."""
        suit_ind = evaluate_location_suitability(
            realized_daily_bottles=150.0,
            unconstrained_daily_bottles=150.0,
            nearest_coripet_km=2.0,
            nearest_pf_name="Conad",
            sales_area_sqm=350.0,
            numero_casse=3.0,
            brand_name="INDIPENDENTE",
            retail_group="UNKNOWN"
        )
        self.assertFalse(any("Commercial retail location" in s for s in suit_ind["key_strengths"]))

        # Aligned retail site
        suit_chain = evaluate_location_suitability(
            realized_daily_bottles=150.0,
            unconstrained_daily_bottles=150.0,
            nearest_coripet_km=2.0,
            nearest_pf_name="Conad",
            sales_area_sqm=350.0,
            numero_casse=3.0,
            brand_name="ESSELUNGA",
            retail_group="ESSELUNGA"
        )
        self.assertTrue(any("Commercial retail location" in s for s in suit_chain["key_strengths"]))

    def test_logistics_feasibility_gate(self):
        """Test logistics servicing radius evaluation for isolated outposts (>25 km)."""
        # Isolated outpost with low volume (<250 b/d)
        suit_iso_low = evaluate_location_suitability(
            realized_daily_bottles=150.0,
            unconstrained_daily_bottles=150.0,
            nearest_coripet_km=32.0,
            nearest_pf_name="Conad",
            sales_area_sqm=500.0,
            numero_casse=4.0
        )
        self.assertEqual(suit_iso_low["gate_results"]["logistics_feasibility_gate"]["status"], "WARN")
        self.assertEqual(suit_iso_low["gate_results"]["logistics_feasibility_gate"]["flag"], "LOGISTICS DEADHEAD")

        # Isolated outpost with high volume (>=250 b/d)
        suit_iso_high = evaluate_location_suitability(
            realized_daily_bottles=450.0,
            unconstrained_daily_bottles=450.0,
            nearest_coripet_km=32.0,
            nearest_pf_name="Conad",
            sales_area_sqm=1200.0,
            numero_casse=6.0
        )
        self.assertEqual(suit_iso_high["gate_results"]["logistics_feasibility_gate"]["status"], "PASS")
        self.assertEqual(suit_iso_high["gate_results"]["logistics_feasibility_gate"]["flag"], "HIGH-YIELD OUTPOST")

    def test_mall_anchor_synergy(self):
        """Test shopping center / mall anchor synergy recognition."""
        suit = evaluate_location_suitability(
            realized_daily_bottles=550.0,
            unconstrained_daily_bottles=550.0,
            nearest_coripet_km=3.0,
            nearest_pf_name="Coop",
            sales_area_sqm=2500.0,
            numero_casse=12.0,
            shopping_centers_count=1
        )
        self.assertTrue(any("Mall Anchor" in s for s in suit["key_strengths"]))
        self.assertGreaterEqual(suit["suitability_score"], 95)


if __name__ == "__main__":
    unittest.main()

