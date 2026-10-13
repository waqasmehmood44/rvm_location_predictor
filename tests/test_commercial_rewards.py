"""
Unit tests verifying Commercial Rewards & Filiera Intelligence integration.
Ensures:
1. Exact 100% Filiera matching with Premialità-Rewards.xlsx
2. Accurate mapping to Sheet 2 reward definitions
3. Safe default fallbacks for greenfield prospective locations
4. Seamless integration with LocationPredictor without regression
"""

import unittest
from src.business.rewards import CommercialRewardsEngine, REWARD_DEFINITIONS
from src.model.predict import RVMNewLocationPredictor


class TestCommercialRewards(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = CommercialRewardsEngine()

    def test_engine_loaded(self):
        self.assertTrue(self.engine.is_loaded, "CommercialRewardsEngine should load successfully.")
        self.assertGreater(len(self.engine.filiera_code_to_reward), 400)
        self.assertGreater(len(self.engine.punto_fisico_to_filiera), 1500)

    def test_exact_code_matching(self):
        # Test national standard voucher
        res_conad = self.engine.resolve_reward(filiera="CONAD_ITALIA")
        self.assertEqual(res_conad["reward_type"], "Emissione Buono sconto")
        self.assertEqual(res_conad["incentive_tier"], "GROCERY_VOUCHER")
        self.assertIn("3€", res_conad["typical_voucher"])

        # Test cooperative loyalty conversion
        res_coop = self.engine.resolve_reward(filiera="Unicoop.Fi")
        self.assertEqual(res_coop["reward_type"], "Conversione a punti fedeltà")
        # Test Esselunga Omaggio
        res_ess_code = self.engine.resolve_reward(filiera="ESSELUNGA")
        self.assertEqual(res_ess_code["reward_type"], "Omaggio")
        self.assertEqual(res_ess_code["incentive_tier"], "DIRECT_GIFT")

    def test_greenfield_fallbacks(self):
        # Esselunga greenfield fallback policy (Direct Gift / Omaggio)
        res_ess = self.engine.resolve_reward(store_name="Esselunga Superstore", retail_group="ESSELUNGA SPA")
        self.assertEqual(res_ess["reward_type"], "Omaggio")
        self.assertEqual(res_ess["incentive_tier"], "DIRECT_GIFT")

        # Brand with GDO standard voucher
        res_carrefour = self.engine.resolve_reward(store_name="Carrefour Market", retail_group="CARREFOUR")
        self.assertEqual(res_carrefour["reward_type"], "Emissione Buono sconto")

        # Cooperative in Toscana
        res_coop_fi = self.engine.resolve_reward(store_name="Coop.fi", regione="TOSCANA")
        self.assertEqual(res_coop_fi["reward_type"], "Conversione a punti fedeltà")

        # Municipal site falls back to standard voucher
        res_muni = self.engine.resolve_reward(store_name="Comune di Milano - Piazza Duomo")
        self.assertEqual(res_muni["reward_type"], "Emissione Buono sconto")

    def test_sheet3_discount_and_points_mechanics(self):
        # 1. Bennet contract
        res_bennet = self.engine.resolve_reward(filiera="Bennet")
        self.assertEqual(res_bennet["points_required"], 200)
        self.assertEqual(res_bennet["discount_euro"], 3.0)
        self.assertEqual(res_bennet["min_spend_euro"], 30.0)
        self.assertEqual(res_bennet["valore_cent_per_bottiglia"], 1.5)
        self.assertEqual(res_bennet["sconto_percentuale_carrello"], 10.0)
        self.assertEqual(res_bennet["multiplier_bottles"], 1.0)
        self.assertEqual(res_bennet["multiplier_presenze"], 1.0)

        # 2. Unicoop.Fi contract (90 points fidelity card conversion)
        res_unicoop = self.engine.resolve_reward(filiera="Unicoop.Fi")
        self.assertEqual(res_unicoop["points_required"], 90)
        self.assertGreater(res_unicoop["multiplier_presenze"], 1.1)

        # 3. CONAD_ITALIA contract
        res_conad = self.engine.resolve_reward(filiera="CONAD_ITALIA")
        self.assertEqual(res_conad["points_required"], 200)
        self.assertEqual(res_conad["discount_euro"], 3.0)
        self.assertIn("Coupon", res_conad["contract_title"])

    def test_predictor_integration(self):
        predictor = RVMNewLocationPredictor()
        res = predictor.predict_new_location(
            store_name="CONAD",
            address="Via Postumia, 15",
            cap="35121",
            machine_type="Ripet_C-front",
            comune="Padova"
        )
        self.assertIn("commercial_rewards", res)
        self.assertEqual(res["commercial_rewards"]["reward_type"], "Emissione Buono sconto")
        self.assertEqual(res["commercial_rewards"]["incentive_tier"], "GROCERY_VOUCHER")
        self.assertIn("points_required", res["commercial_rewards"])
        self.assertIn("valore_cent_per_bottiglia", res["commercial_rewards"])
        self.assertIn("reward_sensitivity_analysis", res)
        selected_items = [item for item in res["reward_sensitivity_analysis"] if item.get("is_selected")]
        self.assertEqual(len(selected_items), 1)
        self.assertEqual(selected_items[0]["reward_type"], "Emissione Buono sconto")


if __name__ == "__main__":
    unittest.main()
