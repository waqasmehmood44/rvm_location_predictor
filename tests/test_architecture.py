"""Unit tests verifying modular components, pipeline contracts, and model architectures."""

import unittest
import numpy as np
import pandas as pd
import torch

from src.data.cleaner import DataCleaner
from src.data.splitter import TimeAndGroupSplitter
from src.features.geocoding import LocationEnricher
from src.features.store_features import StoreFeatureExtractor
from src.features.preprocessor import TabularPreprocessor
from src.models.neural_net import CoripetMultiTaskNN
from src.business.scoring import SuitabilityScorer
from src.business.recommender import PotentialRecommender
from src.business.confidence import ConfidenceAssessor
from src.inference.predictor import LocationPredictor


class TestArchitectureAndPipeline(unittest.TestCase):

    def test_data_cleaner_cap_and_targets(self):
        cleaner = DataCleaner()
        raw_df = pd.DataFrame({
            "GIORNO": ["2024-01-01", "2024-01-02", "invalid_date"],
            "PRESENZE": [25, -5, 30],
            "BOTTIGLIE": [700, 100, 850],
            "CAP": [20147, "185", "corrupt_cap"],
            "Latitudine": ["45,4642", "41,9028", None],
            "Longitudine": ["9,1900", "12,4964", None],
        })

        cleaned = cleaner.clean_dataset(raw_df)
        # Negative target and invalid date should be removed
        self.assertEqual(len(cleaned), 1)
        row = cleaned.iloc[0]
        self.assertEqual(row["cap"], "20147")
        self.assertAlmostEqual(row["latitudine"], 45.4642, places=4)
        self.assertAlmostEqual(row["longitudine"], 9.1900, places=4)


    def test_time_and_group_splitter(self):
        splitter = TimeAndGroupSplitter(date_column="GIORNO", store_key="PF")
        dates = pd.date_range("2023-01-01", periods=200, freq="D")
        df = pd.DataFrame({
            "GIORNO": dates,
            "PF": [f"STORE_{i % 10}" for i in range(200)],
            "PRESENZE": np.random.randint(10, 50, size=200),
            "BOTTIGLIE": np.random.randint(200, 1000, size=200),
        })

        train_df, val_df, test_df = splitter.temporal_train_test_split(df, test_days=30, val_days=20)
        self.assertTrue(train_df["GIORNO"].max() < val_df["GIORNO"].min())
        self.assertTrue(val_df["GIORNO"].max() < test_df["GIORNO"].min())

        # Group store split test
        tr_group, val_group, te_group = splitter.group_store_train_test_split(df, test_ratio=0.2, val_ratio=0.2)
        stores_train = set(tr_group["PF"].unique())
        stores_test = set(te_group["PF"].unique())
        # No store overlap between train and test
        self.assertEqual(len(stores_train.intersection(stores_test)), 0)

    def test_location_enricher_and_store_extractor(self):
        enricher = LocationEnricher()
        extractor = StoreFeatureExtractor()

        # CAP prefix to region
        res = enricher.enrich(postal_code="20147")
        self.assertEqual(res["provincia"], "MI")
        self.assertEqual(res["regione"], "LOMBARDIA")

        res_rome = enricher.enrich(postal_code="00185")
        self.assertEqual(res_rome["provincia"], "RM")
        self.assertEqual(res_rome["regione"], "LAZIO")

        # Brand extraction
        chain = extractor.extract_retail_chain("Esselunga Superstore Baranzate")
        self.assertEqual(chain, "ESSELUNGA")

        chain_coop = extractor.extract_retail_chain("Ipercoop Bonola")
        self.assertEqual(chain_coop, "IPERCOOP")

    def test_neural_network_forward(self):
        # 3 categorical features with cardinalities [10, 5, 20], 2 continuous features
        emb_dims = [(10, 6), (5, 4), (20, 10)]
        model = CoripetMultiTaskNN(emb_dims=emb_dims, num_continuous_features=2)
        model.eval()

        batch_size = 4
        x_cat = torch.randint(0, 5, (batch_size, 3), dtype=torch.long)
        x_cont = torch.randn(batch_size, 2, dtype=torch.float32)

        with torch.no_grad():
            pred_p, pred_b = model(x_cat, x_cont)

        self.assertEqual(pred_p.shape, (batch_size, 1))
        self.assertEqual(pred_b.shape, (batch_size, 1))
        # Output must be non-negative (via Softplus)
        self.assertTrue((pred_p >= 0).all())
        self.assertTrue((pred_b >= 0).all())

    def test_business_rules_and_scoring(self):
        scorer = SuitabilityScorer(config_path=None)
        recommender = PotentialRecommender(config_path=None)
        confidence = ConfidenceAssessor(config_path=None)

        metrics = scorer.calculate_metrics(pred_presenze=25.0, pred_bottiglie=735.0)
        self.assertEqual(metrics["expected_presenze_day"], 25.0)
        self.assertEqual(metrics["expected_bottiglie_day"], 735.0)
        self.assertEqual(metrics["expected_bottiglie_month"], 22050.0)
        self.assertEqual(metrics["bottles_per_person"], 29.4)
        self.assertTrue(0 <= metrics["suitability_score_pct"] <= 100)

        rec = recommender.recommend(metrics["suitability_score_pct"])
        self.assertIn(rec["recommendation"], ["HIGH POTENTIAL", "MEDIUM POTENTIAL", "LOW POTENTIAL"])

        conf = confidence.assess_confidence(
            is_postal_code_known=True,
            is_retail_chain_known=True,
            is_machine_type_known=True,
        )
        self.assertIn(conf["confidence"], ["HIGH", "MEDIUM", "LOW"])

    def test_end_to_end_predictor_pipeline(self):
        predictor = LocationPredictor()
        result = predictor.predict_location(
            store_name="Esselunga Milano Certosa",
            address="Via delle Forze Armate 12",
            postal_code="20147",
            machine_type="Aneco Super",
        )

        self.assertIn("LOCATION ANALYSIS", result["formatted_text"])
        self.assertIn("Expected PRESENZE/day", result["formatted_text"])
        self.assertIn("Expected BOTTIGLIE/day", result["formatted_text"])
        self.assertIn("Expected BOTTIGLIE/month", result["formatted_text"])
        self.assertIn("Bottles per person", result["formatted_text"])
        self.assertIn("Location suitability", result["formatted_text"])
        self.assertIn("Recommendation", result["formatted_text"])
        self.assertIn("Confidence", result["formatted_text"])
