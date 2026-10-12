"""End-to-end inference engine for Coripet prospective location evaluations."""

from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import torch

from src.features.geocoding import LocationEnricher
from src.features.store_features import StoreFeatureExtractor
from src.features.preprocessor import TabularPreprocessor
from src.models.neural_net import CoripetMultiTaskNN
from src.business.scoring import SuitabilityScorer
from src.business.recommender import PotentialRecommender
from src.business.confidence import ConfidenceAssessor


class LocationPredictor:
    """Predicts daily presenze, bottle counts, suitability, and potential for prospective sites."""

    def __init__(
        self,
        model: Optional[CoripetMultiTaskNN] = None,
        preprocessor: Optional[TabularPreprocessor] = None,
        enricher: Optional[LocationEnricher] = None,
        feature_extractor: Optional[StoreFeatureExtractor] = None,
        scorer: Optional[SuitabilityScorer] = None,
        recommender: Optional[PotentialRecommender] = None,
        confidence_assessor: Optional[ConfidenceAssessor] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device) if model is not None else None
        self.preprocessor = preprocessor
        self.enricher = enricher or LocationEnricher()
        self.feature_extractor = feature_extractor or StoreFeatureExtractor()
        self.scorer = scorer or SuitabilityScorer()
        self.recommender = recommender or PotentialRecommender()
        self.confidence_assessor = confidence_assessor or ConfidenceAssessor()

    @classmethod
    def from_checkpoints(
        cls,
        model_checkpoint_path: str | Path,
        preprocessor_path: str | Path,
        business_rules_path: str = "configs/business_rules.yaml",
        device: Optional[str] = None,
    ) -> "LocationPredictor":
        """Factory method loading trained neural network weights and fitted preprocessor."""
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load fitted preprocessor
        prep = TabularPreprocessor.load(preprocessor_path)

        # Reconstruct neural network architecture from preprocessor dimensions
        emb_dims = [
            (prep.cat_dims[col], prep.emb_dims[col])
            for col in prep.categorical_features
            if col in prep.cat_dims
        ]
        num_continuous = len(prep.numerical_features)

        model = CoripetMultiTaskNN(
            emb_dims=emb_dims,
            num_continuous_features=num_continuous,
        )

        checkpoint = torch.load(model_checkpoint_path, map_location=dev)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        return cls(
            model=model,
            preprocessor=prep,
            enricher=LocationEnricher(),
            feature_extractor=StoreFeatureExtractor(),
            scorer=SuitabilityScorer(config_path=business_rules_path),
            recommender=PotentialRecommender(config_path=business_rules_path),
            confidence_assessor=ConfidenceAssessor(config_path=business_rules_path),
            device=dev,
        )

    def predict_location(
        self,
        store_name: str,
        address: str,
        postal_code: str,
        machine_type: str,
    ) -> Dict[str, Any]:
        """Runs the complete inference and business evaluation pipeline."""
        # 1. Feature extraction from realistic inputs
        store_feats = self.feature_extractor.transform_inference_input(
            store_name=store_name, machine_type=machine_type
        )
        geo_feats = self.enricher.enrich(
            postal_code=postal_code, address=address
        )

        input_record = {
            **store_feats,
            **geo_feats,
            "store_name": store_name,
            "address": address,
        }
        input_df = pd.DataFrame([input_record])

        # 2. Tabular transformation
        if self.preprocessor is not None and self.preprocessor.is_fitted:
            x_cat, x_cont = self.preprocessor.transform(input_df)
            x_cat_t = torch.tensor(x_cat, dtype=torch.long, device=self.device)
            x_cont_t = torch.tensor(x_cont, dtype=torch.float32, device=self.device)

            # 3. Model forward pass
            if self.model is not None:
                self.model.eval()
                with torch.no_grad():
                    pred_p_t, pred_b_t = self.model(x_cat_t, x_cont_t)
                    pred_presenze = float(pred_p_t.cpu().item())
                    pred_bottiglie = float(pred_b_t.cpu().item())
            else:
                pred_presenze, pred_bottiglie = 0.0, 0.0
        else:
            # Baseline placeholder when model is not yet loaded
            pred_presenze, pred_bottiglie = 0.0, 0.0

        # 4. Business metric derivation
        metrics = self.scorer.calculate_metrics(pred_presenze, pred_bottiglie)

        # 5. Recommendation tier
        rec_data = self.recommender.recommend(metrics["suitability_score_pct"])

        # 6. Confidence evaluation
        cap_known = False
        chain_known = False
        machine_known = False
        if self.preprocessor is not None and self.preprocessor.is_fitted:
            cap_known = str(geo_feats.get("cap")) in self.preprocessor.cat_to_idx.get("cap", {})
            chain_known = store_feats.get("retail_chain") in self.preprocessor.cat_to_idx.get(
                "retail_chain", {}
            )
            machine_known = store_feats.get("tipo_macchina") in self.preprocessor.cat_to_idx.get(
                "tipo_macchina", {}
            )

        conf_data = self.confidence_assessor.assess_confidence(
            is_postal_code_known=cap_known,
            is_retail_chain_known=chain_known,
            is_machine_type_known=machine_known,
        )

        formatted_report = (
            "LOCATION ANALYSIS\n\n"
            f"Expected PRESENZE/day: {int(round(metrics['expected_presenze_day']))}\n"
            f"Expected BOTTIGLIE/day: {int(round(metrics['expected_bottiglie_day']))}\n"
            f"Expected BOTTIGLIE/month: {int(round(metrics['expected_bottiglie_month'])):,}\n"
            f"Bottles per person: {metrics['bottles_per_person']:.1f}\n"
            f"Location suitability: {int(round(metrics['suitability_score_pct']))}%\n"
            f"Recommendation: {rec_data['recommendation']}\n"
            f"Confidence: {conf_data['confidence']}"
        )

        return {
            "inputs": {
                "store_name": store_name,
                "address": address,
                "postal_code": postal_code,
                "machine_type": machine_type,
            },
            "enriched_features": input_record,
            "metrics": metrics,
            "recommendation": rec_data,
            "confidence": conf_data,
            "formatted_text": formatted_report,
        }
