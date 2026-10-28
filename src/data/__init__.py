"""Data loading, cleaning, historical matching, Nielsen linkage, municipality integration, and validation modules."""

from src.data.load_data import (
    load_daily_bottles,
    load_storico_punto_fisico,
    load_filenielsen,
    load_macro_datasets,
)
from src.data.clean_data import (
    clean_postal_code,
    clean_comune_name,
    clean_decimal_number,
    clean_address_string,
)
from src.data.historical_physical_point import (
    enrich_daily_bottles_with_history,
    analyze_physical_point_history,
)
from src.data.nielsen_matching import match_physical_points_to_nielsen
from src.data.municipality_matching import (
    build_municipality_feature_store,
    enrich_with_municipality_data,
)
from src.data.integrate_data import run_full_integration_pipeline
from src.data.validate_data import validate_master_dataset

__all__ = [
    "load_daily_bottles",
    "load_storico_punto_fisico",
    "load_filenielsen",
    "load_macro_datasets",
    "clean_postal_code",
    "clean_comune_name",
    "clean_decimal_number",
    "clean_address_string",
    "enrich_daily_bottles_with_history",
    "analyze_physical_point_history",
    "match_physical_points_to_nielsen",
    "build_municipality_feature_store",
    "enrich_with_municipality_data",
    "run_full_integration_pipeline",
    "validate_master_dataset",
]
