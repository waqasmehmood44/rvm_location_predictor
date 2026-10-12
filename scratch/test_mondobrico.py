import sys
import os
sys.path.insert(0, os.path.abspath("."))

from src.model.predict import RVMNewLocationPredictor, print_prediction_summary

predictor = RVMNewLocationPredictor()
result = predictor.predict_new_location(
    store_name="Mondobrico",
    address="Via Ezio Vanoni, 18",
    cap="15057",
    comune="Tortona",
    machine_type="Ripet_C-Ter",
    sales_area_sqm=2500.0,
    cash_registers=8.0
)

print_prediction_summary(result)
