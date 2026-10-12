import sys
sys.path.insert(0, ".")

from src.model.scoring import calculate_catchment_realization_factor

# Test 1: with TuDay in name
res_tuday = calculate_catchment_realization_factor(
    unconstrained_daily_bottles=520.0,
    unconstrained_daily_presenze=19.0,
    store_name="TuDay Conad - Pesaro Via Novecento",
    nearest_coripet_km=0.77,
    nearest_pf_name="",
    coripet_count_1km=1,
    coripet_count_3km=2,
    coripet_count_5km=3,
    grocery_total_stores=12,
    network_position="COMPETITIVE_TRADE_AREA",
    sales_area_sqm=443.0,
    numero_casse=3.0,
    is_independent_banner=False,
    nielsen_area="Area 3.0",
    population_density=754.7,
    population=95670.0,
    regione="Marche",
    machine_type="Ripet_C-front"
)

print("TuDay Realized Bottles:", res_tuday["realized_daily_bottles"])
print("TuDay Dilution %:", res_tuday["dilution_percentage"])
print("TuDay Reasons:", res_tuday["dilution_reasons"])
