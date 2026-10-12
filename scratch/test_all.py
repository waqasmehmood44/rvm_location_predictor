import sys
sys.path.insert(0, ".")

from src.model.predict import RVMNewLocationPredictor

p = RVMNewLocationPredictor()
test_stores = [
    ("Unicoop.Fi - Prato", "Via delle Pleiadi", "59100", "Prato", "Ripet_C-front", 5139.0, 38.0, "EXPLICIT_OVERRIDE", "Conversione a punti fedeltà"),
    ("ESSELUNGA - Cusano Milanino", "Viale Unione, 2", "20095", "Cusano Milanino", "DAINT_A-bis", 2500.0, 41.0, "EXPLICIT_OVERRIDE", "Omaggio"),
    ("Conad e Todis (Roma - Via Poppea)", "Via Poppea Sabina, 19", "00118", "Roma", "Ripet_V2", 200.0, 2.0, "EXPLICIT_OVERRIDE", "Emissione Buono sconto"),
    ("Conad TuDay - Pesaro Via Novecento", "Via del Novecento", "61122", "Pesaro", "Ripet_C-front", 443.0, 3.0, "STANDARD_NATIONAL", "Emissione Buono sconto")
]

for name, addr, cap, com, m, sqm, cks, fil, rew in test_stores:
    res = p.predict_new_location(
        store_name=name, address=addr, cap=cap, comune=com,
        machine_type=m, sales_area_sqm=sqm, cash_registers=cks,
        filiera=fil, explicit_reward=rew
    )
    b = res["predictions"]["expected_daily_bottiglie"]
    u = res["predictions"]["expected_daily_presenze"]
    print(f"{name:36} -> Bottles: {b:6.1f} b/d | Users: {u:4.1f} u/d")
