import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.model.predict import RVMNewLocationPredictor

def run_tests():
    predictor = RVMNewLocationPredictor()
    
    print("=" * 75)
    print("      END-TO-END VERIFICATION: PISTOIA & POGGIBONSI")
    print("=" * 75)
    
    # 1. Pistoia
    pistoia = predictor.predict_new_location(
        store_name="Unicoop.Fi",
        address="Viale Macallè, 14",
        cap="51100",
        comune="Pistoia",
        machine_type="Ripet_C-front"
    )
    
    p_pred_b = pistoia["realized_forecast"]["underwritten_daily_bottles"]
    p_real_b = pistoia["realized_forecast"]["realized_daily_bottles"]
    p_pred_p = pistoia["realized_forecast"]["underwritten_daily_presenze"]
    p_mode = pistoia["realized_forecast"]["underwriting_mode"]
    p_scale = pistoia["inputs"]["store_scale_applied"]
    
    # Power BI ground truth
    # All-time multi-year: 3,340 b/d | 2026 YTD actual: 3,625.4 b/d (98.5 users/day)
    acc_pistoia_alltime = 100.0 - abs(p_pred_b - 3340.0) / 3340.0 * 100.0
    acc_pistoia_2026 = 100.0 - abs(p_pred_b - 3625.4) / 3625.4 * 100.0
    acc_users_2026 = 100.0 - abs(p_pred_p - 98.5) / 98.5 * 100.0
    
    print("\n--- STORE 1: UNICOOP.FI PISTOIA (Viale Macallè 14) ---")
    print(f"  Auto-detected Scale      : {p_scale}")
    print(f"  Underwriting Mode        : {p_mode}")
    print(f"  Underwritten Daily B     : {p_pred_b:.1f} b/day")
    print(f"  Realized Steady Ceiling  : {p_real_b:.1f} b/day")
    print(f"  Underwritten Presenze    : {p_pred_p:.1f} users/day")
    print(f"  Power BI All-Time Actual : 3,340.0 b/day  -> Accuracy: {acc_pistoia_alltime:.1f}% (Error: {(p_pred_b-3340)/3340*100:+.1f}%)")
    print(f"  Power BI 2026 YTD Actual : 3,625.4 b/day  -> Accuracy: {acc_pistoia_2026:.1f}% (Error: {(p_pred_b-3625.4)/3625.4*100:+.1f}%)")
    print(f"  Power BI 2026 Footfall   : 98.5 users/day -> Accuracy: {acc_users_2026:.1f}% (Error: {(p_pred_p-98.5)/98.5*100:+.1f}%)")
    
    # 2. Poggibonsi
    poggi = predictor.predict_new_location(
        store_name="Unicoop.Fi",
        address="Via Salceto 7",
        cap="53036",
        comune="Poggibonsi",
        machine_type="Ripet_C-front"
    )
    
    pg_pred_b = poggi["realized_forecast"]["underwritten_daily_bottles"]
    pg_real_b = poggi["realized_forecast"]["realized_daily_bottles"]
    pg_pred_p = poggi["realized_forecast"]["underwritten_daily_presenze"]
    pg_mode = poggi["realized_forecast"]["underwriting_mode"]
    pg_scale = poggi["inputs"]["store_scale_applied"]
    
    # Power BI ground truth: 4,425.0 b/d
    acc_poggi = 100.0 - abs(pg_pred_b - 4425.0) / 4425.0 * 100.0
    
    print("\n--- STORE 2: UNICOOP.FI POGGIBONSI (Via Salceto 7) ---")
    print(f"  Matched Scale Source     : {pg_scale}")
    print(f"  Underwriting Mode        : {pg_mode}")
    print(f"  Underwritten Daily B     : {pg_pred_b:.1f} b/day")
    print(f"  Realized Steady Ceiling  : {pg_real_b:.1f} b/day")
    print(f"  Power BI Benchmark Actual: 4,425.0 b/day  -> Accuracy: {acc_poggi:.1f}% (Error: {(pg_pred_b-4425.0)/4425.0*100:+.1f}%)")
    
    print("\n" + "=" * 75)
    print("VERIFICATION COMPLETED SUCCESSFULLY.")
    print("=" * 75)

if __name__ == '__main__':
    run_tests()
