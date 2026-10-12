"""
Quick lookup tool to search any store from the 331 test holdout set by Descrizione, Comune, or PF.
Usage:
    python scripts/search_store.py "Aprilia"
    python scripts/search_store.py "Mercatò"
"""
import sys
import pandas as pd

def search(query: str):
    csv_path = "reports/holdout_store_predictions_331.csv"
    df = pd.read_csv(csv_path)
    
    q = query.strip().lower()
    mask = (
        df["Descrizione"].str.lower().str.contains(q, na=False) |
        df["PF"].str.lower().str.contains(q, na=False) |
        df["comune"].str.lower().str.contains(q, na=False) |
        df["store_name"].str.lower().str.contains(q, na=False)
    )
    res = df[mask]
    
    if res.empty:
        print(f"No stores found matching '{query}'.")
        return
        
    print(f"\nFound {len(res)} store(s) matching '{query}':\n")
    cols = ["Descrizione", "PF", "comune", "machine_type", "actual_mean_b", "ens_pred_mean_b", "ens_pct_err_b", "ens_accuracy_pct"]
    print(res[cols].to_string(index=False))
    print("\n" + "=" * 90)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Conad"
    search(query)
