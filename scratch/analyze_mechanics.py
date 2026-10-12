import pandas as pd
import re

df = pd.read_excel('data/raw/Premialità-Rewards.xlsx', sheet_name='Foglio3')

print("=== CHECK CONVERSIONE ===")
for c, group in df.groupby('conversione'):
    print(f"\n--- conversione = {c} (count={len(group)}) ---")
    print(group[['NOME', 'DESCRIZIONE', 'titolo', 'valore_punti']].head(6))

print("\n=== PARSING DISCOUNT AMOUNTS & MIN SPEND ===")
patterns = [
    r'(\d+(?:[.,]\d+)?)\s*€?\s*(?:di\s+sconto|euro|sconto)',
    r'sconto\s*(?:di\s*)?(\d+(?:[.,]\d+)?)\s*€?',
]

results = []
for idx, r in df.iterrows():
    text = str(r['descrizione.1']) + " " + str(r['titolo'])
    pts = r['valore_punti']
    
    # discount euros
    disc_m = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:€|euro)\s*(?:di\s*)?sconto', text, re.IGNORECASE)
    if not disc_m:
        disc_m = re.search(r'sconto\s*(?:di\s*)?(\d+(?:[.,]\d+)?)\s*(?:€|euro)', text, re.IGNORECASE)
    if not disc_m:
        disc_m = re.search(r'(\d+(?:[.,]\d+)?)\s*€', str(r['titolo']), re.IGNORECASE)
        
    disc = float(disc_m.group(1).replace(',', '.')) if disc_m else None
    
    # min spend
    min_m = re.search(r'(?:spesa\s+minima|minimo\s+di\s+spesa|minimo\s+scontrino)\s*(?:di\s*)?(\d+(?:[.,]\d+)?)\s*(?:€|euro)?', text, re.IGNORECASE)
    if not min_m:
        min_m = re.search(r'su\s+(\d+(?:[.,]\d+)?)\s*(?:€|euro)\s*di\s*spesa', text, re.IGNORECASE)
        
    min_spend = float(min_m.group(1).replace(',', '.')) if min_m else None
    
    results.append({
        'nome': r['NOME'],
        'descrizione': r['DESCRIZIONE'],
        'titolo': r['titolo'],
        'pts': pts,
        'disc_euro': disc,
        'min_spend': min_spend,
        'conv': r['conversione']
    })

res_df = pd.DataFrame(results)
print("\nExtraction stats:")
print("Total rows:", len(res_df))
print("Discount extracted:", res_df['disc_euro'].notna().sum())
print("Min spend extracted:", res_df['min_spend'].notna().sum())

print("\nTop Discount amounts:")
print(res_df['disc_euro'].value_counts().head(10))

print("\nTop Min Spend amounts:")
print(res_df['min_spend'].value_counts().head(10))

print("\nRows where min_spend or discount missing:")
missing = res_df[res_df['disc_euro'].isna() | res_df['min_spend'].isna()]
print(f"Count missing: {len(missing)}")
for idx, r in missing.head(15).iterrows():
    print(f"[{r['nome']}] | {r['descrizione']} | tit: {r['titolo']} | pts: {r['pts']}")
