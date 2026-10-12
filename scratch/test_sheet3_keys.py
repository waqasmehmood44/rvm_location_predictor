import os
import sys
import pandas as pd
import numpy as np

# Load Sheet 3
df_sheet3 = pd.read_excel('data/raw/Premialità-Rewards.xlsx', sheet_name='Foglio3')
print(f"Loaded {len(df_sheet3)} rows from Foglio3")

# Let's inspect rows for CONAD_ITALIA, Bennet, Unicoop.Fi
test_keys = ['CONAD_ITALIA', 'Bennet', 'Unicoop.Fi', 'Dimar', 'PAM_Decaro', 'Decò']
for k in test_keys:
    row = df_sheet3[df_sheet3['CODICE'].astype(str).str.upper() == k.upper()]
    if len(row) > 0:
        r = row.iloc[0]
        print(f"\n[{k}] -> NOME: {r['NOME']} | DESC: {r['DESCRIZIONE']} | tit: {r['titolo']} | desc1: {r['descrizione.1']} | pts: {r['valore_punti']} | conv: {r['conversione']}")
    else:
        print(f"\n[{k}] NOT FOUND in CODICE, searching NOME...")
        row_n = df_sheet3[df_sheet3['NOME'].astype(str).str.upper() == k.upper()]
        if len(row_n) > 0:
            r = row_n.iloc[0]
            print(f"[{k}] (via NOME) -> CODICE: {r['CODICE']} | tit: {r['titolo']} | pts: {r['valore_punti']}")
        else:
            print(f"[{k}] NOT FOUND")
