import pandas as pd
import re

df = pd.read_excel('data/raw/Premialità-Rewards.xlsx', sheet_name='Foglio3')
print("Total rows in Foglio3:", len(df))
print("Unique NOME values:", df['NOME'].nunique())
print("Unique CODICE values:", df['CODICE'].nunique())
print("\nDESCRIZIONE value counts:")
print(df['DESCRIZIONE'].value_counts())

print("\nvalore_punti top values:")
print(df['valore_punti'].value_counts().head(10))

print("\nconversione top values:")
print(df['conversione'].value_counts().head(10))

print("\nSample 25 titles and descriptions:")
for idx, r in df.sample(25, random_state=42).iterrows():
    nome = r['NOME']
    tit = str(r['titolo']).strip()
    desc1 = str(r['descrizione.1']).strip()[:80]
    pts = r['valore_punti']
    conv = r['conversione']
    print(f"[{nome}] | {tit} | {desc1} | {pts} pts | conv={conv}")
