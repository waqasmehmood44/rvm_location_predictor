import pandas as pd

df_stores = pd.read_csv('data/raw/CaricaViewStoricoPuntoFisico.csv', sep=';', low_memory=False)
df_sheet3 = pd.read_excel('data/raw/Premialità-Rewards.xlsx', sheet_name='Foglio3')

sheet3_codici = set(df_sheet3['CODICE'].dropna().astype(str).str.strip().str.upper())
sheet3_nomi = set(df_sheet3['NOME'].dropna().astype(str).str.strip().str.upper())

df_stores['fil_clean'] = df_stores['Filiera'].fillna('').astype(str).str.strip().str.upper()
df_stores['nomefil_clean'] = df_stores['NomeFiliera'].fillna('').astype(str).str.strip().str.upper()

matched_mask = (df_stores['fil_clean'].isin(sheet3_codici)) | (df_stores['nomefil_clean'].isin(sheet3_nomi))
print("Matched stores count:", matched_mask.sum(), f"({matched_mask.mean()*100:.1f}%)")

unmatched_stores = df_stores[~matched_mask]
print("\nUnmatched Filiera value counts (top 15):")
print(unmatched_stores['Filiera'].value_counts(dropna=False).head(15))
