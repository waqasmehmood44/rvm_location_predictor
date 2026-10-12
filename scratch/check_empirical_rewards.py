import pandas as pd
import numpy as np

# Load store metadata
df_stores = pd.read_csv('data/raw/CaricaViewStoricoPuntoFisico.csv', sep=';', usecols=['PuntoFisico', 'Filiera', 'NomeFiliera'], low_memory=False)
df_stores['PuntoFisico'] = df_stores['PuntoFisico'].astype(str).str.strip().str.upper()
df_stores['Filiera'] = df_stores['Filiera'].fillna('').astype(str).str.strip().str.upper()

# Load Sheet 3
df_sheet3 = pd.read_excel('data/raw/Premialità-Rewards.xlsx', sheet_name='Foglio3')
df_sheet3['CODICE_UP'] = df_sheet3['CODICE'].astype(str).str.strip().str.upper()
df_sheet3['NOME_UP'] = df_sheet3['NOME'].astype(str).str.strip().str.upper()

# Group by PuntoFisico in modeling_daily_rvm.parquet
df_model = pd.read_parquet('data/processed/modeling_daily_rvm.parquet', columns=['PuntoFisico', 'BOTTIGLIE', 'PRESENZE'])
pf_stats = df_model.groupby('PuntoFisico').agg(
    mean_bottles=('BOTTIGLIE', 'mean'),
    median_bottles=('BOTTIGLIE', 'median'),
    mean_presenze=('PRESENZE', 'mean'),
    median_presenze=('PRESENZE', 'median'),
    days_count=('BOTTIGLIE', 'count')
).reset_index()

pf_stats['PuntoFisico'] = pf_stats['PuntoFisico'].astype(str).str.strip().str.upper()

# Merge
merged = pf_stats.merge(df_stores, on='PuntoFisico', how='inner')
merged = merged.merge(df_sheet3, left_on='Filiera', right_on='CODICE_UP', how='left')

print("Total unique RVMs with performance stats:", len(pf_stats))
print("Matched with Sheet 3:", merged['valore_punti'].notna().sum())

# Group by points required
pts_group = merged[merged['valore_punti'].notna()].groupby('valore_punti').agg(
    stores_count=('PuntoFisico', 'nunique'),
    avg_daily_bottles=('mean_bottles', 'mean'),
    median_daily_bottles=('median_bottles', 'median'),
    avg_daily_presenze=('mean_presenze', 'mean'),
    median_daily_presenze=('median_presenze', 'median')
).reset_index()

pts_group = pts_group[pts_group['stores_count'] >= 5].sort_values(by='stores_count', ascending=False)
print("\n=== PERFORMANCE BY VALORE PUNTI (Bottles to Voucher) ===")
print(pts_group.to_string(index=False))
