import pandas as pd
import numpy as np

df = pd.read_parquet('models/preprocessing/test_holdout.parquet')
df['dt'] = pd.to_datetime(df['GIORNO'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
if df['dt'].isna().sum() > len(df) * 0.5:
    df['dt'] = pd.to_datetime(df['GIORNO'], errors='coerce')
df['year'] = df['dt'].dt.year

# Check installation / activation year
if 'DataPrimaAttivazione' in df.columns:
    df['activation_dt'] = pd.to_datetime(df['DataPrimaAttivazione'], errors='coerce')
    df['activation_year'] = df['activation_dt'].dt.year
else:
    df['activation_year'] = np.nan

# 1. PFs installed in 2024 or 2025
pf_installed_24_25 = df[df['activation_year'].isin([2024, 2025])]['PF'].unique()
print(f"Total stores installed in 2024 or 2025: {len(pf_installed_24_25)}")
print(f"   - Installed in 2024: {(df.groupby('PF')['activation_year'].first() == 2024).sum()}")
print(f"   - Installed in 2025: {(df.groupby('PF')['activation_year'].first() == 2025).sum()}")

# 2. PFs active with telemetry in 2024 or 2025
pf_active_24_25 = df[df['year'].isin([2024, 2025])]['PF'].unique()
print(f"\nTotal stores active with telemetry in 2024 or 2025: {len(pf_active_24_25)}")
print(f"   - Active in 2024: {len(df[df['year'] == 2024]['PF'].unique())}")
print(f"   - Active in 2025: {len(df[df['year'] == 2025]['PF'].unique())}")

# Compute 2024 actuals, 2025 actuals, and combined 2024-2025 actuals
b_2024 = df[df['year'] == 2024].groupby('PF')['BOTTIGLIE'].mean()
p_2024 = df[df['year'] == 2024].groupby('PF')['PRESENZE'].mean()

b_2025 = df[df['year'] == 2025].groupby('PF')['BOTTIGLIE'].mean()
p_2025 = df[df['year'] == 2025].groupby('PF')['PRESENZE'].mean()

b_24_25 = df[df['year'].isin([2024, 2025])].groupby('PF')['BOTTIGLIE'].mean()
p_24_25 = df[df['year'].isin([2024, 2025])].groupby('PF')['PRESENZE'].mean()

print("\nTop 10 stores active in 2024/2025 with actuals:")
top_pfs = b_24_25.sort_values(ascending=False).head(10).index
for pf in top_pfs:
    desc = df[df['PF'] == pf]['Descrizione'].iloc[0]
    act_yr = df[df['PF'] == pf]['activation_year'].iloc[0]
    print(f"PF: {pf} | ActYear: {act_yr} | {desc[:35]:35s} | 2024: {b_2024.get(pf, np.nan):6.1f} | 2025: {b_2025.get(pf, np.nan):6.1f} | Avg: {b_24_25.get(pf):6.1f}")
