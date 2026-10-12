import pandas as pd

df = pd.read_parquet('data/processed/master_daily_rvm.parquet')
sub = df[df['PF'] == 'PF00002012'].copy()
sub['date'] = pd.to_datetime(sub['GIORNO'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
sub = sub.dropna(subset=['date'])
sub['year'] = sub['date'].dt.year
sub['month'] = sub['date'].dt.month

print('Total Days:', len(sub))
print('Overall Mean Bottles:', round(sub['BOTTIGLIE'].mean(), 2))
print('Overall Sum Bottles:', int(sub['BOTTIGLIE'].sum()))
print('Overall Mean Users:', round(sub['PRESENZE'].mean(), 2))

for yr, grp in sub.groupby('year'):
    b_mean = round(grp['BOTTIGLIE'].mean(), 1)
    b_sum = int(grp['BOTTIGLIE'].sum())
    u_mean = round(grp['PRESENZE'].mean(), 1)
    print(f'Year {yr}: {len(grp)} days, Mean B: {b_mean}, Sum B: {b_sum}, Mean U: {u_mean}')

sub_2026 = sub[sub['year'] == 2026]
sub_9 = sub_2026[sub_2026['month'] <= 9]
print('2026 YTD (Jan-Sep):', len(sub_9), 'days, Mean B:', round(sub_9['BOTTIGLIE'].mean(), 2), 'Mean U:', round(sub_9['PRESENZE'].mean(), 2))

for m, grp in sub_9.groupby('month'):
    b_mean = round(grp['BOTTIGLIE'].mean(), 1)
    b_tot = int(grp['BOTTIGLIE'].sum())
    u_mean = round(grp['PRESENZE'].mean(), 1)
    u_tot = int(grp['PRESENZE'].sum())
    print(f'M{m:02d}: {len(grp)}d, Mean B: {b_mean}, Tot B: {b_tot}, Mean U: {u_mean}, Tot U: {u_tot}')
