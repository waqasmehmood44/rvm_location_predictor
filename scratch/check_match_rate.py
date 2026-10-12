import pandas as pd

df_stores = pd.read_csv('data/raw/CaricaViewStoricoPuntoFisico.csv', sep=';', low_memory=False)
df_sheet3 = pd.read_excel('data/raw/Premialità-Rewards.xlsx', sheet_name='Foglio3')

print("Stores count:", len(df_stores))
print("Stores columns with filiera:", [c for c in df_stores.columns if 'filiera' in c.lower()])

sheet3_codici = set(df_sheet3['CODICE'].dropna().str.strip().str.upper())
sheet3_nomi = set(df_sheet3['NOME'].dropna().str.strip().str.upper())

stores_filiera = df_stores['Filiera'].dropna().str.strip().str.upper()
stores_nome_filiera = df_stores['NomeFiliera'].dropna().str.strip().str.upper()

match_fil_codice = stores_filiera.isin(sheet3_codici).sum()
match_fil_nome = stores_filiera.isin(sheet3_nomi).sum()
match_nomefil_nome = stores_nome_filiera.isin(sheet3_nomi).sum()
match_nomefil_codice = stores_nome_filiera.isin(sheet3_codici).sum()

either_match = (
    stores_filiera.isin(sheet3_codici) | 
    stores_filiera.isin(sheet3_nomi) | 
    stores_nome_filiera.isin(sheet3_nomi) | 
    stores_nome_filiera.isin(sheet3_codici)
).sum()

print(f"Match by Filiera in CODICE: {match_fil_codice} / {len(df_stores)}")
print(f"Match by Filiera in NOME: {match_fil_nome} / {len(df_stores)}")
print(f"Match by NomeFiliera in NOME: {match_nomefil_nome} / {len(df_stores)}")
print(f"Either match: {either_match} / {len(df_stores)} ({either_match/len(df_stores)*100:.1f}%)")

# Sample unmatched
unmatched = df_stores[~(
    stores_filiera.isin(sheet3_codici) | 
    stores_filiera.isin(sheet3_nomi) | 
    stores_nome_filiera.isin(sheet3_nomi) | 
    stores_nome_filiera.isin(sheet3_codici)
)][['PuntoFisico', 'Filiera', 'NomeFiliera', 'Insegna', 'Comune']].head(10)
print("\nSample unmatched stores:")
print(unmatched)
