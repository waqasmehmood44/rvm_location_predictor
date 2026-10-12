import pandas as pd
import re

df = pd.read_excel('data/raw/Premialità-Rewards.xlsx', sheet_name='Foglio3')

def parse_reward_row(row):
    nome = str(row['NOME']).strip()
    codice = str(row['CODICE']).strip()
    desc_cat = str(row['DESCRIZIONE']).strip()
    titolo = str(row['titolo']).strip() if pd.notna(row['titolo']) else ""
    desc_text = str(row['descrizione.1']).strip() if pd.notna(row['descrizione.1']) else ""
    pts = int(row['valore_punti']) if pd.notna(row['valore_punti']) else 200
    conv = int(row['conversione']) if pd.notna(row['conversione']) else 0
    
    full_text = f"{titolo} {desc_text}".replace('_x000D_', ' ').replace('\n', ' ')
    
    # 1. Discount amount in Euros
    disc_euro = None
    m_disc = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:€|euro)\s*(?:di\s*)?sconto', full_text, re.IGNORECASE)
    if not m_disc:
        m_disc = re.search(r'sconto\s*(?:di\s*)?(\d+(?:[.,]\d+)?)\s*(?:€|euro)', full_text, re.IGNORECASE)
    if not m_disc:
        m_disc = re.search(r'buono\s*(?:del\s*valore\s*di\s*|sconto\s*di\s*)?(\d+(?:[.,]\d+)?)\s*(?:€|euro)', full_text, re.IGNORECASE)
    if not m_disc:
        m_disc = re.search(r'(\d+(?:[.,]\d+)?)\s*€', titolo, re.IGNORECASE)
        
    if m_disc:
        try:
            disc_euro = float(m_disc.group(1).replace(',', '.'))
        except ValueError:
            pass

    # 2. Minimum spend in Euros
    min_spend = None
    m_spend = re.search(r'(?:spesa\s+minima|minimo\s+di\s+spesa|minimo\s+scontrino)\s*(?:di\s*|da\s*)?(\d+(?:[.,]\d+)?)\s*(?:€|euro)?', full_text, re.IGNORECASE)
    if not m_spend:
        m_spend = re.search(r'su\s+(\d+(?:[.,]\d+)?)\s*(?:€|euro)\s*di\s*spesa', full_text, re.IGNORECASE)
    if not m_spend:
        m_spend = re.search(r'su\s*un\s*minimo\s*di\s*(\d+(?:[.,]\d+)?)\s*(?:€|euro)', full_text, re.IGNORECASE)

    if m_spend:
        try:
            min_spend = float(m_spend.group(1).replace(',', '.'))
        except ValueError:
            pass

    # 3. Derived Economic Metrics
    # Value per bottle (cents/bottle)
    val_per_bottle = round((disc_euro / pts), 4) if (disc_euro and pts > 0) else None
    
    # Effective basket discount %
    discount_pct = round((disc_euro / min_spend) * 100.0, 1) if (disc_euro and min_spend and min_spend > 0) else None
    
    return {
        'codice': codice,
        'nome': nome,
        'category': desc_cat,
        'titolo': titolo,
        'descrizione_dettaglio': desc_text,
        'punti_richiesti': pts,
        'sconto_euro': disc_euro,
        'spesa_minima_euro': min_spend,
        'valore_euro_per_bottiglia': val_per_bottle,
        'sconto_percentuale_carrello': discount_pct,
        'conversione_mode': conv
    }

parsed = [parse_reward_row(r) for _, r in df.iterrows()]
res_df = pd.DataFrame(parsed)

print("Parsed successfully:", len(res_df))
print("Discount extracted:", res_df['sconto_euro'].notna().sum(), f"({res_df['sconto_euro'].notna().mean()*100:.1f}%)")
print("Min spend extracted:", res_df['spesa_minima_euro'].notna().sum(), f"({res_df['spesa_minima_euro'].notna().mean()*100:.1f}%)")
print("Value per bottle extracted:", res_df['valore_euro_per_bottiglia'].notna().sum())
print("Discount % on cart extracted:", res_df['sconto_percentuale_carrello'].notna().sum())

print("\nValue per bottle stats (in Euro cents):")
v_cents = res_df['valore_euro_per_bottiglia'].dropna() * 100
print(v_cents.describe())

print("\nCart discount % stats:")
print(res_df['sconto_percentuale_carrello'].dropna().describe())

print("\nSample 10 rich extracted profiles:")
for idx, r in res_df.sample(10, random_state=123).iterrows():
    print(f"[{r['nome']}] -> {r['titolo']} | {r['sconto_euro']}€ discount on {r['spesa_minima_euro']}€ spend at {r['punti_richiesti']} pts | Val/bottle: {r['valore_euro_per_bottiglia']}€ ({r['sconto_percentuale_carrello']}%)")
