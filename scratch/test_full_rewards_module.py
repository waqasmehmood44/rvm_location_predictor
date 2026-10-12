import re
import pandas as pd
import numpy as np

def extract_discount_mechanics(titolo: str, desc1: str, punti: int):
    t_clean = str(titolo or "").strip()
    d_clean = str(desc1 or "").strip()
    full_text = f"{t_clean} {d_clean}".replace('_x000D_', ' ').replace('\n', ' ')
    
    # 1. Discount amount in Euros
    disc_euro = None
    m_disc = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:€|euro)\s*(?:di\s*)?sconto', full_text, re.IGNORECASE)
    if not m_disc:
        m_disc = re.search(r'sconto\s*(?:di\s*)?(\d+(?:[.,]\d+)?)\s*(?:€|euro)', full_text, re.IGNORECASE)
    if not m_disc:
        m_disc = re.search(r'buono\s*(?:del\s*valore\s*di\s*|sconto\s*di\s*)?(\d+(?:[.,]\d+)?)\s*(?:€|euro)', full_text, re.IGNORECASE)
    if not m_disc:
        m_disc = re.search(r'(\d+(?:[.,]\d+)?)\s*€', t_clean, re.IGNORECASE)
        
    if m_disc:
        try:
            disc_euro = float(m_disc.group(1).replace(',', '.'))
        except ValueError:
            pass
            
    # Default fallback for GDO standard if not stated
    if disc_euro is None:
        disc_euro = 3.0

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

    if min_spend is None:
        min_spend = 25.0  # National median

    pts = int(punti) if punti and punti > 0 else 200

    # 3. Derived economics
    val_per_bottle = round(disc_euro / pts, 4) if pts > 0 else 0.015
    discount_pct = round((disc_euro / min_spend) * 100.0, 1) if min_spend > 0 else 10.0

    return {
        "titolo": t_clean,
        "descrizione_dettaglio": d_clean.replace('_x000D_', '').strip(),
        "punti_richiesti": pts,
        "sconto_euro": disc_euro,
        "spesa_minima_euro": min_spend,
        "valore_euro_per_bottiglia": val_per_bottle,
        "valore_cent_per_bottiglia": round(val_per_bottle * 100.0, 2),
        "sconto_percentuale_carrello": discount_pct
    }

def calculate_incentive_elasticity(reward_cat: str, mech: dict):
    pts = mech["punti_richiesti"]
    val_bottle = mech["valore_euro_per_bottiglia"]
    disc_pct = mech["sconto_percentuale_carrello"]
    
    # Baseline: 200 pts, 0.015 €/b (1.5 cents), 10% cart discount
    # 1. Points hurdle elasticity (lower points = higher frequency)
    f_pts = (200.0 / max(50.0, float(pts))) ** 0.15
    f_pts = float(np.clip(f_pts, 0.85, 1.20))
    
    # 2. Value per bottle elasticity
    f_val = 1.0 + 0.10 * ((val_bottle - 0.015) / 0.015)
    f_val = float(np.clip(f_val, 0.90, 1.25))
    
    # 3. Cart discount percentage elasticity
    f_disc = 1.0 + 0.05 * ((disc_pct - 10.0) / 10.0)
    f_disc = float(np.clip(f_disc, 0.92, 1.15))
    
    # 4. Loyalty channel factor
    f_channel = 1.0
    if "Conversione a punti fedeltà" in reward_cat:
        f_channel = 1.15
    elif "Omaggio" in reward_cat:
        f_channel = 1.08
    elif "Solo Coripet" in reward_cat:
        f_channel = 0.88
    elif "Sconto Tari" in reward_cat:
        f_channel = 0.92
        
    mult_presenze = round(float(np.clip(f_pts * f_val * f_disc * f_channel, 0.75, 1.45)), 3)
    mult_bottles = round(float(np.clip(np.sqrt(mult_presenze) * f_val, 0.80, 1.35)), 3)
    
    return {
        "multiplier_presenze": mult_presenze,
        "multiplier_bottles": mult_bottles,
        "incentive_index": round((mult_presenze + mult_bottles) / 2.0, 3)
    }

# Test on major stores
df_sheet3 = pd.read_excel('data/raw/Premialità-Rewards.xlsx', sheet_name='Foglio3')
for code in ['Bennet', 'CONAD_ITALIA', 'Unicoop.Fi', 'Decò', 'PAM_Decaro', 'Comune di Cellole']:
    matches = df_sheet3[df_sheet3['CODICE'].astype(str).str.upper() == code.upper()]
    if len(matches) > 0:
        r = matches.iloc[0]
        mech = extract_discount_mechanics(r['titolo'], r['descrizione.1'], r['valore_punti'])
        elast = calculate_incentive_elasticity(r['DESCRIZIONE'], mech)
        print(f"\nStore: {code}")
        print(f"  Voucher: {mech['titolo']} ({mech['sconto_euro']}€ on {mech['spesa_minima_euro']}€ spend)")
        print(f"  Punti: {mech['punti_richiesti']} bottles | Rate: {mech['valore_cent_per_bottiglia']} ¢/bottle | Cart Discount: {mech['sconto_percentuale_carrello']}%")
        print(f"  Elasticity Multipliers -> Bottles: {elast['multiplier_bottles']}x | Presenze: {elast['multiplier_presenze']}x")
