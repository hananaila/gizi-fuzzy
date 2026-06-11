from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

def calc_lms(x, l, m, s):
    # Definisi baku rumus Box-Cox Power Exponential (LMS Method WHO)
    if abs(l) < 0.01:
        return np.log(x / m) / s
    else:
        return (((x / m) ** l) - 1) / (l * s)

def calculate_z_scores(umur, jk, bb, tb):
    # Batasan aman umur 0 - 60 bulan sesuai standar balita
    umur = max(0, min(umur, 60))
    
    # Referensi Umur & Tinggi (Key points: 0, 12, 24, 36, 48, 60 bulan)
    ages = [0.0, 12.0, 24.0, 36.0, 48.0, 60.0]
    
    # Nilai Median (M) Referensi WHO
    if jk == 'L':
        w_m = [3.3, 9.6, 12.2, 14.3, 16.3, 18.3]   # Median WAZ Boys
        h_m = [50.5, 75.7, 87.1, 96.1, 103.3, 110.0] # Median HAZ Boys
    else:
        w_m = [3.2, 8.9, 11.5, 13.9, 15.8, 17.5]   # Median WAZ Girls
        h_m = [49.1, 74.0, 85.5, 95.1, 102.7, 109.4] # Median HAZ Girls
        
    # Interpolasi linear mencari M secara dinamis berdasarkan bulan
    m_waz = np.interp(umur, ages, w_m)
    m_haz = np.interp(umur, ages, h_m)
    
    # Asumsi Konstanta Deviasi (S) dan Skew / Box-Cox (L)
    # WAZ (Berat per Umur): L mendekati 0, S sekitar 0.12
    l_waz, s_waz = 0.0, 0.12
    # HAZ (Tinggi per Umur): L selalu 1 (distribusi normal), S rasio varians
    l_haz, s_haz = 1.0, 0.04
    
    # Nilai Median WHZ (Berat per Panjang Badan Anak 50 s/d 115cm)
    tb_pts = [50.0, 75.0, 85.0, 95.0, 105.0, 115.0]
    if jk == 'L':
        wh_m = [3.3, 9.5, 12.0, 14.3, 16.8, 19.5]
    else:
        wh_m = [3.3, 9.0, 11.5, 13.8, 16.2, 19.0]
        
    m_whz = np.interp(tb, tb_pts, wh_m)
    l_whz, s_whz = 0.0, 0.08
    
    # Eksekusi Formula Asli LMS WHO (Seperti Di Jurnal Anda)
    waz = calc_lms(bb, l_waz, m_waz, s_waz)
    haz = calc_lms(tb, l_haz, m_haz, s_haz)
    whz = calc_lms(bb, l_whz, m_whz, s_whz)
    
    return round(float(np.clip(waz, -4.0, 4.0)), 2), \
           round(float(np.clip(haz, -4.0, 4.0)), 2), \
           round(float(np.clip(whz, -4.0, 4.0)), 2)

# Fuzzifikasi: Fungsi Trapesium dan Segitiga
def trapesium(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a) if b != a else 1.0
    elif b <= x <= c:
        return 1.0
    elif c < x < d:
        return -(x - d) / (d - c) if d != c else 1.0
    return 0.0

def segitiga(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    elif b < x < c:
        return -(x - c) / (c - b) if c != b else 1.0
    return 0.0

def fuzzify(val):
    return {
        'SL': trapesium(val, -4, -4, -3, -2), # Severely Low
        'L': segitiga(val, -3, -2, 0),        # Low
        'N': trapesium(val, -2, 0, 0, 2),     # Normal
        'H': trapesium(val, 1, 2, 4, 4)       # High
    }

# Inferensi Sugeno Orde Nol (27 Rules)
def inferensi_sugeno(waz_f, haz_f, whz_f):
    rules = []
    
    # === Severely Malnourished (NSS = -3.5) ===
    # R1: SL SL SL
    rules.append((min(waz_f['SL'], haz_f['SL'], whz_f['SL']), -3.5))
    # R2: SL SL L
    rules.append((min(waz_f['SL'], haz_f['SL'], whz_f['L']), -3.5))
    # R3: SL L SL
    rules.append((min(waz_f['SL'], haz_f['L'], whz_f['SL']), -3.5))
    # R4: L SL SL
    rules.append((min(waz_f['L'], haz_f['SL'], whz_f['SL']), -3.5))
    
    # === Malnourished (NSS = -2.5) ===
    # R5: SL L L
    rules.append((min(waz_f['SL'], haz_f['L'], whz_f['L']), -2.5))
    # R6: L SL L
    rules.append((min(waz_f['L'], haz_f['SL'], whz_f['L']), -2.5))
    # R7: L L SL
    rules.append((min(waz_f['L'], haz_f['L'], whz_f['SL']), -2.5))
    # R8: L L L
    rules.append((min(waz_f['L'], haz_f['L'], whz_f['L']), -2.5))
    # R9: L L N
    rules.append((min(waz_f['L'], haz_f['L'], whz_f['N']), -2.5))
    # R10: L N L
    rules.append((min(waz_f['L'], haz_f['N'], whz_f['L']), -2.5))
    # R11: N SL L
    rules.append((min(waz_f['N'], haz_f['SL'], whz_f['L']), -2.5))
    # R12: N L SL
    rules.append((min(waz_f['N'], haz_f['L'], whz_f['SL']), -2.5))
    # R13: N SL SL
    rules.append((min(waz_f['N'], haz_f['SL'], whz_f['SL']), -2.5))
    # R14: N L L
    rules.append((min(waz_f['N'], haz_f['L'], whz_f['L']), -2.5))
    
    # === Normal (NSS = 0.0) ===
    # R15: L N N
    rules.append((min(waz_f['L'], haz_f['N'], whz_f['N']), 0.0))
    # R16: N L N
    rules.append((min(waz_f['N'], haz_f['L'], whz_f['N']), 0.0))
    # R17: N N L
    rules.append((min(waz_f['N'], haz_f['N'], whz_f['L']), 0.0))
    # R18: N N N
    rules.append((min(waz_f['N'], haz_f['N'], whz_f['N']), 0.0))
    # R19: H N N
    rules.append((min(waz_f['H'], haz_f['N'], whz_f['N']), 0.0))
    # R20: N H N
    rules.append((min(waz_f['N'], haz_f['H'], whz_f['N']), 0.0))
    # R21: N N H
    rules.append((min(waz_f['N'], haz_f['N'], whz_f['H']), 0.0))
    # R22: H H N
    rules.append((min(waz_f['H'], haz_f['H'], whz_f['N']), 0.0))
    # R23: H N H
    rules.append((min(waz_f['H'], haz_f['N'], whz_f['H']), 0.0))
    
    # === Overweight (NSS = 2.5) ===
    # R24: N H H
    rules.append((min(waz_f['N'], haz_f['H'], whz_f['H']), 2.5))
    # R25: H H H
    rules.append((min(waz_f['H'], haz_f['H'], whz_f['H']), 2.5))
    # R26: H H L
    rules.append((min(waz_f['H'], haz_f['H'], whz_f['L']), 2.5))
    # R27: L H H
    rules.append((min(waz_f['L'], haz_f['H'], whz_f['H']), 2.5))

    return rules

# Defuzzifikasi: Weighted Average
def defuzzify(rules):
    pembilang = sum(alpha * z for alpha, z in rules)
    penyebut = sum(alpha for alpha, z in rules)
    
    # Jika tidak ada satupun dari 27 rule yang cocok (kondisi un-covered logic)
    if penyebut == 0:
        return None
    return pembilang / penyebut

def get_category(z_star):
    if z_star is None:
        return "Out of Rules (Data Ekstrem)"
        
    categories = {
        -3.5: "Severely Malnourished",
        -2.5: "Malnourished",
        0.0: "Normal",
        2.5: "Overweight"
    }
    # Pemetaan ke singleton terdekat
    closest_key = min(categories.keys(), key=lambda k: abs(k - z_star))
    return categories[closest_key]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/proses', methods=['POST'])
def proses():
    umur = float(request.form['umur'])
    jk = request.form['jk']
    bb = float(request.form['bb'])
    tb = float(request.form['tb'])
    
    # 1. Z-Score Calculation
    waz, haz, whz = calculate_z_scores(umur, jk, bb, tb)
    
    # 2. Fuzzifikasi
    waz_f = fuzzify(waz)
    haz_f = fuzzify(haz)
    whz_f = fuzzify(whz)
    
    # 3. Inferensi Sugeno
    rules = inferensi_sugeno(waz_f, haz_f, whz_f)
    
    # Ekstrak rule yang aktif (alpha > 0) murni untuk bukti/debugging
    active_rules = []
    for i, rule in enumerate(rules):
        alpha, z_val = rule
        if alpha > 0:
            active_rules.append({'no': i + 1, 'alpha': round(float(alpha), 4), 'z': z_val})
    
    # 4. Defuzzifikasi
    z_star = defuzzify(rules)
    
    # 5. Kategori Akhir
    status_gizi = get_category(z_star)
    
    z_val_str = round(float(z_star), 4) if z_star is not None else "N/A"
    
    return render_template('hasil.html', 
                           umur=umur, jk=jk, bb=bb, tb=tb,
                           waz=waz, haz=haz, whz=whz,
                           z_star=z_val_str, kategori=status_gizi,
                           waz_f=waz_f, haz_f=haz_f, whz_f=whz_f, active_rules=active_rules)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
