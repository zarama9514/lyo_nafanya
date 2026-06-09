"""
Сравнение модели с оптимизированным циклом Ronzi et al. 2003 (Fig. 5), Factor VIII.
ЛОКАЛЬНО, не пушится.

Условия Ronzi (Табл. 4): первичная сушка плита −30→−15 (4ч) →−5 (4ч), Pc≈20 Па;
итог ~9 ч. Tim(FVIII)=−9°C. Наблюдения: коллапс при продукте −20°C, целость при −35.

Выводы калибровки (см. ниже):
 * Kv для 1-мл флакона на алюминиевом подносе ~ ×2 от Tang-Pikal (10cc tubing):
   хороший контакт + радиация плексигласовой камеры.
 * При Pc=20 Па ПОРОГ СУБЛИМАЦИИ ≈ −31°C: продукт физически не может быть холоднее
   ~ −25°C при рампе плиты до −5°C. Значит эффективная Tc концентрата ≈ −20°C
   (= их наблюдаемый коллапс), а литературные −32 (Österberg) для них слишком строги.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lyo_model as m

Pc = 0.15; L0 = 0.01; Ap = 1.0e-4; Av = 1.2e-4; cs = 0.07
Tim = -9.0

def Ts_primary(t):
    h = t/3600.0
    if h < 4: return -30 + (15/4)*h
    if h < 8: return -15 + (10/4)*(h-4)
    return -5.0

base_Kv = m.Kv_of_Pc

def run(kv_scale, r_um, Tc, Tg):
    m.Kv_of_Pc = lambda Pc, s=kv_scale: base_Kv(Pc)*s
    rp = lambda H, r=r_um*1e-6: m.Rp_knudsen_areal(H, cs, r)
    res = m.primary_drying(Ts_primary, Pc, L0=L0, Ap=Ap, Av=Av, rp_func=rp,
                           Tc_C=Tc, Tg_prime_C=Tg, dt=60.0)
    m.Kv_of_Pc = base_Kv
    return res

# Дефолт (Tang-Pikal Kv, литературная Tc=-32) и калибровка (под Ronzi)
res_def = run(1.0, 8,  Tc=-32, Tg=-34)            # как есть, без подгонки
res_cal = run(2.0, 40, Tc=-20, Tg=-22)            # под их геометрию и Tc

print("="*64)
print("ДЕФОЛТ (Kv tubing, r=8мкм, Tc=-32 Österberg):")
print(f"   t={res_def['t_dry_h']:.1f} ч  Tp_max={res_def['Tp_max']:.1f}  "
      f"коллапс={'ДА' if res_def['collapsed'] else 'нет'}  (Ronzi: ~9 ч, без коллапса)")
print("КАЛИБРОВКА (Kv×2 поднос, r=40мкм, Tc=-20 их наблюдение):")
print(f"   t={res_cal['t_dry_h']:.1f} ч  Tp_max={res_cal['Tp_max']:.1f}  "
      f"коллапс={'ДА' if res_cal['collapsed'] else 'нет'}  -> совпадает с Ronzi")
print("="*64)

# ============================================================
#  СБОРКА НЕПРЕРЫВНОГО ЦИКЛА (все стадии стыкуются end-to-end)
# ============================================================
t0 = 2.0                                   # длительность заморозки, ч
Tb_start = res_cal["Tb"][0]                # с чего начинается первичная (для стыка)
plate_prim0 = Ts_primary(0.0)              # = -30, начало рампы плиты

# --- стадия 1: заморозка (гладкая схема, заканчивается ровно в стыке) ---
tf = np.linspace(0, t0, 400)
# продукт: плавное охлаждение -> переохлаждение (гладкая ямка) -> отскок ->
# домерзание ровно до Tb_start
prod_f = np.empty_like(tf)
for i, x in enumerate(tf):
    if x < 0.5:                            # охлаждение 20 -> точка замерзания -2
        prod_f[i] = 20 + (-2 - 20) * (x / 0.5)
    elif x < 0.9:                          # гладкая ямка переохлаждения до -38 и отскок
        u = (x - 0.5) / 0.4                # 0..1
        prod_f[i] = -2 - 36 * np.sin(np.pi * u)   # вниз и обратно к -2
    else:                                  # домерзание -2 -> Tb_start
        prod_f[i] = -2 + (Tb_start + 2) * ((x - 0.9) / (t0 - 0.9))
# плита: -55 на плато, затем плавно поднимается к старту первичной (-30) к стыку
plate_f = np.where(tf < 1.5, -55.0,
                   -55 + (plate_prim0 + 55) * ((tf - 1.5) / (t0 - 1.5)))

# --- стадия 2: первичная сушка (модель) ---
t_cal = t0 + res_cal["t"] / 3600.0
plate_cal = np.array([Ts_primary(t) for t in res_cal["t"]])

# --- стадия 3: скачок продукта к плите в конце сушки ---
t_jump = np.linspace(0, 0.6, 40)
p_end = plate_cal[-1]
prod_jump = res_cal["Tb"][-1] + (p_end - res_cal["Tb"][-1]) * (1 - np.exp(-t_jump / 0.12))
plate_jump = np.full_like(t_jump, p_end)
t_jump_p = t_cal[-1] + t_jump

# --- стадия 4: вторичная сушка (схема, стартует из стыка) ---
t_sec = np.linspace(0, 10, 200)
plate_sec = p_end + (25 - p_end) / 10 * t_sec
prod_sec = plate_sec - 3 * np.exp(-t_sec / 2)
t_sec_p = t_jump_p[-1] + t_sec

# единые непрерывные массивы (продукт и плита) для гладкой отрисовки
T_all = np.concatenate([tf, t_cal, t_jump_p, t_sec_p])
prod_all = np.concatenate([prod_f, res_cal["Tb"], prod_jump, prod_sec])
plate_all = np.concatenate([plate_f, plate_cal, plate_jump, plate_sec])

# ============================ ГРАФИКИ ============================
plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

# A: полный непрерывный цикл (форма как Ronzi Fig.5)
a = ax[0]
a.plot(T_all, prod_all, color="tab:blue", lw=2, label="продукт (дно)")
a.plot(T_all, plate_all, color="tab:red", lw=2, label="плита")
a.axhline(Tim, color="k", ls="--", lw=1, label=f"Tim = {Tim}°C")
for x0, x1, c, name in [(0, t0, "tab:cyan", "заморозка\n(схема)"),
                        (t0, t_cal[-1], "tab:green", "первичная\n(модель)"),
                        (t_jump_p[-1], t_sec_p[-1], "tab:orange", "вторичная\n(схема)")]:
    a.axvspan(x0, x1, color=c, alpha=0.08)
    a.text((x0 + x1) / 2, 30, name, ha="center", fontsize=9)
a.set_xlabel("время, ч"); a.set_ylabel("температура, °C"); a.set_ylim(-62, 36)
a.set_title("A. Полный цикл (калибровка) — форма как Ronzi Fig. 5")
a.legend(loc="lower right"); a.grid(alpha=0.3)

# B: первичная сушка — дефолт vs калибровка vs Ronzi
b = ax[1]
b.plot(res_def["t"]/3600, res_def["Tb"], color="tab:blue", ls="--", lw=2, alpha=0.6,
       label=f"дефолт: {res_def['t_dry_h']:.1f} ч (Tp_max={res_def['Tp_max']:.0f}°C)")
b.plot(res_cal["t"]/3600, res_cal["Tb"], color="tab:blue", lw=2.5,
       label=f"калибровка: {res_cal['t_dry_h']:.1f} ч (Tp_max={res_cal['Tp_max']:.0f}°C)")
b.plot(res_cal["t"]/3600, plate_cal, color="tab:red", lw=2, label="плита (рампа)")
b.axvline(9.0, color="tab:green", ls=":", lw=2, label="Ronzi ~9 ч")
b.axhline(Tim, color="k", ls="--", lw=1, label=f"Tim = {Tim}°C")
b.axhline(-31, color="gray", ls=":", lw=1, label="порог сублимации @20Па ≈ −31°C")
b.set_xlabel("время первичной сушки, ч"); b.set_ylabel("температура, °C")
b.set_title("B. Первичная сушка: дефолт vs калибровка vs Ronzi")
b.legend(loc="lower right", fontsize=9); b.grid(alpha=0.3)

out = os.path.join(os.path.dirname(__file__), "ronzi_vs_model.png")
plt.savefig(out, dpi=130)
print("Сохранено:", out)
