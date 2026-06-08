"""
Сравнение ИСХОДНОЙ модели (Tang & Pikal, квазистационар, постоянная Tp)
и УЛУЧШЕННОЙ (подвижный фронт + коллапс + связь с заморозкой).

4 панели:
 A) Tp(t): исходная считает Tp постоянной; улучшенная -- Tp дрейфует, есть Tb-Tp.
 B) Доля высохшего слоя во времени: исходная оптимистична (тёплый фронт с старта).
 C) Время сушки vs температура полки: разрыв = вклад нестационарного старта.
 D) Время сушки vs переохлаждение: у исходной НЕТ физики заморозки (плоская линия).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lyo_model as m

Tc, Tg_prime = -32.0, -34.0
Pc = 0.10                      # Torr
L0 = 0.01                      # м
Ts_demo = -20.0               # °C для панелей A,B

# ---------- A,B: одна полка, динамика против «постоянной Tp» ----------
r = m.primary_drying(Ts_demo, Pc, dT_super=10.0, Tc_C=Tc, Tg_prime_C=Tg_prime)
Tp_const = r["Tp_max"]         # уровень, который исходная приняла бы постоянным

# исходная модель: интеграл при ПОСТОЯННОЙ Tp = Tp_const
def orig_progress(Tp_const, Pc, L0, n=400):
    fcr = m.crystal_factor(10.0)
    Pc_pa = Pc * m.TORR
    H = np.linspace(0, L0, n); t = [0.0]
    for i in range(n - 1):
        Hm = 0.5 * (H[i] + H[i + 1])
        Rp = m.Rp_areal(Hm, 0.5, 20.0, 4.0, fcr)
        Js = max((m.p_ice(Tp_const) - Pc_pa) / Rp, 1e-12)
        t.append(t[-1] + (H[i + 1] - H[i]) / (Js / m.RHO_ICE_CAKE))
    return np.array(t) / 3600.0, H / L0

t_orig, frac_orig = orig_progress(Tp_const, Pc, L0)
t_mine, frac_mine = r["t"] / 3600.0, r["H"] / L0

# ---------- C: время vs полка ----------
Ts_sweep = np.arange(-38, -8, 2.0)
t_dyn, t_orig_c, tp_dyn, coll = [], [], [], []
for Ts in Ts_sweep:
    rr = m.primary_drying(Ts, Pc, dT_super=10.0, Tc_C=Tc, Tg_prime_C=Tg_prime)
    t_dyn.append(rr["t_dry_h"]); tp_dyn.append(rr["Tp_max"])
    coll.append(rr["collapsed"])
    t_orig_c.append(m.tang_pikal_time(rr["Tp_max"], Pc, L0))   # при той же Tp
t_dyn = np.array(t_dyn); t_orig_c = np.array(t_orig_c); coll = np.array(coll)

# ---------- D: время vs переохлаждение (физика заморозки) ----------
dT_list = np.array([2, 4, 6, 8, 10, 14, 18, 22])
t_dyn_D = [m.primary_drying(Ts_demo, Pc, dT_super=d, Tc_C=Tc,
                            Tg_prime_C=Tg_prime)["t_dry_h"] for d in dT_list]
# исходная модель «видит» только зафиксированное R_p (ΔT_ref=10): плоская линия
t_orig_D = m.tang_pikal_time(Tp_const, Pc, L0)

# ================================ ГРАФИКИ ================================
fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# A
ax[0,0].plot(t_mine, r["Tp"], color="tab:green", lw=2, label="улучшенная: Tp(t)")
ax[0,0].plot(t_mine, r["Tb"], color="tab:green", ls=":", label="улучшенная: Tb(t) дно")
ax[0,0].axhline(Tp_const, color="tab:blue", ls="--",
                label=f"исходная: Tp=const={Tp_const:.1f}°C")
ax[0,0].axhline(Tc, color="red", ls="-.", label=f"Tc коллапса={Tc}°C")
ax[0,0].set_xlabel("время, ч"); ax[0,0].set_ylabel("температура, °C")
ax[0,0].set_title(f"A. Профиль температуры (Ts={Ts_demo}°C)\n"
                  "исходная: Tp постоянна; улучшенная: дрейф + градиент")
ax[0,0].legend(fontsize=8); ax[0,0].grid(alpha=0.3)

# B
ax[0,1].plot(t_mine, frac_mine, color="tab:green", lw=2, label="улучшенная (подвижный фронт)")
ax[0,1].plot(t_orig, frac_orig, color="tab:blue", ls="--", label="исходная (Tp=const)")
ax[0,1].set_xlabel("время, ч"); ax[0,1].set_ylabel("доля высохшего слоя")
ax[0,1].set_title("B. Кинетика сушки\nисходная оптимистична (тёплый фронт с старта)")
ax[0,1].legend(fontsize=8); ax[0,1].grid(alpha=0.3)

# C
ok = ~coll
ax[1,0].plot(Ts_sweep, t_orig_c, "s--", color="tab:blue", label="исходная (Tp=const)")
ax[1,0].plot(Ts_sweep[ok], t_dyn[ok], "o-", color="tab:green", label="улучшенная, без коллапса")
ax[1,0].plot(Ts_sweep[~ok], t_dyn[~ok], "x", color="tab:red", ms=9, label="улучшенная: КОЛЛАПС")
ax[1,0].set_yscale("log")
ax[1,0].set_xlabel("температура полки Ts, °C"); ax[1,0].set_ylabel("время сушки, ч (log)")
ax[1,0].set_title("C. Время vs полка\nразрыв = нестационарный старт; крестики = брак")
ax[1,0].legend(fontsize=8); ax[1,0].grid(alpha=0.3)

# D
ax[1,1].plot(dT_list, t_dyn_D, "o-", color="tab:green", label="улучшенная (заморозка→Rp)")
ax[1,1].axhline(t_orig_D, color="tab:blue", ls="--",
                label="исходная (Rp фиксировано, заморозки нет)")
ax[1,1].set_xlabel("степень переохлаждения ΔT_supercool, °C")
ax[1,1].set_ylabel("время сушки, ч")
ax[1,1].set_title("D. Влияние ЗАМОРОЗКИ\nисходная модель его не видит")
ax[1,1].legend(fontsize=8); ax[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("comparison.png", dpi=130)
print("Сохранено: comparison.png")
print(f"A/B: Ts={Ts_demo}: улучшенная t={t_mine[-1]:.1f} ч, "
      f"исходная (Tp={Tp_const:.1f}) t={t_orig[-1]:.1f} ч "
      f"(исходная занижает на {100*(1-t_orig[-1]/t_mine[-1]):.0f}%)")
print(f"C: при Ts={Ts_sweep[ok][-1]:.0f} коллапс начинается; "
      f"исходная этого не сигналит")
print(f"D: улучшенная от {min(t_dyn_D):.1f} до {max(t_dyn_D):.1f} ч; "
      f"исходная одна точка {t_orig_D:.1f} ч")
