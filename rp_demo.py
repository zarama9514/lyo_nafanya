"""
Теоретический расчёт R_p (БЕЗ эксперимента) по кнудсеновской диффузии и его
влияние на время сушки.

  R_p из микроструктуры:  пористость(концентрация) + размер пор(заморозка).

Панели:
 A) R_p(1 см) vs размер пор -- проверка попадания в литературный коридор.
 B) R_p(1 см) vs концентрация сухого.
 C) Время сушки с теоретическим R_p для разных режимов заморозки.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lyo_model as m

print("=" * 66)
print("Проверка: R_p(1 см) для типовых режимов заморозки (5% сухого)")
print("=" * 66)
cs = 0.05
for reg, r in m.PORE_BY_REGIME.items():
    rp = m.Rp_at_1cm_torr(cs, r)
    print(f"  {reg:22s} r_pore={r*1e6:4.0f} мкм -> R_p={rp:5.1f} Torr·см²·ч/г")
print("  (литература Tang&Pikal: ~5 для «средних» ~5% -> модель в коридоре)")

# ----- A: R_p vs размер пор -----
r_um = np.linspace(5, 50, 40)
rp_A = [m.Rp_at_1cm_torr(cs, ru * 1e-6) for ru in r_um]

# ----- B: R_p vs концентрация -----
cs_list = np.linspace(0.01, 0.20, 40)
rp_B = [m.Rp_at_1cm_torr(c, m.PORE_BY_REGIME["normal"]) for c in cs_list]

# ----- C: время сушки с теоретическим R_p по режимам заморозки -----
print("\n" + "=" * 66)
print("Время первичной сушки с ТЕОРЕТИЧЕСКИМ R_p (Ts=-20°C, Pc=0.1 Torr)")
print("=" * 66)
regimes = ["controlled_nucleation", "annealed", "slow_shelf", "normal", "fast", "LN2"]
times_C, labels_C = [], []
for reg in regimes:
    r = m.PORE_BY_REGIME[reg]
    rp_func = lambda H, r=r: m.Rp_knudsen_areal(H, cs, r)
    res = m.primary_drying(-20.0, 0.10, rp_func=rp_func,
                           Tc_C=-32.0, Tg_prime_C=-34.0)
    times_C.append(res["t_dry_h"]); labels_C.append(reg)
    print(f"  {reg:22s} r_pore={r*1e6:4.0f} мкм -> "
          f"t={res['t_dry_h']:5.1f} ч, Tp_max={res['Tp_max']:.1f}°C, "
          f"коллапс={'ДА' if res['collapsed'] else 'нет'}")
print(f"  -> контролируемая нуклеация против LN2: "
      f"{100*(1-times_C[0]/times_C[-1]):.0f}% экономии времени")

# --------------------------- ГРАФИКИ ---------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

ax[0].plot(r_um, rp_A, color="tab:blue")
ax[0].axhspan(3, 7, color="tab:green", alpha=0.15, label="литература ~5% (Tang&Pikal)")
ax[0].set_xlabel("радиус пор r_pore, мкм"); ax[0].set_ylabel("R_p(1 см), Torr·см²·ч/г")
ax[0].set_title("A. Теоретический R_p vs размер пор\n(размер задаёт заморозка)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

ax[1].plot(cs_list * 100, rp_B, color="tab:purple")
ax[1].set_xlabel("концентрация сухого, %"); ax[1].set_ylabel("R_p(1 см), Torr·см²·ч/г")
ax[1].set_title("B. R_p vs концентрация\n(пористость из состава)")
ax[1].grid(alpha=0.3)

colors = plt.cm.viridis(np.linspace(0, 0.9, len(regimes)))
ax[2].barh(range(len(regimes)), times_C, color=colors)
ax[2].set_yticks(range(len(regimes))); ax[2].set_yticklabels(labels_C, fontsize=8)
ax[2].invert_yaxis()
ax[2].set_xlabel("время первичной сушки, ч")
ax[2].set_title("C. Время с теоретическим R_p\nпо режимам заморозки")
ax[2].grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("rp_theory.png", dpi=130)
print("\nГрафики сохранены в rp_theory.png")
