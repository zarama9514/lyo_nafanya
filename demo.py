"""
Численные эксперименты к улучшенной модели лиофилизации.
Три вопроса задания:
 (A) Связь заморозки с сушкой: размер кристаллов -> R_p -> время.
 (B) Градиент температуры в корже и риск коллапса при агрессивном режиме.
 (C) Оптимальный режим: минимум времени при ограничении Tp < Tc (без коллапса).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lyo_model as m

Tc = -32.0          # температура коллапса формуляции, °C
Tg_prime = -34.0
margin = 2.0        # запас безопасности, °C

# ============ ЭКСПЕРИМЕНТ A: заморозка -> время первичной сушки ============
print("=" * 64)
print("A) Влияние режима ЗАМОРОЗКИ (переохлаждения) на время сушки")
print("=" * 64)
dT_list = np.array([2, 4, 6, 8, 10, 14, 18, 22])   # степень переохлаждения, °C
times_A, fcr_A = [], []
Pc_fixed = 0.10                                     # Torr
Ts_fixed = -10.0                                    # °C, постоянная полка
for dT in dT_list:
    r = m.primary_drying(Ts_fixed, Pc_fixed, dT_super=dT,
                         Tc_C=Tc, Tg_prime_C=Tg_prime)
    times_A.append(r["t_dry_h"]); fcr_A.append(r["fcryst"])
    print(f"  ΔT_supercool={dT:5.1f}°C  f_cryst={r['fcryst']:.2f}  "
          f"t_сушки={r['t_dry_h']:6.1f} ч  Tp_max={r['Tp_max']:.1f}°C")
times_A = np.array(times_A)
print(f"  -> контролируемая нуклеация (ΔT=2) экономит "
      f"{100*(1-times_A[0]/times_A[-1]):.0f}% времени против ΔT=22")

# ============ ЭКСПЕРИМЕНТ B: градиент T и коллапс при разной полке =========
print("\n" + "=" * 64)
print("B) Градиент Tb-Tp и риск коллапса при росте температуры полки")
print("=" * 64)
runs_B = {}
for Ts in [-25, -15, -5, +5]:
    r = m.primary_drying(Ts, Pc_fixed, dT_super=10.0,
                         Tc_C=Tc, Tg_prime_C=Tg_prime)
    runs_B[Ts] = r
    grad = np.max(r["Tb"] - r["Tp"])
    print(f"  Ts={Ts:+4d}°C  t={r['t_dry_h']:6.1f} ч  Tp_max={r['Tp_max']:6.1f}°C "
          f"(Tc={Tc})  max(Tb-Tp)={grad:4.1f}°C  "
          f"коллапс={'ДА' if r['collapsed'] else 'нет'}")

# ============ ЭКСПЕРИМЕНТ C: оптимум = минимум t без коллапса =============
print("\n" + "=" * 64)
print("C) Кривая компромисса: время сушки vs Tp; оптимум у границы Tc")
print("=" * 64)
Ts_sweep = np.arange(-40, 11, 2.5)
sweep = [m.primary_drying(Ts, Pc_fixed, dT_super=10.0,
                          Tc_C=Tc, Tg_prime_C=Tg_prime) for Ts in Ts_sweep]
t_sweep = np.array([r["t_dry_h"] for r in sweep])
tp_sweep = np.array([r["Tp_max"] for r in sweep])
coll_sweep = np.array([r["collapsed"] for r in sweep])

safe = [(Ts, r) for Ts, r in zip(Ts_sweep, sweep) if not r["collapsed"]]
Ts_opt, r_opt = min(safe, key=lambda x: x[1]["t_dry_h"])
# консервативный = заметно холоднее оптимума (типичная практика «с запасом»)
r_cons = m.primary_drying(Ts_opt - 12, Pc_fixed, dT_super=10.0,
                          Tc_C=Tc, Tg_prime_C=Tg_prime)
print(f"  Оптимум без коллапса: Ts={Ts_opt:+.1f}°C, Tp_max={r_opt['Tp_max']:.1f}°C "
      f"(Tc={Tc}), t={r_opt['t_dry_h']:.1f} ч")
print(f"  Типичный «с запасом» Ts={Ts_opt-12:+.1f}°C: Tp_max={r_cons['Tp_max']:.1f}°C, "
      f"t={r_cons['t_dry_h']:.1f} ч")
print(f"  -> работа у самой границы Tc вместо избыточного запаса экономит "
      f"{100*(1-r_opt['t_dry_h']/r_cons['t_dry_h']):.0f}% времени")

# ------------------------------- ГРАФИКИ ----------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

ax[0].plot(dT_list, times_A, "o-", color="tab:blue")
ax[0].set_xlabel("степень переохлаждения ΔT_supercool, °C")
ax[0].set_ylabel("время первичной сушки, ч")
ax[0].set_title("A. Заморозка -> сушка\n(крупные кристаллы = быстрее)")
ax[0].grid(alpha=0.3)
ax[0].annotate("контролируемая\nнуклеация", (dT_list[0], times_A[0]),
               xytext=(6, times_A[0]+ (times_A.max()-times_A.min())*0.15),
               arrowprops=dict(arrowstyle="->"))

for Ts, r in runs_B.items():
    ax[1].plot(r["t"]/3600, r["Tp"], label=f"Tp, Ts={Ts:+d}°C")
ax[1].axhline(Tc, color="red", ls="--", label=f"Tc коллапса={Tc}°C")
ax[1].set_xlabel("время, ч"); ax[1].set_ylabel("T продукта на фронте Tp, °C")
ax[1].set_title("B. Чем выше полка, тем\nближе Tp к коллапсу")
ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)

ok = ~coll_sweep
ax[2].plot(tp_sweep[ok], t_sweep[ok], "o-", color="tab:green",
           label="режим без коллапса")
ax[2].plot(tp_sweep[~ok], t_sweep[~ok], "x", color="tab:red",
           label="коллапс (брак)")
ax[2].axvline(Tc, color="red", ls="--", label=f"Tc={Tc}°C")
ax[2].scatter([r_opt["Tp_max"]], [r_opt["t_dry_h"]], s=140,
              facecolors="none", edgecolors="k", linewidths=2, zorder=5,
              label=f"оптимум {r_opt['t_dry_h']:.0f} ч")
ax[2].set_xlabel("макс. Tp за цикл, °C")
ax[2].set_ylabel("время первичной сушки, ч")
ax[2].set_title("C. Компромисс время–дефекты\nоптимум у границы Tc")
ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results.png", dpi=130)
print("\nГрафики сохранены в results.png")
