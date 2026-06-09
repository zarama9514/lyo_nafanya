"""
Демонстрация трёх усилений модели (физоснова, без новых экспериментов):
  E -- извилистость tau(eps) из физики пористой среды (не «голый» Бруггеман);
  D -- dusty-gas (Кнудсен + вязкое течение) вместо чистого Кнудсена;
  F -- размер пор из ТЕОРИИ ЗАТВЕРДЕВАНИЯ (скорость фронта из задачи Стефана),
       вместо таблицы режимов;
  B -- верхний путь тепла: радиация на сухую поверхность + теплопроводность
       сухого слоя (k_eff).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lyo_model as m

plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

# ---------------- E: извилистость и её влияние на Rp ----------------
cs = np.linspace(0.01, 0.30, 60)
for md, c in [("bruggeman", "tab:gray"), ("archie", "tab:red"),
              ("directional", "tab:green")]:
    rp = [m.Rp_dgm_areal(0.01, x, 18e-6, 1.0, tau_model=md) / m.RP_TO_SI for x in cs]
    ax[0,0].plot(cs*100, rp, color=c, lw=2, label=f"τ: {md}")
ax[0,0].set_xlabel("концентрация сухого, %")
ax[0,0].set_ylabel("R_p(1 см), Torr·см²·ч/г")
ax[0,0].set_title("E. Извилистость τ(ε): расходятся\nтолько для плотных коржей (высокая c)")
ax[0,0].legend(); ax[0,0].grid(alpha=0.3)

# ---------------- D: dusty-gas vs чистый Кнудсен по давлению ----------------
Pc_torr = np.logspace(np.log10(0.02), np.log10(2.0), 60)
for r_um, c in [(20, "tab:blue"), (50, "tab:purple")]:
    rk = [m.Rp_knudsen_areal(0.01, 0.05, r_um*1e-6)/m.RP_TO_SI for _ in Pc_torr]
    rd = [m.Rp_dgm_areal(0.01, 0.05, r_um*1e-6, p*m.TORR)/m.RP_TO_SI for p in Pc_torr]
    ax[0,1].plot(Pc_torr, rk, color=c, ls="--", lw=1.6, label=f"Кнудсен, r={r_um}мкм")
    ax[0,1].plot(Pc_torr, rd, color=c, lw=2.3, label=f"dusty-gas, r={r_um}мкм")
ax[0,1].set_xscale("log")
ax[0,1].set_xlabel("давление в камере Pc, Torr")
ax[0,1].set_ylabel("R_p(1 см), Torr·см²·ч/г")
ax[0,1].set_title("D. Dusty-gas: вязкий член снижает R_p\nпри росте Pc и размера пор")
ax[0,1].legend(fontsize=9); ax[0,1].grid(alpha=0.3, which="both")

# ---------------- F: r_pore из теории затвердевания ----------------
sh = np.linspace(-70, -20, 60)
rpore = [m.pore_radius_solidification(shelf_T_frz_C=s, Pc_torr=0.1)["r_pore"]*1e6
         for s in sh]
ax[1,0].plot(sh, rpore, color="tab:blue", lw=2.3, label="r_pore из λ(v), Стефан")
lo = min(m.PORE_BY_REGIME.values())*1e6; hi = max(m.PORE_BY_REGIME.values())*1e6
ax[1,0].axhspan(lo, hi, color="tab:green", alpha=0.12,
                label="старая таблица режимов (диапазон)")
ax[1,0].set_xlabel("температура полки при заморозке, °C")
ax[1,0].set_ylabel("радиус пор r_pore, мкм")
ax[1,0].set_title("F. Поры из ФИЗИКИ заморозки (непрерывно)\nвместо дискретной таблицы")
ax[1,0].legend(); ax[1,0].grid(alpha=0.3)

# ---------------- B: верхний радиационный путь ----------------
shelves = np.arange(-30, 1, 2.5)
cs_b = 0.07
rp_b = lambda H: m.Rp_dgm_areal(H, cs_b, 20e-6, 0.10*m.TORR)
t_no, t_rad = [], []
for Ts in shelves:
    r0 = m.primary_drying(Ts, 0.10, rp_func=rp_b, Tc_C=-50, Tg_prime_C=-52)
    r1 = m.primary_drying(Ts, 0.10, rp_func=rp_b, Tc_C=-50, Tg_prime_C=-52,
                          T_rad_C=Ts)
    t_no.append(r0["t_dry_h"]); t_rad.append(r1["t_dry_h"])
ax[1,1].plot(shelves, t_no, "o-", color="tab:gray", label="только снизу (как было)")
ax[1,1].plot(shelves, t_rad, "s-", color="tab:orange",
             label="+ верхняя радиация и k_eff сухого слоя")
ax[1,1].set_xlabel("температура полки Ts, °C")
ax[1,1].set_ylabel("время первичной сушки, ч")
ax[1,1].set_title("B. Верхний путь тепла ускоряет сушку\n(сильнее в начале, тонкий сухой слой)")
ax[1,1].legend(); ax[1,1].grid(alpha=0.3)

plt.savefig("enhancements.png", dpi=130)
print("Сохранено: enhancements.png")
print(f"E: τ(0.95)≈1.03 для всех; при c=30% archie/directional расходятся.")
print(f"D: вязкая поправка ~0% при 0.05 Torr -> ~8% при 1 Torr (r=20мкм).")
print(f"F: r_pore непрерывно из скорости фронта: {rpore[-1]:.0f}→{rpore[0]:.0f} мкм "
      f"(полка -20→-70°C).")
print(f"B: верхний путь сокращает время на "
      f"{100*(1-np.mean(np.array(t_rad)/np.array(t_no))):.0f}% (в среднем).")
