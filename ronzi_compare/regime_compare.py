"""
Почему у Ronzi (Fig. 3 a,b) Tp первичной сушки ~постоянна и НЕ зависит от полки,
а у нас раньше росла. Ответ: режим лимитирования.

  Tp = Tice( Pc + Js·Rp )           (Js·Rp -- избыток над порогом Tice(Pc))

 * Высокое Rp -> МАССОЛИМИТ: Js·Rp велик и растёт -> Tp ЛЕЗЕТ вверх и зависит
   от полки («не так»).
 * Низкое Rp -> ТЕПЛОЛИМИТ: Js·Rp мал -> Tp прилипает к порогу, ~плоская и
   почти не зависит от полки; полка меняет лишь СКОРОСТЬ (= Ronzi Fig. 3 a,b).

Левая панель — «не так» (высокое Rp). Правая — «так, как Ronzi» (низкое Rp).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lyo_model as m

Pc_torr = 0.20
Pc = Pc_torr * m.TORR
cs = 0.10
plates = [-20, -10, 0, 5]

def floor(Pc_torr):
    Pc = Pc_torr * m.TORR; lo, hi = -90, 20
    for _ in range(60):
        mid = (lo + hi) / 2
        if m.p_ice(mid) < Pc: lo = mid
        else: hi = mid
    return hi
fl = floor(Pc_torr)

def runs(r_pore):
    out = []
    for Ts in plates:
        rp = lambda H, r=r_pore: m.Rp_knudsen_areal(H, cs, r)
        res = m.primary_drying(Ts, Pc_torr, rp_func=rp, Tc_C=0, Tg_prime_C=-2)
        out.append((Ts, res))
    return out

hi_Rp = runs(6e-6)     # «не так» — массолимит
lo_Rp = runs(300e-6)   # «так» — теплолимит (как Ronzi)

plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True, sharey=True)
colors = plt.cm.viridis(np.linspace(0, 0.85, len(plates)))

def panel(a, data, title, sub):
    for (Ts, res), c in zip(data, colors):
        a.plot(res["t"]/3600, res["Tp"], color=c, lw=2.3,
               label=f"Ts={Ts:+d}°C  (t={res['t_dry_h']:.0f} ч)")
    a.axhline(fl, color="gray", ls=":", lw=1.5, label=f"порог Tice(Pc)={fl:.1f}°C")
    a.set_xlabel("время первичной сушки, ч")
    a.set_title(title + "\n" + sub)
    a.legend(loc="upper right", fontsize=9); a.grid(alpha=0.3)

panel(ax[0], hi_Rp,
      "A. «НЕ так»: высокое Rp (массолимит)",
      "Tp лезет вверх и зависит от полки")
ax[0].set_ylabel("Tp фронта, °C")
panel(ax[1], lo_Rp,
      "B. «ТАК, как Ronzi»: низкое Rp (теплолимит)",
      "Tp ~плоская у порога, полка меняет лишь СКОРОСТЬ")

plt.savefig(os.path.join(os.path.dirname(__file__), "regime_compare.png"), dpi=130)
print("Сохранено: regime_compare.png\n")
for name, data in [("НЕ так (высокое Rp)", hi_Rp), ("ТАК как Ronzi (низкое Rp)", lo_Rp)]:
    print(name + ":")
    for Ts, res in data:
        print(f"   Ts={Ts:+3d}: Tp {res['Tp'][0]:.1f}->{res['Tp'][-1]:.1f} "
              f"(размах {res['Tp'][-1]-res['Tp'][0]:.1f}°C), t={res['t_dry_h']:.0f} ч")
