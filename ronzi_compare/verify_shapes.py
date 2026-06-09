"""
Проверка: «разные формы» кривых температуры продукта в ronzi-сравнении и в
корневом results.png — это ОДНА И ТА ЖЕ модель, а отличие вызвано лишь
(1) выбором величины (Tp фронта vs Tb дна) и (2) режимом полки (рампа vs const).

Левая панель  — режим Ronzi (полка РАМПИТСЯ): Tp и Tb растут, плато нет.
Правая панель — те же параметры, но полка ПОСТОЯННАЯ: Tp выходит на плато
                («экспонента с насыщением», как в корневом results.png).
В обоих случаях Tb просто на несколько °C выше Tp (между ними слой льда).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lyo_model as m

# те же параметры, что в калиброванном ronzi-прогоне
Pc = 0.15; L0 = 0.01; Ap = 1.0e-4; Av = 1.2e-4; cs = 0.07
r_pore = 40e-6; Tc, Tg = -20.0, -22.0
base_Kv = m.Kv_of_Pc
rp = lambda H: m.Rp_knudsen_areal(H, cs, r_pore)

def Ts_ramp(t):                       # рампа Ronzi (Табл. 4)
    h = t/3600.0
    if h < 4: return -30 + (15/4)*h
    if h < 8: return -15 + (10/4)*(h-4)
    return -5.0

m.Kv_of_Pc = lambda Pc, s=2.0: base_Kv(Pc)*s     # Kv×2 (поднос), как в калибровке

res_ramp = m.primary_drying(Ts_ramp, Pc, L0=L0, Ap=Ap, Av=Av, rp_func=rp,
                            Tc_C=Tc, Tg_prime_C=Tg, dt=60.0)
Ts_const = -20.0
res_const = m.primary_drying(Ts_const, Pc, L0=L0, Ap=Ap, Av=Av, rp_func=rp,
                             Tc_C=Tc, Tg_prime_C=Tg, dt=60.0)
m.Kv_of_Pc = base_Kv

print(f"Рампа:    t={res_ramp['t_dry_h']:.1f} ч, Tp_max={res_ramp['Tp_max']:.1f}°C")
print(f"Const:    t={res_const['t_dry_h']:.1f} ч, Tp_max={res_const['Tp_max']:.1f}°C")
print("Обе кривые — один и тот же primary_drying(); разница только в режиме полки.")

plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True, sharey=True)

# A: рампа
a = ax[0]
tr = res_ramp["t"]/3600
a.plot(tr, [Ts_ramp(t) for t in res_ramp["t"]], color="tab:red", lw=2, label="полка Ts (рампа)")
a.plot(tr, res_ramp["Tb"], color="tab:orange", lw=2.5, label="продукт дно Tb")
a.plot(tr, res_ramp["Tp"], color="tab:green", lw=2.5, label="фронт сублимации Tp")
a.fill_between(tr, res_ramp["Tp"], res_ramp["Tb"], color="tab:green", alpha=0.12,
               label="слой льда (Tb − Tp)")
a.set_xlabel("время, ч"); a.set_ylabel("температура, °C")
a.set_title(f"A. Полка РАМПИТСЯ (режим Ronzi)\nпродукт растёт за плитой, плато нет — t={res_ramp['t_dry_h']:.1f} ч")
a.legend(loc="lower right"); a.grid(alpha=0.3)

# B: постоянная полка
b = ax[1]
tc = res_const["t"]/3600
b.axhline(Ts_const, color="tab:red", lw=2, label=f"полка Ts = {Ts_const}°C (const)")
b.plot(tc, res_const["Tb"], color="tab:orange", lw=2.5, label="продукт дно Tb")
b.plot(tc, res_const["Tp"], color="tab:green", lw=2.5, label="фронт сублимации Tp")
b.fill_between(tc, res_const["Tp"], res_const["Tb"], color="tab:green", alpha=0.12,
               label="слой льда (Tb − Tp)")
b.set_xlabel("время, ч")
b.set_title(f"B. Полка ПОСТОЯННАЯ (как results.png)\nTp выходит на ПЛАТО — экспонента с насыщением, t={res_const['t_dry_h']:.1f} ч")
b.legend(loc="lower right"); b.grid(alpha=0.3)

out = os.path.join(os.path.dirname(__file__), "verify_shapes.png")
plt.savefig(out, dpi=130)
print("Сохранено:", out)
