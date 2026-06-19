"""
Модуль 1.5 — квазиравновесное приближение.

На каждом шаге считаем установившееся равновесие: тепло от полки = тепло на
сублимацию, а температуры дна (Tb) и фронта (Ti) связаны теплопроводностью через
столб льда (линейный градиент по высоте — асимптотика тонкого квазистационара):

  q = Kv·Av·(Ts − Tb)                         тепло снизу через дно виалы
  q = k_ice·Ap·(Tb − Ti)/(L − l)              кондукция дно → фронт по льду
  q = ΔHs(Ti)·Ap·(Pice(Ti) − Pch)/(Rp(l)+Rs)  сублимация на фронте

Три уравнения, неизвестные (Ti, Tb, q). Tb — максимальная температура продукта,
Ti — минимальная (фронт). Градиент постоянен по высоте: (Tb − Ti)/(L − l).
"""
import numpy as np
from scipy.optimize import brentq
import physics as ph


def solve_front_bottom(Ts_K, Pch_torr, l_m, p: ph.Params):
    """Решает (Ti, Tb, q, Js) на текущем шаге методом дихотомии по Ti."""
    Kv = ph.Kv_si(Pch_torr, p)
    L_ice = max(p.L - l_m, 1e-5)
    k = ph.k_ice(0.5 * (Ts_K))

    def Tb_of_Ti(Ti):
        Js = ph.subl_flux_areal(Ti, Pch_torr, l_m, p)
        q = ph.delta_Hs(Ti) * p.Ap * Js                 # Вт на виалу
        Tb = Ti + q * L_ice / (k * p.Ap)                # из кондукции
        return Tb, q, Js

    def residual(Ti):
        Tb, q, _ = Tb_of_Ti(Ti)
        return Kv * p.Av * (Ts_K - Tb) - q              # тепло полки − сублимация

    floor = ph.front_floor_K(Pch_torr) + 1e-3
    hi = Ts_K - 1e-4
    if hi <= floor:                                     # полка ниже порога: сушки нет
        return floor, floor, 0.0, 0.0
    f_lo, f_hi = residual(floor), residual(hi)
    if f_lo * f_hi > 0:                                 # нет смены знака — край
        Ti = floor if abs(f_lo) < abs(f_hi) else hi
    else:
        Ti = brentq(residual, floor, hi, xtol=1e-4, maxiter=200)
    Tb, q, Js = Tb_of_Ti(Ti)
    return Ti, Tb, q, Js


def run(Ts_lim_K, Pch_torr, p: ph.Params):
    """Симуляция первичной сушки (квазиравновесие). Возвращает словарь рядов и
    сводки (см. ТЗ 1.3)."""
    t = 0.0
    l = 0.0
    rows_t, rows_Ts, rows_Tmin, rows_Tmax = [], [], [], []
    rows_gmin, rows_gmax, rows_rate = [], [], []

    while ph.water_remaining_frac(l, p) > p.end_frac and t < p.t_max_h * 3600:
        # средняя температура полки за интервал (ТЗ: Ts const на шаге)
        Ts_mid = 0.5 * (ph.shelf_temp(t, p, Ts_lim_K)
                        + ph.shelf_temp(t + p.dt, p, Ts_lim_K))
        Ti, Tb, q, Js = solve_front_bottom(Ts_mid, Pch_torr, l, p)
        L_ice = max(p.L - l, 1e-5)
        grad = (Tb - Ti) / L_ice                        # К/м, постоянен по высоте

        rows_t.append(t / 3600.0)
        rows_Ts.append(Ts_mid)
        rows_Tmin.append(Ti)                            # фронт — самый холодный
        rows_Tmax.append(Tb)                            # дно — самый тёплый
        rows_gmin.append(grad); rows_gmax.append(grad)  # градиент постоянен
        rows_rate.append(ph.subl_rate_g_per_h_vial(Js, p))

        l += ph.dl_from_subl(Js, p.dt, p)               # рост кейка (с учётом cs)
        t += p.dt

    arr = lambda x: np.asarray(x)
    Tmax = arr(rows_Tmax)
    return dict(
        t_h=arr(rows_t), Ts_K=arr(rows_Ts),
        Tmin_K=arr(rows_Tmin), Tmax_K=arr(rows_Tmax),
        grad_min=arr(rows_gmin), grad_max=arr(rows_gmax),
        rate_g_h=arr(rows_rate),
        t_dry_h=t / 3600.0,
        Tp_max_K=float(np.max(Tmax)) if len(Tmax) else np.nan,
        rate_mean_g_h=float(np.mean(rows_rate)) if rows_rate else 0.0,
        model="quasi_equilibrium",
    )
