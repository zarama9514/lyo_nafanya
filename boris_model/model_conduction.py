"""
Модуль 1.4 — решение задачи теплопроводности.

В каждом интервале решается нестационарное 1-D уравнение теплопроводности в
столбе льда (z=0 дно … z=L−l фронт), теплопроводностью сухого кейка пренебрегаем
(ТЗ, Сл.9). Граничные условия:
  z=0 (дно):   −k ∂T/∂z = (Kv·Av/Ap)·(Ts − T0)        нагреватель с сопротивлением
  z=L−l (фронт): −k ∂T/∂z = ΔHs·Js,  Js=(Pice(Ti)−Pch)/(Rp+Rs)   охлаждение сублимацией

Схема — неявная (backward Euler), трёхдиагональная прогонка (безусловно
устойчива). Нелинейность на фронте (Js зависит от Ti) снимается итерациями Пикара.
Сетка из n_layers слоёв перестраивается по мере роста l (домен укорачивается),
профиль переинтерполируется. Из температуры фронта Ti считается рост l на шаг.
"""
import numpy as np
import physics as ph


def _thomas(a, b, c, d):
    """Прогонка для трёхдиагональной системы (a-низ, b-диаг, c-верх, d-правая)."""
    n = len(b)
    cp = np.zeros(n); dp = np.zeros(n)
    cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _implicit_substep(T, Ts_K, Pch_torr, l_m, p: ph.Params, dt_s, k, alpha, h_bot):
    """Один неявный подшаг (backward Euler, трёхдиаг.). Поток сублимации на
    фронте берётся из температуры начала подшага (ТЗ 1.4: l и сублимация
    «замораживаются» на интервале, переносятся в следующий шаг)."""
    N = len(T)
    L_ice = max(p.L - l_m, 1e-5)
    dz = L_ice / (N - 1)
    r = alpha * dt_s / dz ** 2
    Js = ph.subl_flux_areal(T[-1], Pch_torr, l_m, p)   # явный фронт из T начала
    q_front = ph.delta_Hs(T[-1]) * Js

    a = np.zeros(N); b = np.zeros(N); c = np.zeros(N); d = np.zeros(N)
    a[1:-1] = -r; b[1:-1] = 1 + 2 * r; c[1:-1] = -r; d[1:-1] = T[1:-1]
    b[0] = 1 + 2 * r + 2 * r * h_bot * dz / k           # дно (Robin, нагреватель)
    c[0] = -2 * r
    d[0] = T[0] + 2 * r * h_bot * dz / k * Ts_K
    a[-1] = -2 * r                                      # фронт (охлаждение сублимацией)
    b[-1] = 1 + 2 * r
    d[-1] = T[-1] - 2 * r * q_front * dz / k
    return _thomas(a, b, c, d)


def step_conduction(T, Ts_K, Pch_torr, l_m, p: ph.Params, dt_s):
    """Шаг по интервалу dt_s (ТЗ 1.4). Решается нестационарная теплопроводность
    в столбе льда; при необходимости интервал автоматически дробится на более
    мелкие подшаги (ТЗ/презентация Сл.10) для точности явного фронта.
    Возвращает (T_new, Ti, Tb, Js) — Ti=фронт (верх), Tb=дно (низ)."""
    N = len(T)
    L_ice = max(p.L - l_m, 1e-5)
    dz = L_ice / (N - 1)
    Tref = float(np.mean(T))
    k = ph.k_ice(Tref); alpha = k / (ph.RHO_ICE * ph.cp_ice(Tref))
    h_bot = ph.Kv_si(Pch_torr, p) * p.Av / p.Ap
    # авто-дробление (ТЗ/Сл.10): дробим только когда нестационарность значима на
    # масштабе ВСЕГО столба льда (Fo_столба ~ 1), т.е. в конце сушки (тонкий лёд).
    # Неявная схема безусловно устойчива, поэтому по-слойное Fo дробить не нужно.
    Fo_col = alpha * dt_s / L_ice ** 2
    n_sub = int(min(max(1, np.ceil(Fo_col / 0.5)), 300))
    sub = dt_s / n_sub
    for _ in range(n_sub):
        T = _implicit_substep(T, Ts_K, Pch_torr, l_m, p, sub, k, alpha, h_bot)

    Ti = float(T[-1]); Tb = float(T[0])
    Js = ph.subl_flux_areal(Ti, Pch_torr, l_m, p)
    return T, Ti, Tb, Js


def _remap(T, N):
    """Переинтерполировать профиль на N узлов (после изменения высоты домена)."""
    if len(T) == N:
        return T
    xs_old = np.linspace(0, 1, len(T))
    xs_new = np.linspace(0, 1, N)
    return np.interp(xs_new, xs_old, T)


def run(Ts_lim_K, Pch_torr, p: ph.Params):
    """Симуляция первичной сушки (теплопроводность). Возвращает ряды и сводку."""
    N = p.n_layers
    t = 0.0
    l = 0.0
    Tfr = p.Tfreeze_C + 273.15
    T = np.full(N, Tfr)                            # старт: равновесие = Tfreeze
    rows_t, rows_Ts, rows_Tmin, rows_Tmax = [], [], [], []
    rows_gmin, rows_gmax, rows_rate = [], [], []

    while ph.water_remaining_frac(l, p) > p.end_frac and t < p.t_max_h * 3600:
        Ts_mid = 0.5 * (ph.shelf_temp(t, p, Ts_lim_K)
                        + ph.shelf_temp(t + p.dt, p, Ts_lim_K))
        T, Ti, Tb, Js = step_conduction(T, Ts_mid, Pch_torr, l, p, p.dt)
        L_ice = max(p.L - l, 1e-5)
        dz = L_ice / (N - 1)
        grad = np.gradient(T, dz)                   # К/м по высоте

        rows_t.append(t / 3600.0)
        rows_Ts.append(Ts_mid)
        rows_Tmin.append(float(np.min(T)))
        rows_Tmax.append(float(np.max(T)))
        rows_gmin.append(float(np.min(grad)))
        rows_gmax.append(float(np.max(grad)))
        rows_rate.append(ph.subl_rate_g_per_h_vial(Js, p))

        l += ph.dl_from_subl(Js, p.dt, p)
        T = _remap(T, N)                            # домен укоротился — переинтерп.
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
        model="conduction",
    )
