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


def run(Ts_lim_K, Pch_torr, p: ph.Params, record_profiles=False):
    """Симуляция первичной сушки (теплопроводность). Возвращает ряды и сводку.
    record_profiles=True -> дополнительно сохраняет профиль T по высоте, толщину
    сухого кейка l и время на каждом шаге (для анимации температурной карты)."""
    N = p.n_layers
    t = 0.0
    l = 0.0
    Tfr = p.Tfreeze_C + 273.15
    T = np.full(N, Tfr)                            # старт: равновесие = Tfreeze
    rows_t, rows_Ts, rows_Tmin, rows_Tmax = [], [], [], []
    rows_gmin, rows_gmax, rows_rate = [], [], []
    prof_T, prof_l = [], []                        # для анимации (К, м)

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
        if record_profiles:
            prof_T.append(T.copy()); prof_l.append(l)

        l += ph.dl_from_subl(Js, p.dt, p)
        T = _remap(T, N)                            # домен укоротился — переинтерп.
        t += p.dt

    arr = lambda x: np.asarray(x)
    Tmax = arr(rows_Tmax)
    out = dict(
        t_h=arr(rows_t), Ts_K=arr(rows_Ts),
        Tmin_K=arr(rows_Tmin), Tmax_K=arr(rows_Tmax),
        grad_min=arr(rows_gmin), grad_max=arr(rows_gmax),
        rate_g_h=arr(rows_rate),
        t_dry_h=t / 3600.0,
        Tp_max_K=float(np.max(Tmax)) if len(Tmax) else np.nan,
        rate_mean_g_h=float(np.mean(rows_rate)) if rows_rate else 0.0,
        model="conduction",
    )
    if record_profiles:
        out["profiles_K"] = prof_T          # список профилей (К), узлы 0..N-1
        out["profiles_l"] = arr(prof_l)     # толщина сухого кейка, м
    return out


def make_temperature_gif(Ts_lim_K, Pch_torr, p: ph.Params, path="vial_temperature.gif",
                         fps=10, duration_s=20, width=100, height=400,
                         cmap="coolwarm"):
    """GIF температурной карты в виале (расчёт model_conduction).
    Прямоугольник width×height px без рамок: низ = дно виалы (тёплое), верх =
    фронт сублимации / сухой кейк (холодное). Сине-красная гамма (синий — холод,
    красный — тепло), фиксированная шкала по всему процессу.
    fps×duration_s кадров (по умолчанию 10×20 = 200)."""
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("нужен Pillow: uv add pillow")

    res = run(Ts_lim_K, Pch_torr, p, record_profiles=True)
    profiles = res["profiles_K"]            # список (К)
    ls = res["profiles_l"]                  # толщина кейка, м
    n_steps = len(profiles)
    if n_steps == 0:
        raise RuntimeError("нет шагов симуляции (сушка не идёт при этих параметрах)")

    # фиксированная цветовая шкала (°C) по всему процессу
    allT = np.concatenate([pr for pr in profiles]) - 273.15
    norm = Normalize(vmin=float(allT.min()), vmax=float(allT.max()))
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    n_frames = int(round(fps * duration_s))
    idx = np.linspace(0, n_steps - 1, n_frames).round().astype(int)
    z_full = np.linspace(0.0, p.L, height)  # высота виалы -> пиксели (0=дно)

    frames = []
    for k in idx:
        prof = profiles[k] - 273.15         # °C, узлы 0..N-1 по льду 0..(L-l)
        l = ls[k]
        L_ice = max(p.L - l, 1e-9)
        z_ice = np.linspace(0.0, L_ice, len(prof))
        col = np.empty(height)
        in_ice = z_full <= L_ice
        col[in_ice] = np.interp(z_full[in_ice], z_ice, prof)  # лёд: профиль
        col[~in_ice] = prof[-1]             # сухой кейк выше фронта: T фронта (Ti)
        img2d = np.repeat(col[:, None], width, axis=1)        # (height, width)
        rgba = (mapper.to_rgba(img2d) * 255).astype(np.uint8) # цвет
        rgb = rgba[..., :3]
        rgb = np.flipud(rgb)                # row0 -> верх изображения (z=L сверху)
        frames.append(Image.fromarray(rgb, mode="RGB"))

    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    return path, res


def make_profile_gif(Ts_lim_K, Pch_torr, p: ph.Params, path="vial_profile.gif",
                     fps=10, duration_s=20):
    """GIF линейного профиля температуры по высоте (расчёт model_conduction):
    ось X — координата z (см, 0 = дно виалы), ось Y — температура продукта (°C).
    Сплошная линия — лёд (0..L-l), пунктир — сухой кейк выше фронта (при T фронта).
    Оси фиксированы по всему процессу. fps×duration_s кадров (10×20 = 200)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("нужен Pillow: uv add pillow")

    res = run(Ts_lim_K, Pch_torr, p, record_profiles=True)
    profiles = res["profiles_K"]; ls = res["profiles_l"]; ts = res["t_h"]
    n_steps = len(profiles)
    if n_steps == 0:
        raise RuntimeError("нет шагов симуляции (сушка не идёт при этих параметрах)")

    allT = np.concatenate([pr for pr in profiles]) - 273.15
    ymin, ymax = float(allT.min()), float(allT.max())
    pad = 0.05 * (ymax - ymin + 1e-6)
    L_cm = p.L_cm
    n_frames = int(round(fps * duration_s))
    idx = np.linspace(0, n_steps - 1, n_frames).round().astype(int)

    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=110)
    frames = []
    for k in idx:
        prof = profiles[k] - 273.15
        l_cm = ls[k] * 100.0
        L_ice_cm = max(L_cm - l_cm, 1e-6)
        z_ice = np.linspace(0.0, L_ice_cm, len(prof))
        ax.clear()
        ax.plot(z_ice, prof, "-", color="tab:red", lw=2.4, label="лёд")
        if l_cm > 1e-4:                                  # сухой кейк выше фронта
            ax.plot([L_ice_cm, L_cm], [prof[-1], prof[-1]], "--",
                    color="tab:gray", lw=1.8, label="сухой кейк (T фронта)")
        ax.axvline(L_ice_cm, color="tab:blue", ls=":", lw=1.2)  # фронт сублимации
        ax.set_xlim(0, L_cm); ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlabel("координата z, см (0 = дно виалы)")
        ax.set_ylabel("температура продукта, °C")
        ax.set_title(f"Профиль T(z), t = {ts[k]:.2f} ч")
        ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(Image.fromarray(buf, mode="RGB"))
    plt.close(fig)

    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    return path, res


def make_combined_gif(Ts_lim_K, Pch_torr, p: ph.Params, path="vial_combined.gif",
                      fps=10, duration_s=20, vial_w=100, vial_h=400, cmap="coolwarm"):
    """Совмещённый GIF: СЛЕВА — температурная карта виалы (vial_w×vial_h px, низ =
    дно/тёплое, верх = фронт/сухой кейк, сине-красная гамма), СПРАВА — линейный
    профиль T(z) (ось X = z, см; ось Y = T продукта, °C). Обе панели синхронны.
    fps×duration_s кадров (10×20 = 200)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("нужен Pillow: uv add pillow")

    res = run(Ts_lim_K, Pch_torr, p, record_profiles=True)
    profiles = res["profiles_K"]; ls = res["profiles_l"]; ts = res["t_h"]
    n_steps = len(profiles)
    if n_steps == 0:
        raise RuntimeError("нет шагов симуляции (сушка не идёт при этих параметрах)")

    allT = np.concatenate([pr for pr in profiles]) - 273.15
    vmin, vmax = float(allT.min()), float(allT.max())
    pad = 0.05 * (vmax - vmin + 1e-6)
    norm = Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    L_cm = p.L_cm
    n_frames = int(round(fps * duration_s))
    idx = np.linspace(0, n_steps - 1, n_frames).round().astype(int)
    z_full = np.linspace(0.0, p.L, vial_h)

    fig = plt.figure(figsize=(9.6, 4.4), dpi=110)
    gs = GridSpec(1, 3, width_ratios=[1.0, 0.10, 3.2], wspace=0.6)
    ax_vial = fig.add_subplot(gs[0, 0])
    ax_cb = fig.add_subplot(gs[0, 1])
    ax_plot = fig.add_subplot(gs[0, 2])
    fig.colorbar(mapper, cax=ax_cb, label="T, °C")

    frames = []
    for k in idx:
        prof = profiles[k] - 273.15
        l = ls[k]; l_cm = l * 100.0
        L_ice = max(p.L - l, 1e-9); L_ice_cm = max(L_cm - l_cm, 1e-6)
        # --- левая панель: карта ---
        z_ice = np.linspace(0.0, L_ice, len(prof))
        col = np.empty(vial_h)
        in_ice = z_full <= L_ice
        col[in_ice] = np.interp(z_full[in_ice], z_ice, prof)
        col[~in_ice] = prof[-1]
        img2d = np.repeat(col[:, None], vial_w, axis=1)
        ax_vial.clear()
        ax_vial.imshow(img2d, origin="lower", cmap=cmap, norm=norm,
                       aspect="auto", extent=[0, 1, 0, L_cm])
        ax_vial.set_xticks([]); ax_vial.set_ylabel("z, см (0 = дно)")
        ax_vial.set_title("виала")
        # --- правая панель: профиль ---
        z_ice_cm = np.linspace(0.0, L_ice_cm, len(prof))
        ax_plot.clear()
        ax_plot.plot(z_ice_cm, prof, "-", color="tab:red", lw=2.4, label="лёд")
        if l_cm > 1e-4:
            ax_plot.plot([L_ice_cm, L_cm], [prof[-1], prof[-1]], "--",
                         color="tab:gray", lw=1.8, label="сухой кейк (T фронта)")
        ax_plot.axvline(L_ice_cm, color="tab:blue", ls=":", lw=1.2)
        ax_plot.set_xlim(0, L_cm); ax_plot.set_ylim(vmin - pad, vmax + pad)
        ax_plot.set_xlabel("координата z, см (0 = дно виалы)")
        ax_plot.set_ylabel("температура продукта, °C")
        ax_plot.set_title(f"профиль T(z),  t = {ts[k]:.2f} ч")
        ax_plot.grid(alpha=0.3); ax_plot.legend(loc="upper right", fontsize=9)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(Image.fromarray(buf, mode="RGB"))
    plt.close(fig)

    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    return path, res


if __name__ == "__main__":
    import os
    p = ph.Params()
    here = os.path.dirname(os.path.abspath(__file__))
    path1, res = make_temperature_gif(0 + 273.15, 0.15, p,
                                      os.path.join(here, "vial_temperature.gif"))
    path2, _ = make_profile_gif(0 + 273.15, 0.15, p,
                                os.path.join(here, "vial_profile.gif"))
    path3, _ = make_combined_gif(0 + 273.15, 0.15, p,
                                 os.path.join(here, "vial_combined.gif"))
    print(f"GIF (совмещённый): {path3}")
    print(f"GIF (карта):    {path1}")
    print(f"GIF (профиль):  {path2}")
    print(f"  время сушки {res['t_dry_h']:.2f} ч, шагов {len(res['profiles_K'])}, "
          f"кадров 200 (10 fps × 20 с)")
    print(f"  диапазон T: {min(pr.min() for pr in res['profiles_K'])-273.15:.1f} .. "
          f"{max(pr.max() for pr in res['profiles_K'])-273.15:.1f} °C")
