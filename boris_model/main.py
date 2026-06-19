"""
Управляющий модуль (ТЗ: "модуль управления").

- задаёт входные параметры (Params, литературные дефолты);
- строит сетку (Ts_lim [K], Pch [Torr]) в заданных рамках;
- прогоняет ОБА модуля (1.4 теплопроводность и 1.5 квазиравновесие) в каждой
  точке сетки;
- строит временные графики (1.3.1–1.3.3) для репрезентативной точки;
- строит 2.5D-карты (1.3.4 A,B,C,D) для каждого модуля;
- сохраняет сводку в CSV.

Температуры на графиках — в °C. Запуск:  uv run python main.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import physics as ph
import model_quasi_equilibrium as qe
import model_conduction as cd

K0 = 273.15
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {"quasi_equilibrium": qe.run, "conduction": cd.run}


# ----------------------------- сетка --------------------------------------
def build_grid(Ts_lim_C=(-35.0, 10.0), n_Ts=10, Pch_torr=(0.05, 0.30), n_Pc=7):
    Ts_C = np.linspace(*Ts_lim_C, n_Ts)
    Pch = np.linspace(*Pch_torr, n_Pc)
    return Ts_C, Pch


def run_grid(run_fn, Ts_C, Pch, p: ph.Params):
    """Прогон одного модуля по всей сетке. Возвращает 2D массивы (Pch×Ts)."""
    nP, nT = len(Pch), len(Ts_C)
    t_dry = np.zeros((nP, nT))
    Tp_max = np.zeros((nP, nT))
    rate = np.zeros((nP, nT))
    for i, pc in enumerate(Pch):
        for j, tsC in enumerate(Ts_C):
            res = run_fn(tsC + K0, pc, p)
            t_dry[i, j] = res["t_dry_h"]
            Tp_max[i, j] = res["Tp_max_K"] - K0
            rate[i, j] = res["rate_mean_g_h"]
    return dict(t_dry=t_dry, Tp_max=Tp_max, rate=rate, Ts_C=Ts_C, Pch=Pch)


# -------------------- временные графики (1.3.1–1.3.3) ---------------------
def plot_timeseries(results, path, title):
    """results: dict {model_name: res}. 3 панели: T мин/макс, градиент, Ts."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    colors = {"quasi_equilibrium": "tab:blue", "conduction": "tab:red"}
    for name, r in results.items():
        c = colors.get(name, "k")
        ax[0].plot(r["t_h"], r["Tmin_K"] - K0, color=c, lw=2, label=f"{name}: Tmin (фронт)")
        ax[0].plot(r["t_h"], r["Tmax_K"] - K0, color=c, ls="--", lw=1.8, label=f"{name}: Tmax (дно)")
        ax[1].plot(r["t_h"], np.asarray(r["grad_max"]) / 100, color=c, lw=2, label=f"{name}: max dT/dz")
        ax[1].plot(r["t_h"], np.asarray(r["grad_min"]) / 100, color=c, ls="--", lw=1.8, label=f"{name}: min dT/dz")
        ax[2].plot(r["t_h"], r["Ts_K"] - K0, color=c, lw=2, label=f"{name}: Ts")
    ax[0].set_xlabel("время, ч"); ax[0].set_ylabel("температура продукта, °C")
    ax[0].set_title("1.3.1 Tmin / Tmax продукта"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("время, ч"); ax[1].set_ylabel("градиент dT/dz, K/см")
    ax[1].set_title("1.3.2 градиент по высоте"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    ax[2].set_xlabel("время, ч"); ax[2].set_ylabel("температура полки, °C")
    ax[2].set_title("1.3.3 температура полки"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    fig.suptitle(title)
    fig.savefig(path, dpi=130); plt.close(fig)


# ----------------------- 2.5D карты (1.3.4) -------------------------------
def _tc_boundary_pch(grid, Tc_C):
    """Для каждой Ts найти Pch, где Tp_max пересекает Tc (для линии в D)."""
    Ts_C, Pch, Tp = grid["Ts_C"], grid["Pch"], grid["Tp_max"]
    out = []
    for j, tsC in enumerate(Ts_C):
        col = Tp[:, j]                                  # Tp_max(Pch) при данной Ts
        pc_cross = np.nan
        for i in range(len(Pch) - 1):
            a, b = col[i] - Tc_C, col[i + 1] - Tc_C
            if a == 0:
                pc_cross = Pch[i]; break
            if a * b < 0:                               # смена знака -> интерполяция
                pc_cross = Pch[i] + (Pch[i + 1] - Pch[i]) * (-a) / (b - a)
                break
        out.append(pc_cross)
    return np.array(out)


def plot_maps(grid, Tc_C, path, title):
    Ts_C, Pch = grid["Ts_C"], grid["Pch"]
    X, Y = np.meshgrid(Ts_C, Pch)
    fig, ax = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    # A: время сушки
    cA = ax[0, 0].contourf(X, Y, grid["t_dry"], 18, cmap="viridis")
    fig.colorbar(cA, ax=ax[0, 0], label="время сушки, ч")
    ax[0, 0].set_title("A. Время сушки [цвет]")
    ax[0, 0].set_xlabel("Ts_lim, °C"); ax[0, 0].set_ylabel("Pch, Torr")

    # B: макс. температура продукта + линия Tc
    cB = ax[0, 1].contourf(X, Y, grid["Tp_max"], 18, cmap="magma")
    fig.colorbar(cB, ax=ax[0, 1], label="max Tp, °C")
    cl = ax[0, 1].contour(X, Y, grid["Tp_max"], levels=[Tc_C], colors="cyan", linewidths=2)
    ax[0, 1].clabel(cl, fmt=f"Tc={Tc_C:.0f}°C")
    ax[0, 1].set_title("B. Макс. температура продукта [цвет]")
    ax[0, 1].set_xlabel("Ts_lim, °C"); ax[0, 1].set_ylabel("Pch, Torr")

    # C: скорость сублимации vs Pch, семейство по Ts (цвет)
    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(Ts_C)))
    for j, tsC in enumerate(Ts_C):
        ax[1, 0].plot(Pch, grid["rate"][:, j], "-o", ms=3, color=cmap[j],
                      label=f"Ts={tsC:.0f}" if j % 2 == 0 else None)
    ax[1, 0].set_xlabel("Pch, Torr"); ax[1, 0].set_ylabel("ср. скорость сублимации, г/(ч·виал)")
    ax[1, 0].set_title("C. Скорость сублимации (цвет = Ts)")
    ax[1, 0].legend(fontsize=7, ncol=2); ax[1, 0].grid(alpha=0.3)

    # D: то же + граница max(Tp)=Tc (область допустимого)
    for j, tsC in enumerate(Ts_C):
        ax[1, 1].plot(Pch, grid["rate"][:, j], "-", color=cmap[j], alpha=0.8)
    pc_cross = _tc_boundary_pch(grid, Tc_C)
    bx, by = [], []
    for j, tsC in enumerate(Ts_C):
        if np.isfinite(pc_cross[j]):
            rate_at = np.interp(pc_cross[j], Pch, grid["rate"][:, j])
            bx.append(pc_cross[j]); by.append(rate_at)
    if bx:
        order = np.argsort(bx)
        ax[1, 1].plot(np.array(bx)[order], np.array(by)[order], "k--o", lw=2,
                      label=f"граница max(Tp)=Tc={Tc_C:.0f}°C")
        ax[1, 1].legend(fontsize=8)
    else:
        ax[1, 1].text(0.5, 0.95, f"во всей сетке max(Tp) не пересекает Tc={Tc_C:.0f}°C",
                      transform=ax[1, 1].transAxes, ha="center", va="top", fontsize=9)
    ax[1, 1].set_xlabel("Pch, Torr"); ax[1, 1].set_ylabel("ср. скорость сублимации, г/(ч·виал)")
    ax[1, 1].set_title("D. То же + граница коллапса (ниже-левее линии — допустимо)")
    ax[1, 1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.savefig(path, dpi=130); plt.close(fig)


def save_csv(grid, path, model_name):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "Ts_lim_C", "Pch_torr", "t_dry_h", "Tp_max_C", "rate_mean_g_h"])
        for i, pc in enumerate(grid["Pch"]):
            for j, tsC in enumerate(grid["Ts_C"]):
                w.writerow([model_name, f"{tsC:.3f}", f"{pc:.4f}",
                            f"{grid['t_dry'][i, j]:.4f}", f"{grid['Tp_max'][i, j]:.3f}",
                            f"{grid['rate'][i, j]:.5f}"])


# --------------------------------- main -----------------------------------
def main():
    p = ph.Params()
    print(f"Параметры: виала 10cc (Ap={p.Ap_cm2} Av={p.Av_cm2} см²), L={p.L_cm} см, "
          f"cs={p.cs}, Tc={p.Tc_C}°C, dt={p.dt_h} ч, стоп при {p.end_frac*100:.0f}% воды")

    # --- репрезентативная точка для временных кривых ---
    Ts_demo_C, Pch_demo = 0.0, 0.15
    ts_results = {name: fn(Ts_demo_C + K0, Pch_demo, p) for name, fn in MODELS.items()}
    plot_timeseries(ts_results, os.path.join(HERE, "timeseries.png"),
                    f"Временные кривые (Ts_lim={Ts_demo_C:.0f}°C, Pch={Pch_demo} Torr)")
    for name, r in ts_results.items():
        print(f"  [{name}] t={r['t_dry_h']:.2f} ч, max Tp={r['Tp_max_K']-K0:.1f}°C, "
              f"ср.скорость={r['rate_mean_g_h']:.3f} г/(ч·виал)")

    # --- сетка: оба модуля ---
    Ts_C, Pch = build_grid()
    print(f"Сетка: Ts_lim {Ts_C[0]:.0f}..{Ts_C[-1]:.0f}°C ({len(Ts_C)}), "
          f"Pch {Pch[0]:.2f}..{Pch[-1]:.2f} Torr ({len(Pch)}). Прогон обоих модулей...")
    grids = {}
    for name, fn in MODELS.items():
        grids[name] = run_grid(fn, Ts_C, Pch, p)
        plot_maps(grids[name], p.Tc_C, os.path.join(HERE, f"maps_{name}.png"),
                  f"2.5D карты — модуль: {name}")
        save_csv(grids[name], os.path.join(HERE, f"grid_{name}.csv"), name)
        print(f"  [{name}] карты -> maps_{name}.png, CSV -> grid_{name}.csv")

    # --- сравнение моделей по времени сушки ---
    dt_diff = np.abs(grids["conduction"]["t_dry"] - grids["quasi_equilibrium"]["t_dry"])
    print(f"Макс. расхождение времени сушки между модулями: {dt_diff.max():.3f} ч "
          f"({100*dt_diff.max()/grids['quasi_equilibrium']['t_dry'].mean():.1f}% от среднего).")
    print("Готово. Файлы: timeseries.png, maps_*.png, grid_*.csv")


if __name__ == "__main__":
    main()
