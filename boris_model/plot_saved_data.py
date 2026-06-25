"""
Просмотр сохранённых данных графиков без повторного расчёта модели.

Примеры:
    python boris_model/plot_saved_data.py
    python boris_model/plot_saved_data.py boris_model/output_data/plot_data.pkl
"""
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

K0 = 273.15
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output_data")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def as_params(params_dict):
    return SimpleNamespace(**params_dict)


def equipment_rate(Pch, p):
    intercept = getattr(p, "equipment_rate_intercept", 0.0)
    slope = getattr(p, "equipment_rate_slope", 0.0)
    return np.maximum(intercept + slope * np.asarray(Pch), 0.0)


def equipment_points(p):
    return [(float(x), float(y)) for x, y in getattr(p, "equipment_limit_points", [])]


def tc_boundary_pch(grid, Tc_C):
    Ts_C, Pch, Tp = grid["Ts_C"], grid["Pch"], grid["Tp_max"]
    out = []
    for j, _ in enumerate(Ts_C):
        col = Tp[:, j]
        pc_cross = np.nan
        for i in range(len(Pch) - 1):
            a, b = col[i] - Tc_C, col[i + 1] - Tc_C
            if a == 0:
                pc_cross = Pch[i]
                break
            if a * b < 0:
                pc_cross = Pch[i] + (Pch[i + 1] - Pch[i]) * (-a) / (b - a)
                break
        out.append(pc_cross)
    return np.asarray(out)


def plot_panel(ax, fig, grid, panel, p):
    Ts_C, Pch = grid["Ts_C"], grid["Pch"]
    X, Y = np.meshgrid(Ts_C, Pch)
    levels = max(2, int(getattr(p, "map_levels", 18)))
    cmap = "coolwarm"
    if panel == "A":
        c = ax.contourf(X, Y, grid["t_dry"], levels, cmap=cmap)
        fig.colorbar(c, ax=ax, label="время сушки, ч")
        ax.set_title("A. Время сушки")
        ax.set_xlabel("Ts_lim, °C")
        ax.set_ylabel("Pch, Torr")
    elif panel == "B":
        c = ax.contourf(X, Y, grid["Tp_max"], levels, cmap=cmap)
        fig.colorbar(c, ax=ax, label="max Tp, °C")
        cl = ax.contour(X, Y, grid["Tp_max"], levels=[p.Tc_C], colors="white", linewidths=2)
        ax.clabel(cl, fmt=f"Tc={p.Tc_C:.0f}°C")
        ax.set_title("B. Макс. температура продукта")
        ax.set_xlabel("Ts_lim, °C")
        ax.set_ylabel("Pch, Torr")
    elif panel in ("C", "D"):
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(Ts_C)))
        for j in range(len(Ts_C) - 1):
            color = plt.cm.coolwarm((j + 0.5) / max(len(Ts_C) - 1, 1))
            ax.fill_between(Pch, grid["rate"][:, j], grid["rate"][:, j + 1],
                            color=color, alpha=0.42, linewidth=0)
        for j, tsC in enumerate(Ts_C):
            label = f"Ts={tsC:.0f}" if panel == "C" and j % max(1, len(Ts_C) // 6) == 0 else None
            ax.plot(Pch, grid["rate"][:, j], "-o" if panel == "C" else "-",
                    ms=3, color=colors[j], lw=1.6, label=label)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(Ts_C.min(), Ts_C.max()))
        fig.colorbar(sm, ax=ax, label="Ts_lim, °C")
        ax.set_xlabel("Pch, Torr")
        ax.set_ylabel("ср. скорость сублимации, г/(ч·виал)")
        ax.set_title("C. Скорость сублимации" if panel == "C" else "D. Рабочая область")
        ax.grid(alpha=0.3)
        if panel == "C":
            ax.legend(fontsize=7, ncol=2)
        if panel == "D":
            draw_d_overlays(ax, grid, p)


def draw_d_overlays(ax, grid, p):
    Pch = grid["Pch"]
    rate_max = float(np.nanmax(grid["rate"]))
    eq = equipment_rate(Pch, p)
    rate_max = max(rate_max, float(np.nanmax(eq)))
    points = equipment_points(p)
    if points:
        rate_max = max(rate_max, max(y for _, y in points))
    pad = max(0.05, 0.12 * rate_max)

    for value, style, label in [
        (p.Tc_C, "k-", f"max(Tp)=Tc={p.Tc_C:.0f}°C"),
        (p.Tc_C - getattr(p, "deltaTc_C", 2.0), "k--",
         f"безопасная max(Tp)=Tc-{getattr(p, 'deltaTc_C', 2.0):g}°C"),
    ]:
        pc_cross = tc_boundary_pch(grid, value)
        xs, ys = [], []
        for j, pc in enumerate(pc_cross):
            if np.isfinite(pc):
                xs.append(pc)
                ys.append(np.interp(pc, Pch, grid["rate"][:, j]))
        if xs:
            order = np.argsort(xs)
            ex, ey = np.asarray(xs)[order], np.asarray(ys)[order]
            ax.plot(ex, ey, style, lw=2.2, label=label)

    ax.plot(Pch, eq, color="red", ls="--", lw=2.2, label="предельный режим: rate(Pch)")
    if points:
        ax.scatter([x for x, _ in points], [y for _, y in points], s=48, color="red",
                   edgecolor="white", linewidth=0.8, zorder=11, label="эксп. точки предела")
    pc_star = 0.29 * 10 ** (0.019 * p.Tc_C)
    tc_cross = tc_boundary_pch(grid, p.Tc_C)
    bx, by = [], []
    for j, pc in enumerate(tc_cross):
        if np.isfinite(pc):
            bx.append(pc)
            by.append(np.interp(pc, Pch, grid["rate"][:, j]))
    if len(bx) >= 2:
        order = np.argsort(bx)
        ex, ey = np.asarray(bx)[order], np.asarray(by)[order]
        y_star = float(np.interp(pc_star, ex, ey))
    else:
        y_star = float(np.nanmean(grid["rate"]))
    ax.scatter([pc_star], [y_star], marker="*", s=190,
               facecolor="white", edgecolor="black", linewidth=1.1, zorder=10,
               label="Teng & Pikal Pch(Tc)")
    ax.set_xlim(float(Pch.min()), float(Pch.max()))
    ax.set_ylim(min(-pad, np.nanmin(grid["rate"]) - pad), rate_max + 1.7 * pad)
    ax.legend(fontsize=8)


def plot_maps(payload):
    p = as_params(payload["params"])
    for name, grid in payload["grids"].items():
        fig, ax = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
        for panel, axis in zip(("A", "B", "C", "D"), ax.flat):
            plot_panel(axis, fig, grid, panel, p)
        fig.suptitle(f"2.5D карты — модуль: {name}")


def plot_timeseries(payload):
    p = as_params(payload["params"])
    results = payload["results"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    colors = {"quasi_equilibrium": "tab:blue", "conduction": "tab:red"}
    for name, r in results.items():
        c = colors.get(name, "k")
        ax[0].plot(r["t_h"], r["Tmin_K"] - K0, color=c, lw=2, label=f"{name}: Tmin")
        ax[0].plot(r["t_h"], r["Tmax_K"] - K0, color=c, ls="--", lw=1.8, label=f"{name}: Tmax")
        ax[1].plot(r["t_h"], np.asarray(r["grad_max"]) / 100, color=c, lw=2, label=f"{name}: max dT/dz")
        ax[1].plot(r["t_h"], np.asarray(r["grad_min"]) / 100, color=c, ls="--", lw=1.8, label=f"{name}: min dT/dz")
    first = next(iter(results.values()))
    ax[0].plot(first["t_h"], first["Ts_K"] - K0, color="tab:green", lw=2, label="Ts полки")
    ax[0].set_xlabel("время, ч")
    ax[0].set_ylabel("температура, °C")
    ax[0].set_title("Температура продукта и полки")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("время, ч")
    ax[1].set_ylabel("градиент dT/dz, K/см")
    ax[1].set_title("Градиент по высоте")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    dry_times = [r["t_dry_h"] for r in results.values()]
    fig.suptitle(
        f"Временные кривые (Ts_lim={payload['Ts_C']:.1f}°C, Pch={payload['Pch_torr']:.3f} Torr)\n"
        f"Полное время сушки: {max(dry_times):.2f} ч"
    )


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUTPUT_DIR, "plot_data.pkl")
    payload = load_pickle(path)
    if payload.get("kind") == "boris_model_plot_data":
        plot_maps(payload)
        ts_path = os.path.join(os.path.dirname(path), "timeseries_data.pkl")
        if os.path.exists(ts_path):
            plot_timeseries(load_pickle(ts_path))
    elif payload.get("kind") == "boris_model_timeseries_data":
        plot_timeseries(payload)
    else:
        raise ValueError(f"Неизвестный формат данных: {path}")
    plt.show()


if __name__ == "__main__":
    main()
