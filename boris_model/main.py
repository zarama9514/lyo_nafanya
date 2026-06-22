"""
Управляющий модуль.

Запуск по умолчанию открывает Tkinter-интерфейс:
    uv run python boris_model/main.py

Пакетный режим:
    uv run python boris_model/main.py --cli
"""
import csv
import os
import pickle
import shutil
import sys
import threading
from dataclasses import fields
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import physics as ph
import model_quasi_equilibrium as qe
import model_conduction as cd

K0 = 273.15
HERE = os.path.dirname(os.path.abspath(__file__))
OPTIONS_PATH = os.path.join(HERE, "options.pkl")
OUTPUT_DIR = os.path.join(HERE, "output_data")
MODELS = {"quasi_equilibrium": qe.run, "conduction": cd.run}

PARAM_GROUPS = {
    "Параметры аппаратуры": [
        "Ap_cm2", "Av_cm2", "Kv_direct", "Kc", "Kd", "condenser_kg_h", "n_vials",
    ],
    "Параметры образца": [
        "Tc_C", "L_cm", "cs", "rho_sol", "R0", "A1", "A2", "Rs",
    ],
    "Параметры процесса": [
        "Ts_min_C", "Ts_max_C", "n_Ts", "Pch_min_torr", "Pch_max_torr", "n_Pc",
        "map_levels", "Tfreeze_C", "ramp_K_per_min", "n_layers", "dt_h",
        "end_frac", "t_max_h",
    ],
}

DEFAULT_CURVE_TS_C = 0.0
DEFAULT_CURVE_PCH_TORR = 0.15


def ensure_output_dir(path=OUTPUT_DIR):
    os.makedirs(path, exist_ok=True)
    return path


def default_sample_name():
    return datetime.now().strftime("%d.%m.%y %H-%M")


def load_options():
    current_names = {f.name for f in fields(ph.Params)}
    if os.path.exists(OPTIONS_PATH):
        with open(OPTIONS_PATH, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, ph.Params):
            return ph.Params(**{k: getattr(obj, k) for k in current_names if hasattr(obj, k)})
        if isinstance(obj, dict):
            return ph.Params(**{k: v for k, v in obj.items() if k in current_names})
    p = ph.Params()
    save_options(p)
    return p


def save_options(p):
    with open(OPTIONS_PATH, "wb") as f:
        pickle.dump(p, f)


def build_grid(p: ph.Params):
    Ts_C = np.linspace(p.Ts_min_C, p.Ts_max_C, int(p.n_Ts))
    Pch = np.linspace(max(0.01, p.Pch_min_torr), p.Pch_max_torr, int(p.n_Pc))
    return Ts_C, Pch


def run_grid(run_fn, Ts_C, Pch, p: ph.Params):
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


def make_timeseries_figure(results, title, figsize=(10.5, 5.0)):
    fig, ax = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    colors = {"quasi_equilibrium": "tab:blue", "conduction": "tab:red"}
    for name, r in results.items():
        c = colors.get(name, "k")
        ax[0].plot(r["t_h"], r["Tmin_K"] - K0, color=c, lw=2, label=f"{name}: Tmin")
        ax[0].plot(r["t_h"], r["Tmax_K"] - K0, color=c, ls="--", lw=1.8, label=f"{name}: Tmax")
        ax[1].plot(r["t_h"], np.asarray(r["grad_max"]) / 100, color=c, lw=2, label=f"{name}: max dT/dz")
        ax[1].plot(r["t_h"], np.asarray(r["grad_min"]) / 100, color=c, ls="--", lw=1.8, label=f"{name}: min dT/dz")
    first = next(iter(results.values()))
    ax[0].plot(first["t_h"], first["Ts_K"] - K0, color="tab:green", lw=2.0, label="Ts полки")
    ax[0].set_xlabel("время, ч"); ax[0].set_ylabel("температура, °C")
    ax[0].set_title("Температура продукта и полки"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("время, ч"); ax[1].set_ylabel("градиент dT/dz, K/см")
    ax[1].set_title("Градиент по высоте"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    dry_times = [r["t_dry_h"] for r in results.values()]
    fig.suptitle(f"{title}\nПолное время сушки: {max(dry_times):.2f} ч")
    return fig


def plot_timeseries(results, path, title, p: ph.Params):
    fig = make_timeseries_figure(results, title, figsize=(13, 4.8))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _tc_boundary_pch(grid, Tc_C):
    Ts_C, Pch, Tp = grid["Ts_C"], grid["Pch"], grid["Tp_max"]
    out = []
    for j, _ in enumerate(Ts_C):
        col = Tp[:, j]
        pc_cross = np.nan
        for i in range(len(Pch) - 1):
            a, b = col[i] - Tc_C, col[i + 1] - Tc_C
            if a == 0:
                pc_cross = Pch[i]; break
            if a * b < 0:
                pc_cross = Pch[i] + (Pch[i + 1] - Pch[i]) * (-a) / (b - a)
                break
        out.append(pc_cross)
    return np.array(out)


def _extended_line(xs, ys, xmin, xmax):
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if len(xs) < 2:
        return xs, ys
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    return np.array([xmin, *xs, xmax]), np.interp([xmin, *xs, xmax], xs, ys)


def tang_pikal_pch_torr(Tp_C):
    return 0.29 * 10 ** (0.019 * Tp_C)


def sublimation_rate_for_tp_pch(Tp_C, Pch_torr, p: ph.Params, l_m=0.0):
    Js = ph.subl_flux_areal(Tp_C + K0, Pch_torr, l_m, p)
    return ph.subl_rate_g_per_h_vial(Js, p)


def _decorate_map_axes(ax):
    ax.set_xlabel("Ts_lim, °C")
    ax.set_ylabel("Pch, Torr")


def _plot_panel(ax, fig, grid, panel, Tc_C, levels, p: ph.Params, add_legend=True):
    Ts_C, Pch = grid["Ts_C"], grid["Pch"]
    X, Y = np.meshgrid(Ts_C, Pch)
    cmap = "coolwarm"
    if panel == "A":
        c = ax.contourf(X, Y, grid["t_dry"], levels, cmap=cmap)
        fig.colorbar(c, ax=ax, label="время сушки, ч")
        ax.set_title("A. Время сушки")
        _decorate_map_axes(ax)
    elif panel == "B":
        c = ax.contourf(X, Y, grid["Tp_max"], levels, cmap=cmap)
        fig.colorbar(c, ax=ax, label="max Tp, °C")
        cl = ax.contour(X, Y, grid["Tp_max"], levels=[Tc_C], colors="white", linewidths=2)
        ax.clabel(cl, fmt=f"Tc={Tc_C:.0f}°C")
        ax.set_title("B. Макс. температура продукта")
        _decorate_map_axes(ax)
    elif panel in ("C", "D"):
        line_colors = plt.cm.coolwarm(np.linspace(0, 1, len(Ts_C)))
        for j, tsC in enumerate(Ts_C):
            label = f"Ts={tsC:.0f}" if panel == "C" and j % max(1, len(Ts_C) // 6) == 0 else None
            ax.plot(Pch, grid["rate"][:, j], "-o" if panel == "C" else "-",
                    ms=3, color=line_colors[j], alpha=0.9, label=label)
        ax.set_xlabel("Pch, Torr")
        ax.set_ylabel("ср. скорость сублимации, г/(ч·виал)")
        ax.set_title("C. Скорость сублимации" if panel == "C" else "D. Рабочая область")
        ax.grid(alpha=0.3)
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(Ts_C.min(), Ts_C.max()))
        fig.colorbar(sm, ax=ax, label="Ts_lim, °C")
        if panel == "C" and add_legend:
            ax.legend(fontsize=7, ncol=2)
        if panel == "D":
            tp_star_C = Tc_C
            pc_star = tang_pikal_pch_torr(tp_star_C)
            y_star = sublimation_rate_for_tp_pch(tp_star_C, pc_star, p, l_m=0.0)
            data_max = float(max(np.nanmax(grid["rate"]), y_star))
            data_pad = max(0.05, 0.12 * data_max)
            pc_cross = _tc_boundary_pch(grid, Tc_C)
            bx, by = [], []
            for j, pc in enumerate(pc_cross):
                if np.isfinite(pc):
                    bx.append(pc)
                    by.append(np.interp(pc, Pch, grid["rate"][:, j]))
            ex, ey = _extended_line(bx, by, float(Pch.min()), float(Pch.max()))
            if len(ex):
                ax.plot(ex, ey, "k--", lw=2.2, label=f"max(Tp)=Tc={Tc_C:.0f}°C")
            cap_g_h_vial = p.condenser_kg_h * 1000.0 / max(int(p.n_vials), 1)
            cap_line = cap_g_h_vial
            cap_label = "макс. массопоток оборудования"
            if cap_g_h_vial > max(data_max * 4.0, data_max + data_pad):
                cap_line = data_max + data_pad
                cap_label = f"макс. массопоток оборудования: {cap_g_h_vial:.1f} г/ч/виал (выше шкалы)"
            ax.axhline(cap_line, color="black", ls=":", lw=2, label=cap_label)
            ax.scatter([pc_star], [y_star], marker="*", s=190, facecolor="white",
                       edgecolor="black", linewidth=1.1, zorder=10,
                       label="Tang & Pikal: Pch(Tp=Tc), l=0")
            ax.set_xlim(float(Pch.min()), float(Pch.max()))
            ax.set_ylim(min(-data_pad, np.nanmin(grid["rate"]) - data_pad), data_max + 1.7 * data_pad)
            ax.legend(fontsize=8)


def plot_maps(grid, Tc_C, path, title, p: ph.Params, panels_dir=None):
    levels = max(2, int(p.map_levels))
    fig, ax = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    for panel, axis in zip(("A", "B", "C", "D"), ax.flat):
        _plot_panel(axis, fig, grid, panel, Tc_C, levels, p)
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    if panels_dir:
        os.makedirs(panels_dir, exist_ok=True)
        for panel in ("A", "B", "C", "D"):
            fig1, ax1 = plt.subplots(figsize=(8, 6), constrained_layout=True)
            _plot_panel(ax1, fig1, grid, panel, Tc_C, levels, p)
            fig1.suptitle(f"{title}: {panel}")
            fig1.savefig(os.path.join(panels_dir, f"map_{panel}.png"), dpi=260)
            plt.close(fig1)


def plot_d_figure(grid, Tc_C, p: ph.Params, title="Карта D"):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    _plot_panel(ax, fig, grid, "D", Tc_C, max(2, int(p.map_levels)), p)
    fig.suptitle(title)
    return fig


def save_csv(grid, path, model_name):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "Ts_lim_C", "Pch_torr", "t_dry_h", "Tp_max_C", "rate_mean_g_h"])
        for i, pc in enumerate(grid["Pch"]):
            for j, tsC in enumerate(grid["Ts_C"]):
                w.writerow([model_name, f"{tsC:.3f}", f"{pc:.4f}",
                            f"{grid['t_dry'][i, j]:.4f}", f"{grid['Tp_max'][i, j]:.3f}",
                            f"{grid['rate'][i, j]:.5f}"])


def calculate_all(p: ph.Params, out_dir=OUTPUT_DIR, make_gif=True,
                  curve_Ts_C=DEFAULT_CURVE_TS_C, curve_Pch_torr=DEFAULT_CURVE_PCH_TORR):
    ensure_output_dir(out_dir)
    Ts_C, Pch = build_grid(p)
    grids = {}
    for name, fn in MODELS.items():
        grids[name] = run_grid(fn, Ts_C, Pch, p)
        panel_dir = os.path.join(out_dir, f"maps_{name}_single")
        plot_maps(grids[name], p.Tc_C, os.path.join(out_dir, f"maps_{name}.png"),
                  f"2.5D карты — модуль: {name}", p, panels_dir=panel_dir)
        save_csv(grids[name], os.path.join(out_dir, f"grid_{name}.csv"), name)
    if make_gif:
        cd.make_combined_gif(
            curve_Ts_C + K0, curve_Pch_torr, p,
            path=os.path.join(out_dir, "vial_combined_model.gif"),
        )
    return dict(grids=grids, out_dir=out_dir)


def calculate_timeseries_only(p: ph.Params, Ts_C, Pch_torr, out_dir=OUTPUT_DIR):
    ensure_output_dir(out_dir)
    ts_results = {
        name: fn(Ts_C + K0, Pch_torr, p)
        for name, fn in MODELS.items()
    }
    plot_timeseries(
        ts_results,
        os.path.join(out_dir, "timeseries.png"),
        f"Временные кривые (Ts_lim={Ts_C:.1f}°C, Pch={Pch_torr:.3f} Torr)",
        p,
    )
    return ts_results


def copy_results_to_sample_folder(sample_name, source_dir=OUTPUT_DIR):
    sample_name = sample_name.strip() or default_sample_name()
    target = os.path.join(HERE, sample_name)
    os.makedirs(target, exist_ok=True)
    for name in os.listdir(source_dir):
        src = os.path.join(source_dir, name)
        dst = os.path.join(target, name)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    return target


class App:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("Модель первичной сушки")
        self.p = load_options()
        self.data = None
        self.sample_var = tk.StringVar(value=default_sample_name())
        self.ts_var = tk.StringVar(value=f"{DEFAULT_CURVE_TS_C:g}")
        self.pch_var = tk.StringVar(value=f"{DEFAULT_CURVE_PCH_TORR:g}")
        self.status_var = tk.StringVar(value="Расчет не выполнен. Нажмите «Запустить расчет».")
        self.busy = False
        self.param_buttons = []
        self.action_buttons = []

        self.left = ttk.Frame(root, padding=10)
        self.left.pack(side="left", fill="y")
        self.right = ttk.Frame(root, padding=10)
        self.right.pack(side="right", fill="both", expand=True)
        self._build_params()
        self._build_results()

    def _build_params(self):
        self.param_buttons = []
        for group, names in PARAM_GROUPS.items():
            frame = self.ttk.LabelFrame(self.left, text=group, padding=8)
            frame.pack(fill="x", pady=(0, 8))
            for name in names[:8]:
                self.ttk.Label(frame, text=f"{name}: {getattr(self.p, name)}").pack(anchor="w")
            if len(names) > 8:
                self.ttk.Label(frame, text=f"... еще {len(names) - 8}").pack(anchor="w")
            btn = self.ttk.Button(frame, text="Изменить", command=lambda g=group: self.edit_group(g))
            btn.pack(fill="x", pady=(6, 0))
            self.param_buttons.append(btn)

    def _build_results(self):
        top = self.ttk.Frame(self.right)
        top.pack(fill="both", expand=True)
        self.plot_holder = self.ttk.Frame(top)
        self.plot_holder.pack(fill="both", expand=True)
        self.ttk.Label(self.right, text="Название образца").pack(anchor="w", pady=(10, 0))
        self.ttk.Entry(self.right, textvariable=self.sample_var).pack(fill="x")
        controls = self.ttk.Frame(self.right)
        controls.pack(fill="x", pady=8)
        self.ttk.Label(controls, text="Ts, °C").pack(side="left")
        self.ttk.Entry(controls, width=8, textvariable=self.ts_var).pack(side="left", padx=(4, 12))
        self.ttk.Label(controls, text="Pch, Torr").pack(side="left")
        self.ttk.Entry(controls, width=8, textvariable=self.pch_var).pack(side="left", padx=(4, 12))
        run_btn = self.ttk.Button(controls, text="Запустить расчет", command=self.recalculate)
        run_btn.pack(side="left", padx=(0, 8))
        curve_btn = self.ttk.Button(controls, text="Получить кривую", command=self.show_timeseries)
        curve_btn.pack(side="left", padx=(0, 8))
        save_btn = self.ttk.Button(controls, text="Сохранить данные", command=self.save_sample)
        save_btn.pack(side="left")
        self.action_buttons.extend([run_btn, curve_btn, save_btn])
        self.ttk.Label(self.right, textvariable=self.status_var).pack(anchor="w")

    def set_busy(self, busy, message=None):
        self.busy = busy
        state = "disabled" if busy else "normal"
        live_buttons = []
        for btn in [*self.param_buttons, *self.action_buttons]:
            try:
                if btn.winfo_exists():
                    btn.configure(state=state)
                    live_buttons.append(btn)
            except self.tk.TclError:
                pass
        self.param_buttons = [btn for btn in self.param_buttons if btn in live_buttons]
        self.action_buttons = [btn for btn in self.action_buttons if btn in live_buttons]
        if message:
            self.status_var.set(message)

    def edit_group(self, group):
        win = self.tk.Toplevel(self.root)
        win.title(group)
        win.transient(self.root)
        win.grab_set()
        entries = {}
        for row, name in enumerate(PARAM_GROUPS[group]):
            self.ttk.Label(win, text=name).grid(row=row, column=0, sticky="w", padx=8, pady=4)
            ent = self.ttk.Entry(win, width=22)
            ent.insert(0, "" if getattr(self.p, name) is None else str(getattr(self.p, name)))
            ent.grid(row=row, column=1, padx=8, pady=4)
            entries[name] = ent
        def save():
            values = {f.name: getattr(self.p, f.name) for f in fields(ph.Params)}
            for name, ent in entries.items():
                text = ent.get().strip()
                old = getattr(self.p, name)
                if text == "":
                    values[name] = None if old is None or name == "Kv_direct" else old
                elif isinstance(old, int):
                    values[name] = int(text)
                elif old is None and name == "Kv_direct":
                    values[name] = float(text)
                else:
                    values[name] = float(text)
            self.p = ph.Params(**values)
            save_options(self.p)
            win.destroy()
            for child in self.left.winfo_children():
                child.destroy()
            self._build_params()
            self.data = None
            self.status_var.set("Параметры сохранены. Нажмите «Запустить расчет».")
        self.ttk.Button(win, text="Сохранить", command=save).grid(row=len(entries), column=0, columnspan=2, sticky="ew", padx=8, pady=8)

    def recalculate(self):
        if self.busy:
            return
        self.set_busy(True, "Считаю карты в фоне...")
        p_snapshot = ph.Params(**{f.name: getattr(self.p, f.name) for f in fields(ph.Params)})
        curve_Ts_C = float(self.ts_var.get())
        curve_Pch_torr = float(self.pch_var.get())

        def work():
            try:
                data = calculate_all(p_snapshot, OUTPUT_DIR, make_gif=False,
                                     curve_Ts_C=curve_Ts_C,
                                     curve_Pch_torr=curve_Pch_torr)
            except Exception as exc:
                message = str(exc)
                self.root.after(0, lambda: self.finish_with_error(message))
                return
            self.root.after(0, lambda: self.finish_recalculate(data))

        threading.Thread(target=work, daemon=True).start()

    def finish_recalculate(self, data):
        self.data = data
        self.draw_d()
        self.set_busy(False, f"Готово: {OUTPUT_DIR}. GIF будет создан при сохранении данных.")

    def finish_with_error(self, message):
        self.set_busy(False, f"Ошибка расчета: {message}")

    def draw_d(self):
        if not self.data:
            return
        for child in self.plot_holder.winfo_children():
            child.destroy()
        fig = plot_d_figure(self.data["grids"]["conduction"], self.p.Tc_C, self.p, "D. Теплопроводность")
        canvas = FigureCanvasTkAgg(fig, master=self.plot_holder)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.configure(cursor="hand2")
        widget.pack(fill="both", expand=True)
        widget.bind("<Button-1>", lambda _e: self.open_interactive_d())
        plt.close(fig)

    def open_interactive_d(self):
        if not self.data:
            return
        win = self.tk.Toplevel(self.root)
        win.title("График D")
        fig = plot_d_figure(self.data["grids"]["conduction"], self.p.Tc_C, self.p, "D. Теплопроводность")
        canvas = FigureCanvasTkAgg(fig, master=win)
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def show_timeseries(self):
        if self.busy:
            return
        Ts_C = float(self.ts_var.get())
        Pch_torr = float(self.pch_var.get())
        self.set_busy(True, "Считаю временную кривую...")
        p_snapshot = ph.Params(**{f.name: getattr(self.p, f.name) for f in fields(ph.Params)})

        def work():
            try:
                results = calculate_timeseries_only(p_snapshot, Ts_C, Pch_torr, OUTPUT_DIR)
            except Exception as exc:
                message = str(exc)
                self.root.after(0, lambda: self.finish_with_error(message))
                return
            self.root.after(0, lambda: self.open_timeseries_window(results, Ts_C, Pch_torr))

        threading.Thread(target=work, daemon=True).start()

    def open_timeseries_window(self, results, Ts_C, Pch_torr):
        self.set_busy(False, f"Готово: {OUTPUT_DIR}")
        win = self.tk.Toplevel(self.root)
        win.title("Временные кривые")
        win.geometry("1100x650")
        title = f"Временные кривые (Ts_lim={Ts_C:.1f}°C, Pch={Pch_torr:.3f} Torr)"
        fig = make_timeseries_figure(results, title, figsize=(10.5, 5.0))
        canvas = FigureCanvasTkAgg(fig, master=win)
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def save_sample(self):
        if self.busy:
            return
        if self.data is None:
            self.status_var.set("Сначала нажмите «Запустить расчет».")
            return
        self.set_busy(True, "Готовлю GIF и сохраняю данные...")
        name = self.sample_var.get()
        p_snapshot = ph.Params(**{f.name: getattr(self.p, f.name) for f in fields(ph.Params)})
        Ts_C = float(self.ts_var.get())
        Pch_torr = float(self.pch_var.get())

        def work():
            try:
                gif_path = os.path.join(OUTPUT_DIR, "vial_combined_model.gif")
                cd.make_combined_gif(Ts_C + K0, Pch_torr, p_snapshot, path=gif_path)
                target = copy_results_to_sample_folder(name, OUTPUT_DIR)
            except Exception as exc:
                message = str(exc)
                self.root.after(0, lambda: self.finish_with_error(message))
                return
            self.root.after(0, lambda: self.finish_save(target))

        threading.Thread(target=work, daemon=True).start()

    def finish_save(self, target):
        self.set_busy(False, f"Сохранено: {target}")


def cli_main():
    p = load_options()
    print(f"Параметры: виала 10cc (Ap={p.Ap_cm2} Av={p.Av_cm2} см²), L={p.L_cm} см, "
          f"cs={p.cs}, Tc={p.Tc_C}°C, dt={p.dt_h} ч, стоп при {p.end_frac*100:.0f}% воды")
    result = calculate_all(p, OUTPUT_DIR, make_gif=True)
    grids = result["grids"]
    dt_diff = np.abs(grids["conduction"]["t_dry"] - grids["quasi_equilibrium"]["t_dry"])
    print(f"Макс. расхождение времени сушки между модулями: {dt_diff.max():.3f} ч")
    print(f"Готово. Файлы сохранены в {OUTPUT_DIR}")


def gui_main():
    import tkinter as tk
    root = tk.Tk()
    root.geometry("1280x820")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        cli_main()
    else:
        gui_main()
