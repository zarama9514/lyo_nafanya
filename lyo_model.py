"""
Физически обоснованная модель первичной сушки при лиофилизации.

Базовая модель -- Tang & Pikal (Pharm. Res. 2004), квазистационарная,
сосредоточенная (0-D на флакон):
    dm/dt = (P_ice - P_c) / (R_p + R_s)            (массоперенос)
    dQ/dt = A_v K_v (T_s - T_b)                     (теплоперенос флакон<-полка)
    dQ/dt = dHs * dm/dt                             (энергобаланс на фронте)
    T_s   = T_p + (1/A_v) dQ/dt (1/K_v + l_ice/k1)  (нужная T полки)

Что здесь улучшено (физически обоснованно):
 1. Подвижный фронт сублимации (задача Стефана), 1-D теплопроводность через
    замороженный слой -> ЯВНЫЙ градиент температуры T_b - T_p, меняющийся во
    времени, а не одна алгебраическая поправка.
 2. R_p(H) -- сопротивление сухого слоя растёт с его толщиной; его параметры
    СВЯЗАНЫ с режимом заморозки через размер кристаллов льда.
 3. Связь заморозка -> сушка: степень переохлаждения / температура нуклеации
    задаёт размер кристаллов d; R_p ~ 1/d (удельная поверхность пор).
 4. Критерий коллапса -- вязкостный (время вязкого закрытия поры против
    времени локальной сушки), Tc привязан к локальной концентрации (Tg').

Единицы -- СИ (Па, кг, с, м, К) внутри; пользовательский ввод в °C/Torr.
"""

import numpy as np

# ------------------------- физические константы --------------------------
DHSUB = 2.84e6        # теплота сублимации льда, Дж/кг
K_ICE = 2.30          # теплопроводность льда, Вт/(м*К)
RHO_ICE_CAKE = 900.0  # масса льда на единицу объёма замороженного коржа, кг/м3
TORR = 133.322        # Па/Torr
# перевод Rp из (Torr*cm2*h/g) в СИ (Pa*m2*s/kg):
RP_TO_SI = TORR * 1e-4 * 3600.0 / 1e-3      # = 4.7996e4
CAL = 4.184           # Дж/кал


def p_ice(T_C):
    """Давление насыщ. пара надо льдом, Па (Murphy & Koop 2005). T в °C."""
    T = T_C + 273.15
    return np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T)
                  - 0.00728332 * T)


def Kv_of_Pc(Pc_torr, KC=2.64e-4, KD=3.64):
    """Коэф. теплопередачи флакон<-полка, Вт/(м2*К). Eq.(6) Tang&Pikal,
    10cc tubing. P в Torr. Возврат в СИ."""
    Kv_cal = KC + 3.32e-3 * Pc_torr / (1.0 + KD * Pc_torr)  # кал/(с*см2*К)
    return Kv_cal * CAL * 1e4                                # Вт/(м2*К)


# ------------------- связь ЗАМОРОЗКА -> сопротивление --------------------
def crystal_factor(dT_super, dT_ref=10.0, exponent=0.5):
    """Множитель сопротивления f = (dTsc/dTref)^p.
    Меньше переохлаждение (контролируемая нуклеация) -> крупнее кристаллы ->
    меньше удельная поверхность пор -> меньше R_p (f<1).
    p~0.5 соответствует дендритному масштабу λ ~ ΔT^(-1/2)."""
    return (dT_super / dT_ref) ** exponent


def Rp_areal(H, Rp0, A1, A2, fcryst):
    """Площадно-нормированное сопротивление сухого слоя, СИ.
    H -- толщина сухого слоя [м]. Параметры в (Torr*cm2*h/g), H_cm = H*100.
    fcryst масштабирует R_p0 и A1 (морфология льда из заморозки)."""
    Hc = H * 100.0
    Rp_tp = fcryst * Rp0 + fcryst * A1 * Hc / (1.0 + A2 * Hc)
    return Rp_tp * RP_TO_SI


# ----------------------------- коллапс -----------------------------------
def viscosity_amorphous(T_C, Tg_prime_C, eta_g=1e12, C1=17.4, C2=51.6):
    """Вязкость аморфной фазы (WLF) над Tg'. eta_g~1e12 Па*с в стекле."""
    dT = T_C - Tg_prime_C
    if dT <= -C2 + 1e-6:
        return np.inf
    log_aT = -C1 * dT / (C2 + dT)     # WLF
    return eta_g * 10.0 ** log_aT


def collapse_number(T_C, Tg_prime_C, Tc_C):
    """Число коллапса Co = eta(Tc)/eta(Tp) -- отношение текучестей.
    Привязано к ИЗМЕРЯЕМОЙ температуре коллапса Tc (DSC/FDM, как требуют обе
    статьи): Co=1 ровно при Tp=Tc, Co>1 при Tp>Tc (матрица текучее, чем на
    пороге коллапса -> вязкое закрытие пор быстрее ухода фронта -> коллапс).
    Механизм -- вязкое течение freeze-concentrate; Tc есть точка, где время
    вязкого течения сравнивается с временем сушки."""
    eta_tp = viscosity_amorphous(T_C, Tg_prime_C)
    eta_tc = viscosity_amorphous(Tc_C, Tg_prime_C)
    return eta_tc / eta_tp


# ----------------- УЛУЧШЕННАЯ динамическая модель сушки -------------------
def solve_step(H, Ts_C, Pc_torr, L0, Rp0, A1, A2, fcryst, Ap, Av):
    """Решает квазистационарную систему на текущей толщине сухого слоя H.
    Возвращает (Tp_C, Tb_C, Js) -- Js = поток массы на ед. площади [кг/(м2 с)].

    Уравнения (на единицу площади продукта):
      Js = (P_ice(Tp) - Pc) / Rp(H)
      Js*dHs = Kv*(Av/Ap)*(Ts - Tb)
      Js*dHs = (k_ice/(L0-H))*(Tb - Tp)
    """
    Kv = Kv_of_Pc(Pc_torr)
    Pc = Pc_torr * TORR
    Rp = Rp_areal(H, Rp0, A1, A2, fcryst)
    L_froz = max(L0 - H, 1e-5)

    def residual(Tp):
        # residual растёт с Tp: Tb_cond растёт, Tb_shelf падает
        Js = (p_ice(Tp) - Pc) / Rp
        q = Js * DHSUB
        Tb_cond = Tp + q * L_froz / K_ICE        # из теплопроводности льда
        Tb_shelf = Ts_C - q * Ap / (Av * Kv)     # из теплопередачи полки
        return Tb_cond - Tb_shelf

    # нижняя граница -- порог сублимации (где p_ice = Pc), иначе Js<=0
    tlo, thi = -90.0, Ts_C
    for _ in range(60):                            # найти Tp, где p_ice=Pc
        tm = 0.5 * (tlo + thi)
        if p_ice(tm) < Pc:
            tlo = tm
        else:
            thi = tm
    Tp_floor = thi + 1e-3
    lo, hi = Tp_floor, Ts_C
    if lo >= hi:                                   # полка ниже порога: нет сушки
        return Tp_floor, Tp_floor, 0.0
    flo, fhi = residual(lo), residual(hi)
    if flo * fhi > 0:
        Tp = lo if abs(flo) < abs(fhi) else hi
    else:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fm = residual(mid)
            if flo * fm <= 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        Tp = 0.5 * (lo + hi)

    Js = max((p_ice(Tp) - Pc) / Rp, 0.0)
    q = Js * DHSUB
    Tb = Tp + q * L_froz / K_ICE
    return Tp, Tb, Js


def primary_drying(Ts_C, Pc_torr, L0=0.01, Rp0=0.5, A1=20.0, A2=4.0,
                   dT_super=10.0, Ap=3.80e-4, Av=4.71e-4,
                   Tc_C=-32.0, Tg_prime_C=-34.0, dt=60.0, t_max=3e5):
    """Интегрирует первичную сушку до полного ухода льда (H -> L0).

    Ts_C, Pc_torr могут быть числом (постоянные) или callable(t).
    Возвращает словарь с траекториями и итоговым временем/риском коллапса.
    """
    fcryst = crystal_factor(dT_super)
    H = 0.0
    t = 0.0
    ts, Hs, Tps, Tbs, Js_s, Co_s = [], [], [], [], [], []

    def getval(x, t):
        return x(t) if callable(x) else x

    collapsed = False
    while H < L0 and t < t_max:
        Ts = getval(Ts_C, t)
        Pc = getval(Pc_torr, t)
        Tp, Tb, Js = solve_step(H, Ts, Pc, L0, Rp0, A1, A2, fcryst, Ap, Av)
        dHdt = Js / RHO_ICE_CAKE
        Co = collapse_number(Tp, Tg_prime_C, Tc_C)
        if Co >= 1.0:
            collapsed = True

        ts.append(t); Hs.append(H); Tps.append(Tp); Tbs.append(Tb)
        Js_s.append(Js); Co_s.append(Co)

        H += dHdt * dt
        t += dt

    return dict(t=np.array(ts), H=np.array(Hs), Tp=np.array(Tps),
                Tb=np.array(Tbs), Js=np.array(Js_s), Co=np.array(Co_s),
                t_dry_h=t / 3600.0, fcryst=fcryst, collapsed=collapsed,
                Tp_max=max(Tps) if Tps else np.nan, Tc=Tc_C)


# ------------- базовая (Tang&Pikal) стационарная оценка времени -----------
def tang_pikal_time(Tp_C, Pc_torr, L0=0.01, Rp0=0.5, A1=20.0, A2=4.0,
                    dT_super=10.0, Ap=3.80e-4):
    """Грубая оценка времени по Eq.(1): интеграл dt = rho/Js dH при ЗАДАННОЙ
    (постоянной целевой) Tp. Это «идеальная» оценка из статьи."""
    fcryst = crystal_factor(dT_super)
    Pc = Pc_torr * TORR
    n = 400
    Hgrid = np.linspace(0, L0, n)
    t = 0.0
    for i in range(n - 1):
        H = 0.5 * (Hgrid[i] + Hgrid[i + 1])
        Rp = Rp_areal(H, Rp0, A1, A2, fcryst)
        Js = max((p_ice(Tp_C) - Pc) / Rp, 1e-12)
        dHdt = Js / RHO_ICE_CAKE
        t += (Hgrid[i + 1] - Hgrid[i]) / dHdt
    return t / 3600.0
