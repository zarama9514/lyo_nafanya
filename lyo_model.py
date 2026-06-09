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
R_GAS = 8.314         # Дж/(моль*К)
M_WATER = 0.018015    # кг/моль
DHFUS_ICE = 334e3     # теплота плавления льда, Дж/кг (для модели заморозки, F)
MU_VAPOR = 9.0e-6     # вязкость водяного пара, Па*с (dusty-gas, D)
SIGMA_SB = 5.670e-8   # Стефан-Больцман, Вт/(м2*К4) (радиация, B)
RHO_ICE = 917.0       # плотность льда, кг/м3


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


# ========== ТЕОРЕТИЧЕСКИЙ R_p из МИКРОСТРУКТУРЫ (Кнудсен) ==========
# Не требует эксперимента на R_p: пористость -- из концентрации, размер пор --
# из режима заморозки, извилистость -- из пористости. Режим течения пара при
# ~10 Па -- кнудсеновский (длина своб. пробега >> размер пор).

# типичные радиусы пор (м) по протоколу заморозки (= ~ размер кристаллов льда):
PORE_BY_REGIME = {
    "controlled_nucleation": 40e-6,   # крупные кристаллы, низкое R_p
    "annealed":              35e-6,
    "slow_shelf":            22e-6,
    "normal":                18e-6,
    "fast":                   8e-6,
    "LN2":                    6e-6,    # мелкие кристаллы, высокое R_p
}


def porosity_from_conc(cs, rho_solid=1580.0, rho_water=1000.0):
    """Пористость сухого коржа = объёмная доля бывшего льда = 1 - φ_solid.
    cs -- масс. доля сухого в исходном растворе."""
    v_solid = cs / rho_solid
    v_water = (1.0 - cs) / rho_water
    phi_solid = v_solid / (v_solid + v_water)
    return 1.0 - phi_solid


def mean_speed(T_C):
    """Средняя тепловая скорость молекул воды, м/с."""
    T = T_C + 273.15
    return np.sqrt(8.0 * R_GAS * T / (np.pi * M_WATER))


def pore_radius(regime=None, cooling_rate=None):
    """Радиус пор r_pore [м]. Приоритет -- таблица режимов; иначе оценка по
    скорости охлаждения d ~ rate^(-0.4) (мелкие кристаллы при быстром охл.)."""
    if regime is not None:
        return PORE_BY_REGIME[regime]
    if cooling_rate is not None:               # rate в °C/мин
        # калибровка: ~1 °C/мин -> ~18 мкм (режим "normal")
        return 18e-6 * (1.0 / max(cooling_rate, 1e-3)) ** 0.4
    return PORE_BY_REGIME["normal"]


def Rp_knudsen_areal(H, cs, r_pore, tau=None, T_C=-25.0,
                     rho_solid=1580.0, Rp0_skin_torr=0.5):
    """Площадно-нормированное R_p [СИ] из кнудсеновской диффузии через корж.

      R_p(H) = R_p0_skin + (R*T)/(M*D_eff) * H
      D_eff  = (eps/tau) * D_K,   D_K = (2/3)*r_pore*v_mean
      eps    = пористость(cs),    tau = eps^(-1/2) (Бруггеман)

    Линейна по толщине H -> прямо даёт наклон R_p(H) без подгонки.
    R_p0_skin -- малый интерсепт (плотная корка на поверхности), Torr*cm2*h/g.
    """
    eps = porosity_from_conc(cs, rho_solid)
    if tau is None:
        tau = eps ** (-0.5)                    # Бруггеман
    Dk = (2.0 / 3.0) * r_pore * mean_speed(T_C)
    Deff = eps / tau * Dk
    T = T_C + 273.15
    slope = R_GAS * T / (M_WATER * Deff)        # СИ, на метр толщины
    return Rp0_skin_torr * RP_TO_SI + slope * H


def Rp_at_1cm_torr(cs, r_pore, **kw):
    """Удобный вывод: R_p при H=1 см в Torr*cm2*h/g (для сверки с литературой)."""
    return Rp_knudsen_areal(0.01, cs, r_pore, **kw) / RP_TO_SI


# ================== УСИЛЕНИЯ E, D, F (физоснова, без новых опытов) =========

# ---- E: извилистость из физики пористой среды (вместо «голого» Бруггемана) --
def tortuosity(eps, model="bruggeman"):
    """Извилистость tau(eps).
      bruggeman   -- tau = eps^(-1/2) (случайная среда; было по умолчанию);
      archie      -- tau = eps^(-3/2) (перколяционный, плотные структуры);
      directional -- tau ~ 1 + 0.5(1-eps) (КОЛОННЫЕ поры направленной заморозки:
                     каналы почти прямые -> низкая извилистость).
    При eps->1 все модели дают tau->1; различие важно лишь для плотных коржей."""
    if model == "bruggeman":
        return eps ** (-0.5)
    if model == "archie":
        return eps ** (-1.5)
    if model == "directional":
        return 1.0 + 0.5 * (1.0 - eps)
    raise ValueError(f"unknown tortuosity model: {model}")


# ---- D: dusty-gas (Кнудсен + вязкое пуазейлево течение) --------------------
def Rp_dgm_areal(H, cs, r_pore, Pmean_pa, T_C=-25.0, tau_model="directional",
                 rho_solid=1580.0, Rp0_skin_torr=0.5):
    """R_p [СИ] по dusty-gas: D_eff = (eps/tau)*D_K + B0*Pmean/mu.
      D_K = (2/3) r_pore v_mean       -- кнудсеновская диффузия
      B0  = eps r_pore^2/(8 tau)      -- вязкая проницаемость (Carman-Kozeny)
    Вязкий член растёт с давлением и размером пор; при глубоком вакууме мал
    (модель сводится к чистому Кнудсену). Pmean ~ (Pice+Pc)/2 (берём ~Pc)."""
    eps = porosity_from_conc(cs, rho_solid)
    tau = tortuosity(eps, tau_model)
    vbar = mean_speed(T_C)
    Dk = (2.0 / 3.0) * r_pore * vbar
    B0 = eps * r_pore ** 2 / (8.0 * tau)            # м2
    Deff = eps / tau * Dk + B0 * Pmean_pa / MU_VAPOR
    T = T_C + 273.15
    slope = R_GAS * T / (M_WATER * Deff)
    return Rp0_skin_torr * RP_TO_SI + slope * H


def knudsen_number(r_pore, P_pa, T_C=-25.0, d_mol=2.8e-10):
    """Число Кнудсена Kn = lambda_mfp / r_pore. Kn>>1 -> чистый Кнудсен;
    Kn<~1 -> нужен вязкий член (dusty-gas)."""
    T = T_C + 273.15
    mfp = 1.380649e-23 * T / (np.sqrt(2.0) * np.pi * d_mol ** 2 * P_pa)
    return mfp / r_pore


# ---- F: размер пор из ТЕОРИИ ЗАТВЕРДЕВАНИЯ (вместо таблицы режимов) ---------
def freezing_front(shelf_T_frz_C, Kv, L0=0.01, T_f_C=-2.0):
    """Скорость фронта v и градиент G при заморозке из задачи Стефана.
    Возвращает (v [м/с], G [К/м], t_freeze [с]). Тепло снимается через
    последовательно: Kv (флакон) + теплопроводность намёрзшего льда."""
    dT = T_f_C - shelf_T_frz_C                       # движущая разность, К
    q = dT / (1.0 / Kv + L0 / (2.0 * K_ICE))         # средний поток тепла, Вт/м2
    t_freeze = RHO_ICE * DHFUS_ICE * L0 / q
    v = L0 / t_freeze
    G = q / K_ICE
    return v, G, t_freeze


def pore_radius_solidification(shelf_T_frz_C=-45.0, Kv=None, Pc_torr=0.1,
                               L0=0.01, T_f_C=-2.0,
                               r_ref=15e-6, v_ref=3.0e-6, exponent=0.5):
    """Радиус пор из режима заморозки ЧЕРЕЗ ФИЗИКУ, без таблицы:
      v, G  -- из freezing_front() (задача Стефана при заморозке);
      lambda ~ v^(-exponent)  -- масштаб направленной кристаллизации
               (Kurz-Fisher: быстрее фронт -> мельче структура);
      r_pore = r_ref (v_ref/v)^exponent.
    r_ref, v_ref -- одна литературная привязка (Searles/Nakagawa): при
    v~3e-6 м/с (умеренная заморозка) r_pore~15 мкм. Это НЕ опыт на продукте,
    а реперная точка из обзорной литературы."""
    if Kv is None:
        Kv = Kv_of_Pc(Pc_torr)
    v, G, tf = freezing_front(shelf_T_frz_C, Kv, L0, T_f_C)
    r = r_ref * (v_ref / v) ** exponent
    return dict(r_pore=r, v=v, G=G, t_freeze=tf)


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
# ---- B: эффективная теплопроводность сухого слоя -----------------------------
def k_dry_effective(cs, k_solid=0.20, rho_solid=1580.0):
    """Эффективная теплопроводность сухого пористого коржа (Maxwell-Eucken,
    газ в порах при вакууме ~ не проводит): k_eff = k_solid (1-eps)/(1+eps/2).
    Очень мала (~0.005-0.02 Вт/мК): сухой корж теплоизолирует."""
    eps = porosity_from_conc(cs, rho_solid)
    return k_solid * (1.0 - eps) / (1.0 + 0.5 * eps)


def solve_step(H, Ts_C, Pc_torr, L0, rp_func, Ap, Av,
               T_rad_C=None, emiss=0.9, k_dry=0.012):
    """Квазистационарный энергобаланс на толщине сухого слоя H.
    rp_func(H) -> R_p [СИ]. Возвращает (Tp_C, Tb_C, Js).

    Энергобаланс на фронте (на ед. площади продукта):
       Js*dHs = q_bot + q_top
       q_bot = (Ts - Tp) / ( Ap/(Av*Kv) + (L0-H)/k_ice )      снизу через лёд
       q_top = (T_rad - Tp) / ( 1/h_rad + H/k_dry )           сверху (B): радиация
               на сухую поверхность + теплопроводность вниз через сухой слой
       h_rad = 4*emiss*sigma*Tm^3 (линеаризовано)
    T_rad_C=None -> верхний путь выключен (q_top=0) -- как раньше (совместимость).
    Масса:  Js = (P_ice(Tp) - Pc)/Rp.
    """
    Kv = Kv_of_Pc(Pc_torr)
    Pc = Pc_torr * TORR
    Rp = rp_func(H)
    L_froz = max(L0 - H, 1e-5)
    Rbot = Ap / (Av * Kv) + L_froz / K_ICE

    def q_top_of(Tp):
        if T_rad_C is None:
            return 0.0
        Tm = 0.5 * (T_rad_C + Tp) + 273.15
        h_rad = 4.0 * emiss * SIGMA_SB * Tm ** 3
        Rtop = 1.0 / h_rad + H / max(k_dry, 1e-6)
        return (T_rad_C - Tp) / Rtop

    def residual(Tp):                              # q_supply - q_demand, падает с Tp
        Js = (p_ice(Tp) - Pc) / Rp
        q_demand = Js * DHSUB
        q_supply = (Ts_C - Tp) / Rbot + q_top_of(Tp)
        return q_supply - q_demand

    # нижняя граница -- порог сублимации (p_ice = Pc)
    tlo, thi = -90.0, 40.0
    for _ in range(60):
        tm = 0.5 * (tlo + thi)
        if p_ice(tm) < Pc:
            tlo = tm
        else:
            thi = tm
    Tp_floor = thi + 1e-3
    hi_src = Ts_C if T_rad_C is None else max(Ts_C, T_rad_C)
    lo, hi = Tp_floor, hi_src
    if lo >= hi:
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
    q_bot = (Ts_C - Tp) / Rbot                     # тепло снизу
    Tb = Tp + q_bot * L_froz / K_ICE               # дно из нижнего потока
    return Tp, Tb, Js


def primary_drying(Ts_C, Pc_torr, L0=0.01, Rp0=0.5, A1=20.0, A2=4.0,
                   dT_super=10.0, Ap=3.80e-4, Av=4.71e-4,
                   Tc_C=-32.0, Tg_prime_C=-34.0, dt=60.0, t_max=3e5,
                   rp_func=None, T_rad_C=None, emiss=0.9, k_dry=0.012):
    """Интегрирует первичную сушку до полного ухода льда (H -> L0).

    Ts_C, Pc_torr могут быть числом (постоянные) или callable(t).
    rp_func(H) -- модель сопротивления (СИ). Если None -- эмпирическая
    Rp_areal с масштабом заморозки fcryst (обратная совместимость).
    T_rad_C -- если задано, включается верхний радиационный путь (B); None -- нет.
    Возвращает словарь с траекториями и итоговым временем/риском коллапса.
    """
    fcryst = crystal_factor(dT_super)
    if rp_func is None:
        rp_func = lambda H: Rp_areal(H, Rp0, A1, A2, fcryst)
    H = 0.0
    t = 0.0
    ts, Hs, Tps, Tbs, Js_s, Co_s = [], [], [], [], [], []

    def getval(x, t):
        return x(t) if callable(x) else x

    collapsed = False
    while H < L0 and t < t_max:
        Ts = getval(Ts_C, t)
        Pc = getval(Pc_torr, t)
        Trad = getval(T_rad_C, t) if T_rad_C is not None else None
        Tp, Tb, Js = solve_step(H, Ts, Pc, L0, rp_func, Ap, Av,
                                T_rad_C=Trad, emiss=emiss, k_dry=k_dry)
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
