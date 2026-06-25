"""
Общая физика первичной сушки (используется обоими модулями: теплопроводность
1.4 и квазиравновесие 1.5).

Внутри всё в СИ (К, м, кг, с, Па). На границе пользователь задаёт привычные
единицы (Pch в Torr, L в см, dt в часах, R0/A1/A2 в Torr·см²·ч/г). Температуры на
графиках выводятся в °C (см. main).

Процесс (см. ТЗ 1.1):
  - старт: продукт = полка = Tfreeze, давление камеры Pch установилось;
  - полка греется со скоростью ramp_rate до предела Ts_lim, дальше держится;
  - тепло идёт снизу через дно виалы (Kv) и сквозь столб льда (k_ice) к фронту;
  - на фронте лёд сублимирует, унося тепло dHs·dm/dt; сухой кейк растёт (l);
  - Rp растёт с толщиной кейка; стоп, когда осталось < end_frac воды.
"""
from dataclasses import dataclass, field
import numpy as np

# ----------------------------- константы/конверсии ------------------------
TORR = 133.322                 # Па / Torr
CAL = 4.184                    # Дж / кал
# перевод площадно-нормированного сопротивления Torr·см²·ч/г -> Па·с·м²/кг:
RP_TO_SI = TORR * 1e-4 * 3600.0 / 1e-3      # = 4.7996e4
R_GAS = 8.314                  # Дж/(моль·К)
M_WATER = 0.018015             # кг/моль
RHO_ICE = 917.0                # кг/м³


# --------------------------- свойства веществ -----------------------------
def p_ice(T_K):
    """Давление насыщенного пара надо льдом, Па (Murphy & Koop 2005)."""
    return np.exp(9.550426 - 5723.265 / T_K + 3.53068 * np.log(T_K)
                  - 0.00728332 * T_K)


def delta_Hs(T_K):
    """Теплота сублимации льда, Дж/кг (функция T; слабая зависимость).
    ~2.838e6 при 0 °C, лёгкий рост к низким T (Feistel & Wagner порядок)."""
    return 2.838e6 + 1.5e3 * (273.15 - T_K) / 100.0


def k_ice(T_K):
    """Теплопроводность льда, Вт/(м·К). Функция-обёртка (можно сделать k(T))."""
    return 2.30 * np.ones_like(T_K) if np.ndim(T_K) else 2.30


def cp_ice(T_K):
    """Удельная теплоёмкость льда, Дж/(кг·К). Функция-обёртка."""
    return 2030.0 * np.ones_like(T_K) if np.ndim(T_K) else 2030.0


# ------------------------------ параметры ---------------------------------
@dataclass
class Params:
    """Входные параметры (ТЗ 1.2) + параметры симуляции. Дефолты литературные:
    виала 10cc tubing (Tang & Pikal 2004, Табл. II), ~5% сахароза."""
    # геометрия виалы
    Ap_cm2: float = 3.80         # внутренняя площадь дна, см²
    Av_cm2: float = 4.71         # внешняя площадь дна, см²
    L_cm: float = 1.0            # высота заполнения, см
    # теплопередача виалы: либо прямой Kv (кал/с/см²/К), либо Kc,Kd (формула)
    Kv_direct: float = None      # если задано — используется напрямую
    Kc: float = 2.64e-4          # кал/(с·см²·К), Tang & Pikal Табл. II (10cc)
    Kd: float = 3.64             # 1/Torr
    # сопротивление массопереносу (площадно-нормированное, Torr·см²·ч/г)
    R0: float = 0.5              # интерсепт
    A1: float = 2.5              # параметр Tang & Pikal по l, на 1 см
    A2: float = 0.0              # параметр Tang & Pikal по l, на 1 см
    Rs: float = 0.2              # сопротивление пробки
    # термодинамика/состав
    Tc_C: float = -32.0          # критическая температура продукта, °C
    deltaTc_C: float = 2.0       # безопасный отступ от Tc, °C
    cs: float = 0.05             # массовая доля сухого в растворе
    rho_sol: float = 1000.0      # плотность раствора, кг/м³
    # оборудование
    condenser_kg_h: float = 1.0  # производительность конденсатора, кг/ч (вся партия)
    n_vials: int = 1             # число виал (для проверки перегрузки)
    # условия/симуляция
    Tfreeze_C: float = -40.0     # старт: продукт=полка
    ramp_K_per_min: float = 1.0  # скорость нагрева полки
    dt_h: float = 0.01           # шаг по времени, ч
    end_frac: float = 0.05       # стоп: осталось < end_frac воды (5%)
    n_layers: int = 40           # число слоёв для модуля теплопроводности
    t_max_h: float = 200.0       # предохранитель
    # сетка/вывод
    Ts_min_C: float = -35.0
    Ts_max_C: float = 10.0
    n_Ts: int = 10
    Pch_min_torr: float = 0.01
    Pch_max_torr: float = 0.30
    n_Pc: int = 7
    map_levels: int = 18
    # предельный режим оборудования: rate = intercept + slope * Pch
    equipment_rate_intercept: float = -0.04469035532994933
    equipment_rate_slope: float = 4.0456852791878175
    equipment_limit_points: list = field(default_factory=lambda: [
        (0.02, 0.037),
        (0.03, 0.077),
        (0.04, 0.115),
        (0.058, 0.191),
    ])

    # --- производные (СИ) ---
    @property
    def Ap(self):  return self.Ap_cm2 * 1e-4         # м²
    @property
    def Av(self):  return self.Av_cm2 * 1e-4         # м²
    @property
    def L(self):   return self.L_cm * 1e-2           # м
    @property
    def Tc(self):  return self.Tc_C + 273.15         # К
    @property
    def dt(self):  return self.dt_h * 3600.0         # с
    @property
    def ramp(self): return self.ramp_K_per_min / 60.0  # К/с
    @property
    def lambda_water(self):
        """Масса воды на единицу объёма замороженного коржа, кг/м³."""
        return self.rho_sol * (1.0 - self.cs)


# --------------------------- модельные функции ----------------------------
def Kv_si(Pch_torr, p: Params):
    """Коэф. теплопередачи флакон↔полка, Вт/(м²·К).
    Прямой Kv_direct (кал/с/см²/К) или формула Kc + 3.32e-3 P/(1+Kd P)."""
    if p.Kv_direct is not None:
        Kv_cal = p.Kv_direct
    else:
        Kv_cal = p.Kc + 3.32e-3 * Pch_torr / (1.0 + p.Kd * Pch_torr)
    return Kv_cal * CAL * 1e4                          # -> Вт/(м²·К)


def Rp_areal_si(l_m, p: Params):
    """Площадно-нормированное сопротивление сухого кейка, Па·с·м²/кг.
    Tang & Pikal: (R0 + A1·l[см] / (1 + A2·l[см]))·перевод.
    При A2=0 форма совпадает с прежней линейной моделью."""
    l_cm = l_m * 100.0
    denom = max(1.0 + p.A2 * l_cm, 1e-9)
    return (p.R0 + p.A1 * l_cm / denom) * RP_TO_SI


def Rs_areal_si(p: Params):
    return p.Rs * RP_TO_SI


def shelf_temp(t_s, p: Params, Ts_lim_K):
    """Температура полки в момент t: рампа от Tfreeze до Ts_lim, затем плато."""
    Tfr = p.Tfreeze_C + 273.15
    return min(Tfr + p.ramp * t_s, Ts_lim_K)


def subl_flux_areal(Ti_K, Pch_torr, l_m, p: Params):
    """Площадной поток сублимации Js [кг/(м²·с)] = (Pice(Ti)-Pch)/(Rp+Rs)."""
    Pice = p_ice(Ti_K)
    Pch = Pch_torr * TORR
    R = Rp_areal_si(l_m, p) + Rs_areal_si(p)
    return max((Pice - Pch) / R, 0.0)


def front_floor_K(Pch_torr):
    """Порог сублимации: Ti, при котором Pice = Pch (ниже сушки нет)."""
    Pch = Pch_torr * TORR
    lo, hi = 150.0, 290.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if p_ice(mid) < Pch:
            lo = mid
        else:
            hi = mid
    return hi


def dl_from_subl(Js_areal, dt_s, p: Params):
    """Прирост толщины сухого кейка за шаг из площадного потока (ТЗ: учёт
    концентрации через lambda_water)."""
    return Js_areal * dt_s / p.lambda_water


def water_remaining_frac(l_m, p: Params):
    """Доля оставшейся (несублимированной) воды от начальной = (L-l)/L."""
    return max(p.L - l_m, 0.0) / p.L


def subl_rate_g_per_h_vial(Js_areal, p: Params):
    """Скорость сублимации в г/(ч·виал) для вывода."""
    dm_dt = p.Ap * Js_areal          # кг/с на виалу
    return dm_dt * 1000.0 * 3600.0   # г/ч
