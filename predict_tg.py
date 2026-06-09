"""
Безэкспериментальная оценка Tg' и Tc формуляции.

Три уровня:
  1) табличные Tg' чистых эксципиентов;
  2) Gordon-Taylor: Tg(состав) для бинара «растворённое+вода» + пересечение
     с кривой плавления льда (диаграмма состояния) -> предсказание (Cg', Tg');
  3) смешение по Фоксу: Tg' многокомпонентной формуляции из табличных Tg'.
Tc = Tg' + offset (аморфные) или Te (кристаллические).

Точность ~ ±2..5 °C. Источники чисел: Roos 1993; Slade & Levine; обзоры по
лиофилизации. Соли/буферы и ионная сила сильно понижают Tg' -- учитывать отдельно.
"""
import numpy as np

K0 = 273.15
TG_WATER = 136.0          # Tg воды, K (-137 °C)

# ---- Уровень 1: табличные свойства чистых эксципиентов ----
# Tg_dry -- стеклование сухого вещества (°C); Tg_prime -- Tg' (°C);
# w_unfrozen -- масс. доля незамёрзшей воды в максимально-конц. фазе (Cg');
# rho -- плотность (для оценки k по Simha-Boyer, г/см3); type.
EXCIPIENTS = {
    # name        Tg_dry  Tg_prime  w_unfrozen  rho   type
    "sucrose":   dict(Tg_dry=67,  Tg_prime=-32.0, w_unf=0.20, rho=1.59, kind="amorphous"),
    "trehalose": dict(Tg_dry=115, Tg_prime=-29.5, w_unf=0.20, rho=1.58, kind="amorphous"),
    "lactose":   dict(Tg_dry=101, Tg_prime=-28.0, w_unf=0.21, rho=1.55, kind="amorphous"),
    "maltose":   dict(Tg_dry=87,  Tg_prime=-30.0, w_unf=0.20, rho=1.54, kind="amorphous"),
    "sorbitol":  dict(Tg_dry=-2,  Tg_prime=-43.5, w_unf=0.19, rho=1.49, kind="amorphous"),
    "glycerol":  dict(Tg_dry=-93, Tg_prime=-65.0, w_unf=0.18, rho=1.26, kind="amorphous"),
    "dextran":   dict(Tg_dry=220, Tg_prime=-10.0, w_unf=0.30, rho=1.50, kind="amorphous"),
    "PVP_K30":   dict(Tg_dry=164, Tg_prime=-20.0, w_unf=0.28, rho=1.20, kind="amorphous"),
    "HES":       dict(Tg_dry=180, Tg_prime=-10.0, w_unf=0.30, rho=1.50, kind="amorphous"),
    # кристаллизующиеся: ограничение -- эвтектика Te, а не Tg'
    "mannitol":  dict(Tg_dry=11,  Te=-1.5, kind="crystalline"),
    "glycine":   dict(Tg_dry=None, Te=-3.0, kind="crystalline"),
    "NaCl":      dict(Tg_dry=None, Te=-21.1, kind="crystalline"),
}


def gordon_taylor(w_solute, Tg_solute_C, Tg_water_C=TG_WATER - K0, k=None,
                  rho_solute=1.5):
    """Tg(состав) бинара растворённое+вода (Gordon-Taylor), °C."""
    Tg1, Tg2 = Tg_solute_C + K0, Tg_water_C + K0
    if k is None:                                  # Simha-Boyer
        k = rho_solute * Tg1 / (1.0 * Tg2)
    w1, w2 = w_solute, 1.0 - w_solute
    Tg = (w1 * Tg1 + k * w2 * Tg2) / (w1 + k * w2)
    return Tg - K0


def liquidus_ice_C(w_solute, Mw_solute, Kf=1.86):
    """Грубая кривая плавления льда (понижение т. замерзания, коллигативно).
    Для разбавленной области; у Cg' сильно неидеально -> только как ориентир."""
    # моляльность ~ (w/(1-w))/Mw*1000
    m = (w_solute / max(1 - w_solute, 1e-6)) / Mw_solute * 1000.0
    return -Kf * m


def estimate_Tg_prime_binary(name, k=None):
    """Уровень 2: Tg' из пересечения Tg(w) и кривой плавления (диаграмма).
    Если в таблице есть Cg' (w_unf) -- используем его (надёжнее)."""
    e = EXCIPIENTS[name]
    w_solute = 1.0 - e["w_unf"]                    # доля сухого в Cg'
    return gordon_taylor(w_solute, e["Tg_dry"], k=k, rho_solute=e["rho"])


def fox_mix_Tg_prime(components):
    """Уровень 3: Tg' многокомпонентной аморфной формуляции (Фокс).
    components: dict {name: dry_mass_fraction}. Возвращает Tg' (°C)."""
    # нормировка ТОЛЬКО по аморфным компонентам (кристаллические Tg' не дают)
    amorph = {n: w for n, w in components.items()
              if EXCIPIENTS[n]["kind"] == "amorphous"}
    tot = sum(amorph.values())
    if tot <= 0:
        return None
    inv = sum((w / tot) / (EXCIPIENTS[n]["Tg_prime"] + K0)
              for n, w in amorph.items())
    return 1.0 / inv - K0


def predict_Tc(components, offset=2.0):
    """Tc формуляции. Если есть кристаллический компонент в избытке -- Te;
    иначе Tg' + offset."""
    cryst = [(n, w) for n, w in components.items()
             if EXCIPIENTS[n]["kind"] == "crystalline"]
    amorph_w = sum(w for n, w in components.items()
                   if EXCIPIENTS[n]["kind"] == "amorphous")
    Tg_p = fox_mix_Tg_prime(components) if amorph_w > 0 else None
    if cryst and amorph_w < sum(w for _, w in cryst):
        Te = min(EXCIPIENTS[n]["Te"] for n, _ in cryst)
        return dict(Tc=Te, basis="eutectic Te", Tg_prime=Tg_p)
    return dict(Tc=Tg_p + offset, basis="Tg'+offset", Tg_prime=Tg_p)


if __name__ == "__main__":
    print("== Уровень 1: табличные Tg' ==")
    for n in ["sucrose", "trehalose", "sorbitol", "dextran"]:
        print(f"  {n:10s} Tg'={EXCIPIENTS[n]['Tg_prime']:+.1f} °C")

    print("\n== Уровень 2: Gordon-Taylor (бинар + Cg') vs таблица ==")
    for n in ["sucrose", "trehalose", "sorbitol"]:
        gt = estimate_Tg_prime_binary(n)
        tab = EXCIPIENTS[n]["Tg_prime"]
        print(f"  {n:10s} GT={gt:+6.1f}  таблица={tab:+6.1f}  Δ={gt-tab:+.1f} °C")

    print("\n== Уровень 3: смешение по Фоксу (формуляция) ==")
    cases = [
        ("сахароза:трегалоза 1:1", {"sucrose": 0.5, "trehalose": 0.5}),
        ("сахароза 90% + сорбитол 10%", {"sucrose": 0.9, "sorbitol": 0.1}),
        ("трегалоза 80% + декстран 20%", {"trehalose": 0.8, "dextran": 0.2}),
    ]
    for label, comp in cases:
        tgp = fox_mix_Tg_prime(comp)
        tc = predict_Tc(comp)
        print(f"  {label:32s} Tg'={tgp:+6.1f}  Tc={tc['Tc']:+6.1f} ({tc['basis']})")

    print("\n== Кристаллический bulking agent (маннитол + сахароза) ==")
    comp = {"mannitol": 0.8, "sucrose": 0.2}
    tc = predict_Tc(comp)
    print(f"  маннитол 80% + сахароза 20%: Tc={tc['Tc']:+.1f} ({tc['basis']}), "
          f"Tg'(аморф.фазы)={tc['Tg_prime']:+.1f}")
