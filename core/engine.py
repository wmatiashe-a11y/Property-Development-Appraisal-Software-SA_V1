from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from .models import Assumptions, Output
from .finance import irr_monthly, annualize_monthly_rate, monthly_rate_from_pa


def _weights(n: int, curve: str) -> np.ndarray:
    if n <= 0:
        return np.array([])
    x = np.linspace(0, 1, n)
    if curve == "front":
        w = 1 - x**1.8
    elif curve == "back":
        w = x**1.8 + 0.01
    else:
        w = np.ones(n)
    w = np.maximum(w, 1e-9)
    return w / w.sum()


def _build_weights(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([])
    x = np.linspace(-2, 2, n)
    w = 1 / (1 + np.exp(-x))  # sigmoid
    w = np.diff(np.concatenate([[0], w]))
    w = np.maximum(w, 1e-9)
    return w / w.sum()


def _product_sellable_sqm(p: Dict[str, Any]) -> float:
    t = (p.get("type") or "").lower()
    if t == "residential":
        units = float(p.get("units") or 0.0)
        size = float(p.get("avg_unit_size_sqm") or 0.0)
        return units * size
    else:
        return float(p.get("sellable_sqm") or 0.0)


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _simulate_finance(
    months: int,
    base_net: np.ndarray,
    land: float,
    equity_injection_m0: float,
    interest_rate_pa: float,
    arrangement_fee: float,
    exit_fee: float,
) -> Dict[str, Any]:
    """
    Full monthly finance simulation:
    - base_net = revenue - costs (ex land, ex finance), by month
    - land paid month 0
    - equity injection is project inflow at month 0 (but equity investor outflow)
    - debt draws to cover cash deficits, repaid from surpluses
    - interest accrues monthly on end-of-month debt
    - arrangement fee applied in month 0; exit fee applied in last month
    Returns: schedules + totals, including equity distribution at end.
    """
    r_m = monthly_rate_from_pa(interest_rate_pa)

    cash = 0.0
    debt = 0.0
    peak_debt = 0.0

    debt_draw = np.zeros(months)
    debt_repay = np.zeros(months)
    interest = np.zeros(months)
    cash_end = np.zeros(months)

    # Project cashflow including land and fees (fees are costs)
    for m in range(months):
        net = float(base_net[m])

        if m == 0:
            net -= land
            net -= arrangement_fee
            net += equity_injection_m0  # project receives equity injection

        if m == months - 1:
            net -= exit_fee

        cash += net

        # draw debt if cash negative
        if cash < 0:
            draw = -cash
            debt += draw
            debt_draw[m] = draw
            cash = 0.0

        # repay if surplus cash and debt outstanding
        if cash > 0 and debt > 0:
            repay = min(cash, debt)
            debt -= repay
            debt_repay[m] = repay
            cash -= repay

        # interest on end-of-month debt (capitalised)
        i = debt * r_m
        debt += i
        interest[m] = i

        peak_debt = max(peak_debt, debt)
        cash_end[m] = cash

    # Force close any remaining debt at end via equity top-up (if needed)
    equity_topup_end = 0.0
    if debt > 1e-6:
        # Use remaining cash first
        if cash > 0:
            repay = min(cash, debt)
            debt -= repay
            cash -= repay
        if debt > 1e-6:
            equity_topup_end = debt
            debt = 0.0

    equity_distribution_end = cash  # what remains after all repayments
    finance_costs = float(interest.sum() + arrangement_fee + exit_fee)

    return {
        "peak_debt": float(peak_debt),
        "finance_costs": float(finance_costs),
        "debt_draw": debt_draw,
        "debt_repay": debt_repay,
        "interest": interest,
        "cash_end": cash_end,
        "equity_topup_end": float(equity_topup_end),
        "equity_distribution_end": float(equity_distribution_end),
    }


def run_appraisal(a: Assumptions, max_iter: int = 30) -> Output:
    audit: List[Dict[str, Any]] = []
    currency = a.currency or "ZAR"

    build_months = int(max(1, a.build_months))
    sales_months = int(max(1, a.sales_months))
    months = int(build_months + sales_months)

    bw = _build_weights(build_months)
    sw = _weights(sales_months, a.sales_curve)

    # --- Product table + line GDVs / costs ---
    product_rows: List[Dict[str, Any]] = []
    gdv_total = 0.0
    build_base_total = 0.0

    # Inclusionary applies only to eligible products (typically residential)
    inc_rate = float(np.clip(a.inclusionary_rate, 0.0, 1.0))
    inc_price = float(a.inclusionary_price_per_sqm)

    for p in (a.products or []):
        name = str(p.get("name") or "Product")
        ptype = str(p.get("type") or "commercial").lower()
        sellable_sqm = _product_sellable_sqm(p)

        price_psm = _safe_float(p.get("price_per_sqm"))
        cost_psm = _safe_float(p.get("build_cost_per_sqm"))

        eligible = bool(p.get("inclusionary_eligible", ptype == "residential"))
        if eligible and inc_rate > 0:
            aff_sqm = sellable_sqm * inc_rate
            mkt_sqm = sellable_sqm - aff_sqm
            gdv_line = mkt_sqm * price_psm + aff_sqm * inc_price
        else:
            aff_sqm = 0.0
            mkt_sqm = sellable_sqm
            gdv_line = sellable_sqm * price_psm

        build_line = sellable_sqm * cost_psm

        gdv_total += gdv_line
        build_base_total += build_line

        product_rows.append({
            "Product": name,
            "Type": ptype,
            "Sellable sqm": float(sellable_sqm),
            "Price / sqm": float(price_psm),
            "GDV": float(gdv_line),
            "Build cost / sqm": float(cost_psm),
            "Build base": float(build_line),
            "Inclusionary sqm": float(aff_sqm),
        })

    # --- Build uplifts / escalation / contingency ---
    heritage_uplift = build_base_total * float(a.heritage_cost_uplift_rate)
    build_base2 = build_base_total + heritage_uplift

    build_years = build_months / 12.0
    escalation = build_base2 * float(a.escalation_rate_pa) * (build_years / 2.0)  # half-period average
    contingency = (build_base2 + escalation) * float(a.contingency_rate)
    build_total = build_base2 + escalation + contingency

    prof_fees = build_total * float(a.professional_fees_rate)
    marketing = gdv_total * float(a.marketing_rate)
    overhead = float(a.overhead_per_month) * months
    statutory = float(a.statutory_costs)
    incentive = float(a.incentive_grant)

    tdc_ex_land_ex_finance = build_total + prof_fees + statutory + marketing + overhead - incentive

    audit.extend([
        {"section": "Revenue", "key": "GDV total", "value": gdv_total, "unit": currency},
        {"section": "Costs", "key": "Build base", "value": build_base_total, "unit": currency},
        {"section": "Costs", "key": "Heritage uplift", "value": heritage_uplift, "unit": currency},
        {"section": "Costs", "key": "Escalation", "value": escalation, "unit": currency},
        {"section": "Costs", "key": "Contingency", "value": contingency, "unit": currency},
        {"section": "Costs", "key": "Build total", "value": build_total, "unit": currency},
        {"section": "Costs", "key": "Professional fees", "value": prof_fees, "unit": currency},
        {"section": "Costs", "key": "Statutory costs", "value": statutory, "unit": currency},
        {"section": "Costs", "key": "Marketing", "value": marketing, "unit": currency},
        {"section": "Costs", "key": "Overhead", "value": overhead, "unit": currency},
        {"section": "Costs", "key": "Incentive grant (reduces cost)", "value": incentive, "unit": currency},
    ])

    # --- Monthly base cashflows (revenue & cost ex land & ex finance) ---
    monthly_costs = np.zeros(months)
    monthly_revenue = np.zeros(months)

    # Spread build_total + prof_fees + statutory across build months
    monthly_costs[:build_months] += build_total * bw
    monthly_costs[:build_months] += prof_fees / build_months
    monthly_costs[:build_months] += statutory / build_months

    # Marketing over sales months
    monthly_costs[build_months:] += marketing * sw

    # Overhead evenly
    monthly_costs += overhead / months

    # Incentive as month 0 cost reduction
    monthly_costs[0] -= incentive

    # Revenue over sales months
    monthly_revenue[build_months:] = gdv_total * sw

    base_net = monthly_revenue - monthly_costs

    # --- Residual land + finance iteration ---
    land = float(a.land_price_input) if a.land_price_input is not None else 0.0

    # fees depend on peak debt, which depends on land; iterate with a couple passes
    arrangement_fee = 0.0
    exit_fee = 0.0

    for _ in range(max_iter):
        if a.use_debt:
            sim0 = _simulate_finance(
                months=months,
                base_net=base_net,
                land=land,
                equity_injection_m0=float(a.equity_injection_month0),
                interest_rate_pa=float(a.debt_interest_rate_pa),
                arrangement_fee=arrangement_fee,
                exit_fee=exit_fee,
            )
            peak = sim0["peak_debt"]
            new_arr = peak * float(a.debt_arrangement_fee_rate)
            new_exit = peak * float(a.debt_exit_fee_rate)

            # damp changes
            arrangement_fee = 0.5 * arrangement_fee + 0.5 * new_arr
            exit_fee = 0.5 * exit_fee + 0.5 * new_exit

            sim = _simulate_finance(
                months=months,
                base_net=base_net,
                land=land,
                equity_injection_m0=float(a.equity_injection_month0),
                interest_rate_pa=float(a.debt_interest_rate_pa),
                arrangement_fee=arrangement_fee,
                exit_fee=exit_fee,
            )
            finance_costs = sim["finance_costs"]
            peak_debt = sim["peak_debt"]
        else:
            arrangement_fee = 0.0
            exit_fee = 0.0
            sim = _simulate_finance(
                months=months,
                base_net=base_net,
                land=land,
                equity_injection_m0=float(a.equity_injection_month0),
                interest_rate_pa=0.0,
                arrangement_fee=0.0,
                exit_fee=0.0,
            )
            finance_costs = 0.0
            peak_debt = 0.0

        # If user fixed land, stop
        if a.land_price_input is not None:
            break

        # Solve land to hit target profit (fixed-point; finance depends on land)
        if a.target_profit_basis == "gdv":
            desired_profit = gdv_total * float(a.target_profit_rate)
            land_new = gdv_total - (tdc_ex_land_ex_finance + finance_costs) - desired_profit
        else:
            # gdv = (cost_ex_land + finance + land) * (1 + p)
            c = tdc_ex_land_ex_finance + finance_costs
            land_new = (gdv_total / (1.0 + float(a.target_profit_rate))) - c

        land_new = float(land_new)
        if abs(land_new - land) < 10.0:
            land = land_new
            break
        # damp to avoid oscillation
        land = 0.6 * land + 0.4 * land_new

    # final finance sim for schedules
    if a.use_debt:
        sim = _simulate_finance(
            months=months,
            base_net=base_net,
            land=land,
            equity_injection_m0=float(a.equity_injection_month0),
            interest_rate_pa=float(a.debt_interest_rate_pa),
            arrangement_fee=arrangement_fee,
            exit_fee=exit_fee,
        )
        finance_costs = sim["finance_costs"]
        peak_debt = sim["peak_debt"]
    else:
        sim = _simulate_finance(
            months=months,
            base_net=base_net,
            land=land,
            equity_injection_m0=float(a.equity_injection_month0),
            interest_rate_pa=0.0,
            arrangement_fee=0.0,
            exit_fee=0.0,
        )
        finance_costs = 0.0
        peak_debt = 0.0

    total_cost_incl = tdc_ex_land_ex_finance + finance_costs + land
    profit = gdv_total - total_cost_incl
    profit_rate_gdv = profit / gdv_total if gdv_total else 0.0
    profit_rate_cost = profit / total_cost_incl if total_cost_incl else 0.0

    # Equity IRR (simple but real): equity outflows are injections (m0 + forced topup end),
    # inflow is end distribution.
    equity_cf = [0.0] * months
    equity_cf[0] = -float(a.equity_injection_month0)
    if sim["equity_topup_end"] > 0:
        equity_cf[-1] -= float(sim["equity_topup_end"])
    equity_cf[-1] += float(sim["equity_distribution_end"])

    irr_pa = annualize_monthly_rate(irr_monthly(equity_cf))

    # Build cashflow rows for UI
    rows: List[Dict[str, Any]] = []
    debt_draw = sim["debt_draw"]
    debt_repay = sim["debt_repay"]
    interest = sim["interest"]

    for m in range(months):
        land_row = land if m == 0 else 0.0
        arr_row = arrangement_fee if m == 0 else 0.0
        exit_row = exit_fee if m == months - 1 else 0.0
        equity_inj = float(a.equity_injection_month0) if m == 0 else 0.0

        rows.append({
            "Month": m + 1,
            "Revenue": float(monthly_revenue[m]),
            "Costs (ex land, ex fin)": float(monthly_costs[m]),
            "Land": float(land_row),
            "Arr fee": float(arr_row),
            "Exit fee": float(exit_row),
            "Debt draw": float(debt_draw[m]),
            "Debt repay": float(debt_repay[m]),
            "Interest (cap.)": float(interest[m]),
            "Equity inject": float(equity_inj),
            "Cash end": float(sim["cash_end"][m]),
        })

    audit.extend([
        {"section": "Profit", "key": "Target basis", "value": a.target_profit_basis, "unit": ""},
        {"section": "Profit", "key": "Target profit rate", "value": float(a.target_profit_rate), "unit": "ratio"},
        {"section": "Land", "key": "Land value (residual or input)", "value": float(land), "unit": currency},
        {"section": "Finance", "key": "Debt interest p.a.", "value": float(a.debt_interest_rate_pa), "unit": "ratio"},
        {"section": "Finance", "key": "Arrangement fee", "value": float(arrangement_fee), "unit": currency},
        {"section": "Finance", "key": "Exit fee", "value": float(exit_fee), "unit": currency},
        {"section": "Finance", "key": "Finance costs (total)", "value": float(finance_costs), "unit": currency},
        {"section": "Finance", "key": "Peak debt", "value": float(peak_debt), "unit": currency},
        {"section": "Profit", "key": "Total cost (incl land+finance)", "value": float(total_cost_incl), "unit": currency},
        {"section": "Profit", "key": "Profit", "value": float(profit), "unit": currency},
        {"section": "Profit", "key": "Profit % GDV", "value": float(profit_rate_gdv), "unit": "ratio"},
        {"section": "Profit", "key": "Profit % Cost", "value": float(profit_rate_cost), "unit": "ratio"},
    ])

    return Output(
        currency=currency,
        gdv=float(gdv_total),
        tdc_ex_land_ex_finance=float(tdc_ex_land_ex_finance),
        finance_costs=float(finance_costs),
        land_value=float(land),
        total_cost_incl_land_finance=float(total_cost_incl),
        profit=float(profit),
        profit_rate_on_gdv=float(profit_rate_gdv),
        profit_rate_on_cost=float(profit_rate_cost),
        peak_debt=float(peak_debt),
        equity_irr_pa=None if irr_pa is None else float(irr_pa),
        months=int(months),
        cashflow_rows=rows,
        audit=audit,
        product_rows=product_rows,
    )


def sensitivity_grid(
    a: Assumptions,
    price_steps=(-0.1, -0.05, 0, 0.05, 0.1),
    cost_steps=(-0.1, -0.05, 0, 0.05, 0.1),
) -> Tuple[List[str], List[str], np.ndarray]:
    rows = [f"{int(p*100)}%" for p in price_steps]
    cols = [f"{int(c*100)}%" for c in cost_steps]
    mat = np.zeros((len(rows), len(cols)))

    for i, dp in enumerate(price_steps):
        for j, dc in enumerate(cost_steps):
            aa = Assumptions.from_dict(a.to_dict())
            # uplift ALL product prices/costs for sensitivity
            prods = []
            for p in aa.products:
                pp = dict(p)
                pp["price_per_sqm"] = _safe_float(pp.get("price_per_sqm")) * (1 + dp)
                pp["build_cost_per_sqm"] = _safe_float(pp.get("build_cost_per_sqm")) * (1 + dc)
                prods.append(pp)
            aa.products = prods
            out = run_appraisal(aa)
            mat[i, j] = out.land_value
    return rows, cols, mat
