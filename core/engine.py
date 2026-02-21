from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from .models import Assumptions, Output
from .finance import irr_monthly, annualize_monthly_rate
from .taxes_sa import VATConfig, split_vat_from_gross, add_vat_to_net, transfer_duty_za


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
    w = 1 / (1 + np.exp(-x))
    w = np.diff(np.concatenate([[0], w]))
    w = np.maximum(w, 1e-9)
    return w / w.sum()


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _phase_map(phases: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(p.get("name")): p for p in (phases or [])}


def _compute_presales_gate_month(a: Assumptions, product_rows: List[Dict[str, Any]]) -> int:
    """
    Determine first month when cumulative residential presales value >= required % of total residential GDV.
    """
    required_pct = float(a.presales_required_pct_resi)
    if required_pct <= 0:
        return 0

    # Residential GDV total + schedule of achieved
    resi_rows = [r for r in product_rows if r.get("is_residential")]
    total_resi_gdv = sum(float(r["gdv_gross"]) for r in resi_rows)
    if total_resi_gdv <= 0:
        return 0

    target_value = required_pct * total_resi_gdv

    # month -> value achieved
    month_vals: Dict[int, float] = {}
    for r in resi_rows:
        m = int(r.get("presales_achieved_month", 0))
        v = float(r.get("presales_value_gross", 0.0))
        month_vals[m] = month_vals.get(m, 0.0) + v

    cum = 0.0
    for m in sorted(month_vals.keys()):
        cum += month_vals[m]
        if cum >= target_value:
            return int(m)
    return 10_000  # effectively never


def run_appraisal(a: Assumptions, max_iter: int = 30) -> Output:
    ccy = a.currency or "ZAR"
    audit: List[Dict[str, Any]] = []

    phase_by_name = _phase_map(a.phases)
    if not phase_by_name:
        # fallback
        phase_by_name = {"Phase 1": {"name": "Phase 1", "start_month": 0, "build_months": 18, "sales_months": 12, "sales_curve": "linear"}}

    # VAT config
    vat_cfg = VATConfig(**(a.vat or {}))
    vr = float(vat_cfg.vat_rate)

    # Interest rate = prime + margin
    interest_pa = float(a.prime_rate_pa + a.debt_margin_over_prime)

    # --- Build product rows, compute Net area, GBA, GDV gross, build net/gross ---
    product_rows: List[Dict[str, Any]] = []
    gdv_gross_total = 0.0
    gdv_net_total = 0.0
    build_net_total = 0.0

    for p in (a.products or []):
        name = str(p.get("name") or "Product")
        ptype = str(p.get("type") or "residential_sale")
        phase = str(p.get("phase") or "Phase 1")
        ph = phase_by_name.get(phase) or list(phase_by_name.values())[0]

        is_resi = ptype.startswith("residential")
        eff = float(np.clip(_safe_float(p.get("efficiency_ratio", 0.83)), 0.50, 0.98))

        if is_resi:
            units = int(_safe_float(p.get("units", 0)))
            avg_net = _safe_float(p.get("avg_unit_net_sqm", 0.0))
            net_sqm = units * avg_net
        else:
            net_sqm = _safe_float(p.get("net_sqm", 0.0))

        gba_sqm = net_sqm / eff if eff > 0 else 0.0

        sale_price_net = _safe_float(p.get("sale_price_per_net_sqm", 0.0))  # as entered (typically VAT-inclusive)
        build_cost_gba = _safe_float(p.get("build_cost_per_gba_sqm", 0.0))   # as entered (typically excl VAT)

        gdv_gross = net_sqm * sale_price_net

        # Economic net GDV: if VAT-enabled and prices include VAT and outputs are standard-rated,
        # then net revenue excludes VAT portion (VAT payable to SARS).
        if vat_cfg.enabled and vat_cfg.prices_include_vat:
            gdv_net, out_vat = split_vat_from_gross(gdv_gross, vr)
        elif vat_cfg.enabled and (not vat_cfg.prices_include_vat):
            # prices are net; add VAT for cash gross; econ net is net
            gdv_net = gdv_gross
            out_vat = gdv_gross * vr
            gdv_gross = gdv_gross * (1 + vr)
        else:
            gdv_net = gdv_gross
            out_vat = 0.0

        # Build cost net (economic): assume builder quote is net if costs_include_vat False
        build_net = gba_sqm * build_cost_gba
        if vat_cfg.enabled and vat_cfg.costs_include_vat:
            build_net, in_vat = split_vat_from_gross(build_net, vr)
        else:
            in_vat = build_net * vr if vat_cfg.enabled else 0.0

        # Presales values (gross) used for gate
        presales_pct = float(np.clip(_safe_float(p.get("presales_pct", 0.0)), 0.0, 1.0))
        presales_value_gross = gdv_gross * presales_pct
        presales_month = int(_safe_float(p.get("presales_achieved_month", 0)))

        product_rows.append({
            "name": name,
            "type": ptype,
            "phase": phase,
            "phase_start": int(_safe_float(ph.get("start_month", 0))),
            "build_months": int(_safe_float(ph.get("build_months", 18))),
            "sales_months": int(_safe_float(ph.get("sales_months", 12))),
            "sales_curve": str(ph.get("sales_curve", "linear")),

            "is_residential": is_resi,
            "net_sqm": float(net_sqm),
            "efficiency": float(eff),
            "gba_sqm": float(gba_sqm),

            "sale_price_per_net_sqm": float(sale_price_net),
            "gdv_gross": float(gdv_gross),
            "gdv_net": float(gdv_net),
            "output_vat_total": float(out_vat),

            "build_cost_per_gba_sqm": float(build_cost_gba),
            "build_net": float(build_net),
            "input_vat_build_total": float(in_vat),

            "offplan_share": float(np.clip(_safe_float(p.get("offplan_share", 0.0)), 0.0, 1.0)),
            "deposit_pct": float(np.clip(_safe_float(p.get("deposit_pct", 0.0)), 0.0, 1.0)),
            "deposit_released_during_build": bool(p.get("deposit_released_during_build", False)),

            "presales_pct": presales_pct,
            "presales_value_gross": float(presales_value_gross),
            "presales_achieved_month": presales_month,
        })

        gdv_gross_total += gdv_gross
        gdv_net_total += gdv_net
        build_net_total += build_net

    # --- Determine total timeline length ---
    end_month = 0
    for r in product_rows:
        start = int(r["phase_start"])
        b = int(r["build_months"])
        s = int(r["sales_months"])
        end_month = max(end_month, start + b + s)
    months = int(max(1, end_month))

    # --- Build monthly revenue (gross cash), monthly costs (net), and VAT ledgers ---
    revenue_gross = np.zeros(months)
    costs_net = np.zeros(months)

    # Spread build costs by phase build months
    # Apply heritage uplift + escalation + contingency globally (MVP) on total build net
    heritage_uplift = build_net_total * float(a.heritage_cost_uplift_rate)
    build_net2 = build_net_total + heritage_uplift
    build_years = (sum(int(r["build_months"]) for r in product_rows) / max(1, len(product_rows))) / 12.0
    escalation = build_net2 * float(a.escalation_rate_pa) * (build_years / 2.0)
    contingency = (build_net2 + escalation) * float(a.contingency_rate)
    build_total_net = build_net2 + escalation + contingency

    # Allocate build_total_net back to months by each product’s phase build weights proportionally
    build_net_by_product = {r["name"]: float(r["build_net"]) for r in product_rows}
    build_net_sum = max(1e-9, sum(build_net_by_product.values()))
    for r in product_rows:
        start = int(r["phase_start"])
        b = int(r["build_months"])
        w = _build_weights(b)
        share = build_net_by_product[r["name"]] / build_net_sum
        costs_net[start:start+b] += (build_total_net * share) * w

    # Prof fees, statutory during build (spread across months where any build occurs)
    prof_fees_net = build_total_net * float(a.professional_fees_rate)
    statutory_net = float(a.statutory_costs)

    # Determine build-active months (union)
    build_active = np.zeros(months)
    for r in product_rows:
        start = int(r["phase_start"])
        b = int(r["build_months"])
        build_active[start:start+b] = 1.0
    build_month_indices = np.where(build_active > 0)[0]
    if len(build_month_indices) == 0:
        build_month_indices = np.array([0])

    costs_net[build_month_indices] += prof_fees_net / len(build_month_indices)
    costs_net[build_month_indices] += statutory_net / len(build_month_indices)

    # Marketing during sales months (per phase)
    marketing_net = gdv_net_total * float(a.marketing_rate)
    # allocate marketing to each phase sales window by product GDV share
    gdv_net_sum = max(1e-9, sum(float(r["gdv_net"]) for r in product_rows))
    for r in product_rows:
        start = int(r["phase_start"])
        b = int(r["build_months"])
        s = int(r["sales_months"])
        w = _weights(s, r["sales_curve"])
        share = float(r["gdv_net"]) / gdv_net_sum
        costs_net[start+b:start+b+s] += (marketing_net * share) * w

    # Overhead across all months
    overhead_net = float(a.overhead_per_month) * months
    costs_net += overhead_net / months

    # --- Revenue streams with off-plan vs completion ---
    # Cash collection rules:
    # - Deposits: if released during build => cash inflow in build months; else at transfer (completion stream)
    # - Offplan_share influences CONTRACTS + presales gate; cash depends on deposit rule.
    for r in product_rows:
        start = int(r["phase_start"])
        b = int(r["build_months"])
        s = int(r["sales_months"])
        w_build = _weights(b, "back")  # deposits tend to ramp later as marketing ramps
        w_sales = _weights(s, r["sales_curve"])

        gdv_gross = float(r["gdv_gross"])
        offplan = float(r["offplan_share"])
        deposit_pct = float(r["deposit_pct"])
        deposit_cash = gdv_gross * offplan * deposit_pct

        completion_cash = gdv_gross - deposit_cash  # remaining collected on transfer (simplified)

        if bool(r["deposit_released_during_build"]) and deposit_cash > 0:
            revenue_gross[start:start+b] += deposit_cash * w_build
        else:
            completion_cash += deposit_cash  # all cash at transfer

        revenue_gross[start+b:start+b+s] += completion_cash * w_sales

    # --- Land acquisition + friction (month 0 at earliest start) ---
    earliest_start = min(int(r["phase_start"]) for r in product_rows) if product_rows else 0
    land_month = max(0, earliest_start)

    # Land economic net + VAT cash
    land_gross_entered = float(a.land_price)
    land_treatment = str(a.land_treatment)

    transfer_duty = 0.0
    land_vat_out = 0.0
    land_net = land_gross_entered

    if a.solve_residual_land:
        # in residual mode, land is solved later; for now set to 0 and solve in iterations
        land_net = 0.0
        land_gross_entered = 0.0
        land_treatment = "transfer_duty"

    if vat_cfg.enabled:
        if land_treatment == "vat_standard":
            # assume land price entered is VAT-inclusive if prices_include_vat flag, else net
            if vat_cfg.prices_include_vat:
                land_net, land_vat_out = split_vat_from_gross(land_gross_entered, vr)
            else:
                land_net = land_gross_entered
                land_gross_entered = land_gross_entered * (1 + vr)
                land_vat_out = land_gross_entered - land_net
        elif land_treatment == "vat_zero":
            land_net = land_gross_entered
            land_vat_out = 0.0
        else:
            # transfer duty applies when not VAT vendor transaction (simplified)
            transfer_duty = transfer_duty_za(land_gross_entered)
    else:
        # VAT disabled => transfer duty treatment (typical)
        transfer_duty = transfer_duty_za(land_gross_entered)

    legal_net = float(a.legal_conveyancing)
    other_net = float(a.land_other_disbursements)
    friction_net = transfer_duty + legal_net + other_net

    # Land + friction are costs (net economics), paid as cash outflows
    # For cash outflow, land_gross_entered includes any VAT if vatable
    costs_net[land_month] += 0.0  # keep net costs separate; land handled explicitly

    # --- VAT ledger per month (cash settlement lag) ---
    vat_out = np.zeros(months)
    vat_in = np.zeros(months)

    if vat_cfg.enabled:
        # Output VAT from revenue gross
        if vat_cfg.prices_include_vat:
            _, vout = split_vat_from_gross(revenue_gross.sum(), vr)  # total; but we need monthly
            # compute monthly split
            for m in range(months):
                _, v = split_vat_from_gross(float(revenue_gross[m]), vr)
                vat_out[m] += v
        else:
            # if prices are net, revenue_gross already includes VAT in our setup (we grossed it up earlier for those cases)
            for m in range(months):
                # treat the VAT portion as gross - net
                net, v = split_vat_from_gross(float(revenue_gross[m]), vr)
                vat_out[m] += v

        # Input VAT from costs:
        # costs_net is net-of-vat economics; cash may include vat depending on costs_include_vat.
        for m in range(months):
            net_cost = float(costs_net[m])
            if net_cost <= 0:
                continue
            # if costs entered include VAT, net already stripped earlier at build-line level; here we approximate:
            # treat net_cost as VAT-exclusive and add VAT for cash.
            _, vin = add_vat_to_net(net_cost, vr)
            vat_in[m] += vin

        # Land VAT (if vatable) is input VAT (recoverable) for developer vendor
        if land_vat_out > 0:
            vat_in[land_month] += land_vat_out

    # VAT settlement with lag
    vat_settle = np.zeros(months)
    if vat_cfg.enabled:
        lag = max(0, int(vat_cfg.settlement_lag_months))
        for m in range(months):
            net_payable = vat_out[m] - (vat_in[m] if vat_cfg.input_vat_recoverable else 0.0)
            pay_month = m + lag
            if pay_month < months:
                # positive => cash outflow later; negative => refund later
                vat_settle[pay_month] += net_payable

    vat_net_payable_total = float(vat_settle.sum())

    # --- Build cashflow arrays (cash basis) ---
    # cash costs: if VAT enabled, cash outflow includes VAT on vatable costs.
    costs_cash = np.zeros(months)
    for m in range(months):
        net_cost = float(costs_net[m])
        if vat_cfg.enabled:
            gross, _ = add_vat_to_net(net_cost, vr)
            costs_cash[m] = gross
        else:
            costs_cash[m] = net_cost

    # Land cash paid (gross) + friction cash (transfer duty is cash) at land_month
    costs_cash[land_month] += land_gross_entered + friction_net

    # VAT settlement cash (positive payable = cost; negative refund = inflow)
    # We'll handle as an extra line in cashflow.
    vat_cash = vat_settle.copy()

    # --- Presales gate month ---
    presales_gate_month = _compute_presales_gate_month(a, product_rows)

    # --- Finance simulation with constraints ---
    # Debt cap:
    # - LTC: max_ltc * (net cost budget incl land+friction, excl finance)
    # - LTV: max_ltv * GDV_net
    # Note: conservative banks apply to "total project cost"; using net-of-VAT economics for caps.
    cost_budget_net = build_total_net + prof_fees_net + statutory_net + marketing_net + overhead_net + land_net + friction_net
    max_debt_cap = min(float(a.max_ltc) * cost_budget_net, float(a.max_ltv) * gdv_net_total) if a.use_debt else 0.0

    # Minimum equity requirement on budget (net)
    min_equity_required = float(a.min_equity_pct) * cost_budget_net

    # Finance fees based on peak debt -> iterate
    arrangement_fee = 0.0
    exit_fee = 0.0

    # arrays we will output
    debt_bal = np.zeros(months)
    debt_draw = np.zeros(months)
    debt_repay = np.zeros(months)
    interest_cap = np.zeros(months)
    equity_inj = np.zeros(months)

    def simulate_with_fees(arr_fee: float, ex_fee: float) -> Tuple[float, float, float, float, float]:
        cash = 0.0
        debt = 0.0
        peak = 0.0
        equity_total = 0.0

        # Gate: debt cannot draw before presales achieved (unless allowed)
        gate = presales_gate_month

        for m in range(months):
            # monthly net cash movement before finance
            net = float(revenue_gross[m]) - float(costs_cash[m]) - float(vat_cash[m])

            if m == land_month:
                # already included in costs_cash; nothing extra
                pass

            if m == 0:
                net -= arr_fee
            if m == months - 1:
                net -= ex_fee

            cash += net

            # Interest-only during build: block repayments until last build month across all phases
            last_build_end = 0
            for r in product_rows:
                last_build_end = max(last_build_end, int(r["phase_start"]) + int(r["build_months"]))
            in_build = m < last_build_end

            # If cash negative, try debt draw subject to gate and cap; otherwise equity inject
            if cash < 0:
                need = -cash
                can_draw = 0.0
                if a.use_debt:
                    if a.allow_debt_draw_before_presales or (m >= gate):
                        can_draw = max(0.0, max_debt_cap - debt)
                draw = min(need, can_draw)
                if draw > 0:
                    debt += draw
                    debt_draw[m] += draw
                    cash += draw

                # if still negative => equity injection
                if cash < 0:
                    inj = -cash
                    equity_inj[m] += inj
                    equity_total += inj
                    cash = 0.0

            # If cash positive and we are allowed to repay (not interest-only build), repay debt
            if cash > 0 and debt > 0 and (not (a.debt_interest_only_during_build and in_build)):
                repay = min(cash, debt)
                debt -= repay
                debt_repay[m] += repay
                cash -= repay

            # Capitalise interest monthly
            if a.use_debt and debt > 0:
                i = debt * ((1 + interest_pa) ** (1 / 12) - 1)
                debt += i
                interest_cap[m] += i

            debt_bal[m] = debt
            peak = max(peak, debt)

        # If debt remains, force equity top-up at end
        if debt > 1e-6:
            equity_inj[-1] += debt
            equity_total += debt
            debt = 0.0
            debt_bal[-1] = 0.0

        # Ensure min equity requirement: if equity_total < min_equity_required, inject at month 0
        # (MVP: bank looks at equity contribution over life; we force extra at start)
        if equity_total < min_equity_required:
            top = min_equity_required - equity_total
            equity_inj[0] += top
            equity_total += top

        # Equity distribution at end: remaining cash after repayment isn’t explicitly tracked here.
        # Approx: equity distribution = profit_net (economic) + total equity injected
        return peak, equity_total, float(interest_cap.sum()), float(arr_fee), float(ex_fee)

    # Fee iteration (peak debt depends on fees; fees depend on peak)
    for _ in range(max_iter):
        # reset arrays each loop
        debt_draw[:] = 0
        debt_repay[:] = 0
        interest_cap[:] = 0
        equity_inj[:] = 0
        debt_bal[:] = 0

        peak, equity_total, interest_total, _, _ = simulate_with_fees(arrangement_fee, exit_fee)
        new_arr = peak * float(a.arrangement_fee_rate) if a.use_debt else 0.0
        new_exit = peak * float(a.exit_fee_rate) if a.use_debt else 0.0

        if abs(new_arr - arrangement_fee) < 10.0 and abs(new_exit - exit_fee) < 10.0:
            arrangement_fee, exit_fee = new_arr, new_exit
            break

        arrangement_fee = 0.6 * arrangement_fee + 0.4 * new_arr
        exit_fee = 0.6 * exit_fee + 0.4 * new_exit

    # final simulate once
    debt_draw[:] = 0
    debt_repay[:] = 0
    interest_cap[:] = 0
    equity_inj[:] = 0
    debt_bal[:] = 0
    peak_debt, equity_total, interest_total, _, _ = simulate_with_fees(arrangement_fee, exit_fee)

    finance_costs = float(interest_total + arrangement_fee + exit_fee)

    # --- Economics (net-of-VAT if recoverable) ---
    # Costs net ex land/fin: build_total_net + prof + statutory + marketing + overhead
    costs_net_ex_land_ex_fin = float(build_total_net + prof_fees_net + statutory_net + marketing_net + overhead_net)

    # Land net:
    if a.solve_residual_land:
        # residual: solve land to hit target profit (net economics)
        if a.target_profit_basis == "gdv":
            target_profit = float(a.target_profit_rate) * gdv_net_total
            land_net = gdv_net_total - (costs_net_ex_land_ex_fin + friction_net + finance_costs) - target_profit
        else:
            # gdv = (cost_ex_land_fin + land + friction + finance) * (1 + p)
            c = costs_net_ex_land_ex_fin + friction_net + finance_costs
            land_net = (gdv_net_total / (1.0 + float(a.target_profit_rate))) - c

    total_cost_net = float(costs_net_ex_land_ex_fin + land_net + friction_net + finance_costs)
    profit_net = float(gdv_net_total - total_cost_net)
    profit_on_cost = float(profit_net / total_cost_net) if total_cost_net else 0.0
    profit_on_gdv = float(profit_net / gdv_net_total) if gdv_net_total else 0.0

    # Equity IRR (monthly cashflows):
    # equity injections are outflows; end inflow = equity_total + profit_net (net economics)
    equity_cf = [-float(equity_inj[m]) for m in range(months)]
    equity_cf[-1] += float(equity_total + profit_net)
    irr_pa = annualize_monthly_rate(irr_monthly(equity_cf))

    # --- Build cashflow rows for UI ---
    cashflow_rows: List[Dict[str, Any]] = []
    for m in range(months):
        cashflow_rows.append({
            "Month": m + 1,
            "Revenue (gross)": float(revenue_gross[m]),
            "Costs (gross incl VAT)": float(costs_cash[m]),
            "VAT settlement (+pay / -refund)": float(vat_cash[m]),
            "Net pre-finance": float(revenue_gross[m] - costs_cash[m] - vat_cash[m]),
            "Debt balance": float(debt_bal[m]),
            "Debt draw": float(debt_draw[m]),
            "Debt repay": float(debt_repay[m]),
            "Interest (cap.)": float(interest_cap[m]),
            "Equity inject (auto)": float(equity_inj[m]),
        })

    # --- Audit ---
    audit.extend([
        {"section": "Rates", "key": "VAT rate", "value": vr, "unit": "ratio"},
        {"section": "Rates", "key": "Prime p.a.", "value": float(a.prime_rate_pa), "unit": "ratio"},
        {"section": "Rates", "key": "Debt margin over prime", "value": float(a.debt_margin_over_prime), "unit": "ratio"},
        {"section": "Finance", "key": "Debt interest p.a. (all-in)", "value": float(interest_pa), "unit": "ratio"},
        {"section": "Finance", "key": "Max LTC", "value": float(a.max_ltc), "unit": "ratio"},
        {"section": "Finance", "key": "Max LTV", "value": float(a.max_ltv), "unit": "ratio"},
        {"section": "Finance", "key": "Debt cap (R)", "value": float(max_debt_cap), "unit": ccy},
        {"section": "Finance", "key": "Arrangement fee", "value": float(arrangement_fee), "unit": ccy},
        {"section": "Finance", "key": "Exit fee", "value": float(exit_fee), "unit": ccy},
        {"section": "Finance", "key": "Finance costs (total)", "value": float(finance_costs), "unit": ccy},
        {"section": "Finance", "key": "Peak debt", "value": float(peak_debt), "unit": ccy},
        {"section": "VAT", "key": "VAT net settlement total (+pay/-refund)", "value": float(vat_net_payable_total), "unit": ccy},
        {"section": "Land", "key": "Land treatment", "value": land_treatment, "unit": ""},
        {"section": "Land", "key": "Land net", "value": float(land_net), "unit": ccy},
        {"section": "Land", "key": "Transfer duty", "value": float(transfer_duty), "unit": ccy},
        {"section": "Land", "key": "Legal conveyancing", "value": float(legal_net), "unit": ccy},
        {"section": "Land", "key": "Other disbursements", "value": float(other_net), "unit": ccy},
        {"section": "Revenue", "key": "GDV gross", "value": float(gdv_gross_total), "unit": ccy},
        {"section": "Revenue", "key": "GDV net", "value": float(gdv_net_total), "unit": ccy},
        {"section": "Costs", "key": "Build total net (incl uplift/escal/cont)", "value": float(build_total_net), "unit": ccy},
        {"section": "Costs", "key": "Professional fees net", "value": float(prof_fees_net), "unit": ccy},
        {"section": "Costs", "key": "Marketing net", "value": float(marketing_net), "unit": ccy},
        {"section": "Costs", "key": "Overhead net", "value": float(overhead_net), "unit": ccy},
        {"section": "Profit", "key": "Total cost net", "value": float(total_cost_net), "unit": ccy},
        {"section": "Profit", "key": "Profit net", "value": float(profit_net), "unit": ccy},
        {"section": "Profit", "key": "Profit % Cost", "value": float(profit_on_cost), "unit": "ratio"},
        {"section": "Profit", "key": "Profit % GDV", "value": float(profit_on_gdv), "unit": "ratio"},
        {"section": "Gate", "key": "Presales required % resi", "value": float(a.presales_required_pct_resi), "unit": "ratio"},
        {"section": "Gate", "key": "Presales gate month", "value": int(presales_gate_month), "unit": ""},
    ])

    # product rows for UI
    ui_products = []
    for r in product_rows:
        ui_products.append({
            "Product": r["name"],
            "Type": r["type"],
            "Phase": r["phase"],
            "Net sqm": r["net_sqm"],
            "Efficiency": r["efficiency"],
            "GBA sqm": r["gba_sqm"],
            "Price / Net sqm": r["sale_price_per_net_sqm"],
            "GDV gross": r["gdv_gross"],
            "GDV net": r["gdv_net"],
            "Build / GBA sqm": r["build_cost_per_gba_sqm"],
            "Build net": r["build_net"],
            "Off-plan share": r["offplan_share"],
            "Presales %": r["presales_pct"],
            "Presales month": r["presales_achieved_month"],
        })

    return Output(
        currency=ccy,
        gdv_net=float(gdv_net_total),
        costs_net_ex_land_ex_fin=float(costs_net_ex_land_ex_fin),
        land_net=float(land_net),
        friction_net=float(friction_net),
        finance_costs=float(finance_costs),
        total_cost_net=float(total_cost_net),
        profit_net=float(profit_net),
        profit_on_cost=float(profit_on_cost),
        profit_on_gdv=float(profit_on_gdv),
        peak_debt=float(peak_debt),
        equity_irr_pa=None if irr_pa is None else float(irr_pa),
        months=int(months),
        vat_net_payable_total=float(vat_net_payable_total),
        presales_gate_month=int(presales_gate_month if presales_gate_month < 10_000 else -1),
        product_rows=ui_products,
        cashflow_rows=cashflow_rows,
        audit=audit,
    )


def sensitivity_grid(a: Assumptions, price_steps=(-0.1, -0.05, 0, 0.05, 0.1), cost_steps=(-0.1, -0.05, 0, 0.05, 0.1)) -> Tuple[List[str], List[str], np.ndarray]:
    rows = [f"{int(p*100)}%" for p in price_steps]
    cols = [f"{int(c*100)}%" for c in cost_steps]
    mat = np.zeros((len(rows), len(cols)))

    for i, dp in enumerate(price_steps):
        for j, dc in enumerate(cost_steps):
            aa = Assumptions.from_dict(a.to_dict())
            prods = []
            for p in aa.products:
                pp = dict(p)
                pp["sale_price_per_net_sqm"] = _safe_float(pp.get("sale_price_per_net_sqm")) * (1 + dp)
                pp["build_cost_per_gba_sqm"] = _safe_float(pp.get("build_cost_per_gba_sqm")) * (1 + dc)
                prods.append(pp)
            aa.products = prods
            out = run_appraisal(aa)
            mat[i, j] = out.land_net if aa.solve_residual_land else out.profit_net
    return rows, cols, mat
