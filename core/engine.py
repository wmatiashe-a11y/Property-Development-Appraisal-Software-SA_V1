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


def _annuity_payment(principal: float, monthly_rate: float, n_months: int) -> float:
    if principal <= 0 or n_months <= 0:
        return 0.0
    r = max(0.0, monthly_rate)
    if r == 0:
        return principal / n_months
    a = (1 + r) ** n_months
    return principal * (r * a) / (a - 1)


def _compute_presales_gate_month(a: Assumptions, product_meta: List[Dict[str, Any]]) -> int:
    required_pct = float(a.presales_required_pct_resi)
    if required_pct <= 0:
        return 0

    resi_rows = [r for r in product_meta if r.get("is_residential_sale")]
    total_resi_gdv_gross = sum(float(r.get("gdv_gross", 0.0)) for r in resi_rows)
    if total_resi_gdv_gross <= 0:
        return 0

    target_value = required_pct * total_resi_gdv_gross

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
    return 10_000


def run_appraisal(a: Assumptions, max_iter: int = 30) -> Output:
    ccy = a.currency or "ZAR"
    audit: List[Dict[str, Any]] = []

    phase_by_name = _phase_map(a.phases)
    if not phase_by_name:
        phase_by_name = {"Phase 1": {"name": "Phase 1", "start_month": 0, "build_months": 18, "sales_months": 12, "sales_curve": "linear"}}

    vat_cfg = VATConfig(**(a.vat or {}))
    vr = float(vat_cfg.vat_rate)

    # Rates
    construction_rate_pa = float(a.prime_rate_pa + a.debt_margin_over_prime)
    construction_rate_m = (1 + construction_rate_pa) ** (1 / 12) - 1

    term_rate_pa = float(a.prime_rate_pa + a.term_margin_over_prime)
    term_rate_m = (1 + term_rate_pa) ** (1 / 12) - 1
    term_n = int(max(1, int(a.term_amort_years) * 12))

    # ---------- PRODUCT META ----------
    product_meta: List[Dict[str, Any]] = []
    build_net_total = 0.0
    sale_gdv_gross_total = 0.0
    sale_gdv_net_total = 0.0

    rental_value_net_for_ltv = 0.0
    rental_noi_annual_total = 0.0

    for p in (a.products or []):
        name = str(p.get("name") or "Product")
        ptype = str(p.get("type") or "residential_sale")
        phase = str(p.get("phase") or "Phase 1")
        ph = phase_by_name.get(phase) or list(phase_by_name.values())[0]

        start = int(_safe_float(ph.get("start_month", 0)))
        build_months = int(_safe_float(ph.get("build_months", 18)))
        sales_months = int(_safe_float(ph.get("sales_months", 12)))
        sales_curve = str(ph.get("sales_curve", "linear"))

        eff = float(np.clip(_safe_float(p.get("efficiency_ratio", 0.83)), 0.50, 0.98))
        is_res_sale = ptype == "residential_sale"
        is_sale = ptype in ("residential_sale", "commercial_sale")
        is_rental = ptype == "rental_yield"

        # Net area
        if is_res_sale:
            units = int(_safe_float(p.get("units", 0)))
            avg_net = _safe_float(p.get("avg_unit_net_sqm", 0.0))
            net_sqm = units * avg_net
        else:
            net_sqm = _safe_float(p.get("net_sqm", 0.0))

        gba_sqm = net_sqm / eff if eff > 0 else 0.0

        # Build cost net
        build_cost_gba = _safe_float(p.get("build_cost_per_gba_sqm", 0.0))
        build_net = gba_sqm * build_cost_gba
        if vat_cfg.enabled and vat_cfg.costs_include_vat:
            build_net, _ = split_vat_from_gross(build_net, vr)

        build_net_total += build_net

        # Sale GDV with IH overlay
        gdv_gross = 0.0
        gdv_net = 0.0
        ih_sqm = 0.0
        ih_value_gross = 0.0

        if is_sale:
            sale_price = _safe_float(p.get("sale_price_per_net_sqm", 0.0))
            ih_eligible = bool(p.get("ih_eligible", is_res_sale))

            if a.inclusionary_enabled and ih_eligible and is_res_sale and float(a.inclusionary_rate) > 0:
                ih_sqm = net_sqm * float(np.clip(a.inclusionary_rate, 0.0, 1.0))
                mkt_sqm = net_sqm - ih_sqm
                ih_value_gross = ih_sqm * float(a.inclusionary_price_per_net_sqm)
                gdv_gross = mkt_sqm * sale_price + ih_value_gross
            else:
                gdv_gross = net_sqm * sale_price

            if vat_cfg.enabled and vat_cfg.prices_include_vat:
                gdv_net, _ = split_vat_from_gross(gdv_gross, vr)
            elif vat_cfg.enabled and (not vat_cfg.prices_include_vat):
                gdv_net = gdv_gross
                gdv_gross = gdv_gross * (1 + vr)
            else:
                gdv_net = gdv_gross

            sale_gdv_gross_total += gdv_gross
            sale_gdv_net_total += gdv_net

        # Rental inputs (for value & NOI)
        rent_psm_m = _safe_float(p.get("rent_per_net_sqm_month", 0.0))
        opex_ratio = float(np.clip(_safe_float(p.get("opex_ratio", 0.35)), 0.0, 0.95))
        vac = float(np.clip(_safe_float(p.get("vacancy_stabilized", 0.05)), 0.0, 0.50))
        letting = int(max(0, _safe_float(p.get("letting_up_months", 12))))
        hold_after_build = int(max(0, _safe_float(p.get("hold_months_after_build", 36))))
        exit_cap = float(np.clip(_safe_float(p.get("exit_cap_rate", 0.095)), 0.03, 0.30))
        selling_cost_rate = float(np.clip(_safe_float(p.get("selling_cost_rate", 0.02)), 0.0, 0.10))
        rent_is_vat_exclusive = bool(p.get("rent_is_vat_exclusive", True))

        product_meta.append({
            "name": name,
            "type": ptype,
            "phase": phase,
            "start": start,
            "build_months": build_months,
            "sales_months": sales_months,
            "sales_curve": sales_curve,

            "net_sqm": float(net_sqm),
            "efficiency": float(eff),
            "gba_sqm": float(gba_sqm),

            "build_cost_per_gba_sqm": float(build_cost_gba),
            "build_net": float(build_net),

            "is_sale": is_sale,
            "is_residential_sale": is_res_sale,
            "gdv_gross": float(gdv_gross),
            "gdv_net": float(gdv_net),
            "ih_sqm": float(ih_sqm),
            "ih_value_gross": float(ih_value_gross),

            "offplan_share": float(np.clip(_safe_float(p.get("offplan_share", 0.0)), 0.0, 1.0)),
            "deposit_pct": float(np.clip(_safe_float(p.get("deposit_pct", 0.0)), 0.0, 0.30)),
            "deposit_released_during_build": bool(p.get("deposit_released_during_build", False)),
            "presales_pct": float(np.clip(_safe_float(p.get("presales_pct", 0.0)), 0.0, 1.0)),
            "presales_achieved_month": int(_safe_float(p.get("presales_achieved_month", 0))),

            "is_rental": is_rental,
            "rent_per_net_sqm_month": float(rent_psm_m),
            "rent_is_vat_exclusive": rent_is_vat_exclusive,
            "opex_ratio": float(opex_ratio),
            "vacancy_stabilized": float(vac),
            "letting_up_months": letting,
            "hold_months_after_build": hold_after_build,
            "exit_cap_rate": float(exit_cap),
            "selling_cost_rate": float(selling_cost_rate),
        })

        if is_rental:
            occ_target = max(0.0, 1.0 - vac)
            annual_gross_rent_net = net_sqm * rent_psm_m * occ_target * 12.0
            noi_annual = annual_gross_rent_net * (1.0 - opex_ratio)
            rental_noi_annual_total += noi_annual
            rental_value_net_for_ltv += (noi_annual / max(1e-6, exit_cap))

    # ---------- TIMELINE ----------
    end_month = 0
    last_build_end = 0
    for r in product_meta:
        start = int(r["start"])
        b = int(r["build_months"])
        last_build_end = max(last_build_end, start + b)
        if r["is_sale"]:
            end_month = max(end_month, start + b + int(r["sales_months"]))
        else:
            end_month = max(end_month, start + b + int(r["hold_months_after_build"]))
    months = int(max(1, end_month))

    # ---------- COSTS (NET) ----------
    heritage_uplift = build_net_total * float(a.heritage_cost_uplift_rate) if a.heritage_enabled else 0.0
    build_net2 = build_net_total + heritage_uplift

    avg_build_months = float(np.mean([int(r["build_months"]) for r in product_meta]) if product_meta else 18.0)
    build_years = avg_build_months / 12.0
    escalation = build_net2 * float(a.escalation_rate_pa) * (build_years / 2.0)
    contingency = (build_net2 + escalation) * float(a.contingency_rate)
    build_total_net = build_net2 + escalation + contingency

    prof_fees_net = build_total_net * float(a.professional_fees_rate)
    statutory_net = float(a.statutory_costs)
    marketing_net = sale_gdv_net_total * float(a.marketing_rate)
    overhead_net = float(a.overhead_per_month) * months

    costs_net = np.zeros(months)

    build_net_sum = max(1e-9, sum(float(r["build_net"]) for r in product_meta))
    for r in product_meta:
        start = int(r["start"])
        b = int(r["build_months"])
        w = _build_weights(b)
        share = float(r["build_net"]) / build_net_sum
        costs_net[start:start+b] += (build_total_net * share) * w

    build_active = np.zeros(months)
    for r in product_meta:
        start = int(r["start"])
        b = int(r["build_months"])
        build_active[start:start+b] = 1.0
    build_idxs = np.where(build_active > 0)[0]
    if len(build_idxs) == 0:
        build_idxs = np.array([0])

    costs_net[build_idxs] += prof_fees_net / len(build_idxs)
    costs_net[build_idxs] += statutory_net / len(build_idxs)

    gdv_sale_net_sum = max(1e-9, sum(float(r["gdv_net"]) for r in product_meta if r["is_sale"]))
    for r in product_meta:
        if not r["is_sale"]:
            continue
        start = int(r["start"])
        b = int(r["build_months"])
        s = int(r["sales_months"])
        w = _weights(s, r["sales_curve"])
        share = float(r["gdv_net"]) / gdv_sale_net_sum
        costs_net[start+b:start+b+s] += (marketing_net * share) * w

    costs_net += overhead_net / months

    # ---------- REVENUE (CASH GROSS) ----------
    revenue_gross = np.zeros(months)

    # Sale receipts
    for r in product_meta:
        if not r["is_sale"]:
            continue
        start = int(r["start"])
        b = int(r["build_months"])
        s = int(r["sales_months"])
        w_build = _weights(b, "back")
        w_sales = _weights(s, r["sales_curve"])
        gdv_gross = float(r["gdv_gross"])

        offplan = float(r["offplan_share"])
        dep_pct = float(r["deposit_pct"])
        dep_cash = gdv_gross * offplan * dep_pct
        completion_cash = gdv_gross - dep_cash

        if bool(r["deposit_released_during_build"]) and dep_cash > 0:
            revenue_gross[start:start+b] += dep_cash * w_build
        else:
            completion_cash += dep_cash
        revenue_gross[start+b:start+b+s] += completion_cash * w_sales

    # Rental receipts + exit via cap (cash)
    rental_income_net_total = 0.0
    rental_exit_value_net_total = 0.0
    rental_exit_value_gross_total = 0.0

    # For refinance month: define stabilisation month as max(end_of_letting_up) across rental lines
    stabilisation_month = -1

    for r in product_meta:
        if not r["is_rental"]:
            continue

        start = int(r["start"])
        b = int(r["build_months"])
        hold = int(r["hold_months_after_build"])
        hold_start = start + b
        hold_end = min(months, hold_start + hold)

        net_sqm = float(r["net_sqm"])
        rent_psm_m = float(r["rent_per_net_sqm_month"])
        opex_ratio = float(r["opex_ratio"])
        vac = float(r["vacancy_stabilized"])
        letting = int(r["letting_up_months"])
        exit_cap = float(r["exit_cap_rate"])
        sell_cost = float(r["selling_cost_rate"])

        occ_target = max(0.0, 1.0 - vac)
        stabilisation_month = max(stabilisation_month, hold_start + max(0, letting) - 1)

        # Monthly rent stream
        for m in range(hold_start, hold_end):
            t = m - hold_start + 1
            occ = occ_target if letting <= 0 else occ_target * min(1.0, t / letting)
            gross_rent_cash = net_sqm * rent_psm_m * occ

            # Rent VAT handling: if rent is VAT-exclusive and VAT enabled, rent cash includes VAT
            if vat_cfg.enabled and bool(r["rent_is_vat_exclusive"]):
                gross_rent_cash *= (1 + vr)

            revenue_gross[m] += gross_rent_cash

            # Net economics rent
            if vat_cfg.enabled and bool(r["rent_is_vat_exclusive"]):
                net_rent = gross_rent_cash / (1 + vr)
            else:
                net_rent = split_vat_from_gross(gross_rent_cash, vr)[0] if vat_cfg.enabled and vat_cfg.prices_include_vat else gross_rent_cash
            rental_income_net_total += net_rent

        # Stabilised NOI annual (net-of-VAT)
        annual_gross_rent_net = net_sqm * rent_psm_m * occ_target * 12.0
        noi_stab_annual = annual_gross_rent_net * (1.0 - opex_ratio)

        # Exit value (net) = NOI / cap, then selling costs
        exit_value_net = (noi_stab_annual / max(1e-6, exit_cap)) * (1.0 - sell_cost)
        exit_value_gross = exit_value_net * (1 + vr) if vat_cfg.enabled and vat_cfg.prices_include_vat else exit_value_net

        exit_month = hold_end - 1 if hold_end > hold_start else hold_start
        if 0 <= exit_month < months:
            revenue_gross[exit_month] += exit_value_gross

        rental_exit_value_net_total += exit_value_net
        rental_exit_value_gross_total += exit_value_gross

        r["noi_stab_annual"] = float(noi_stab_annual)
        r["exit_value_net"] = float(exit_value_net)
        r["exit_value_gross"] = float(exit_value_gross)

    # If no rental lines, disable term refi (no point)
    has_rental = any(r["is_rental"] for r in product_meta)
    if stabilisation_month < 0:
        stabilisation_month = last_build_end  # fallback

    # ---------- LAND + FRICTION ----------
    earliest_start = min(int(r["start"]) for r in product_meta) if product_meta else 0
    land_month = max(0, earliest_start)

    land_gross_entered = float(a.land_price)
    land_treatment = str(a.land_treatment)

    transfer_duty = 0.0
    land_vat_component = 0.0
    land_net = land_gross_entered

    if a.solve_residual_land:
        land_net = 0.0
        land_gross_entered = 0.0
        land_treatment = "transfer_duty"

    if vat_cfg.enabled:
        if land_treatment == "vat_standard":
            if vat_cfg.prices_include_vat:
                land_net, land_vat_component = split_vat_from_gross(land_gross_entered, vr)
            else:
                land_net = land_gross_entered
                land_gross_entered *= (1 + vr)
                land_vat_component = land_gross_entered - land_net
        elif land_treatment == "vat_zero":
            land_net = land_gross_entered
            land_vat_component = 0.0
        else:
            transfer_duty = transfer_duty_za(land_gross_entered)
    else:
        transfer_duty = transfer_duty_za(land_gross_entered)

    legal_net = float(a.legal_conveyancing)
    other_net = float(a.land_other_disbursements)
    friction_net = float(transfer_duty + legal_net + other_net)

    # ---------- VAT LEDGER + SETTLEMENT ----------
    vat_out = np.zeros(months)
    vat_in = np.zeros(months)

    if vat_cfg.enabled:
        for m in range(months):
            _, v = split_vat_from_gross(float(revenue_gross[m]), vr)
            vat_out[m] += v
        for m in range(months):
            net_cost = float(costs_net[m])
            if net_cost > 0:
                _, vin = add_vat_to_net(net_cost, vr)
                vat_in[m] += vin
        if land_vat_component > 0:
            vat_in[land_month] += land_vat_component

    vat_settle = np.zeros(months)
    if vat_cfg.enabled:
        lag = max(0, int(vat_cfg.settlement_lag_months))
        for m in range(months):
            payable = vat_out[m] - (vat_in[m] if vat_cfg.input_vat_recoverable else 0.0)
            pm = m + lag
            if pm < months:
                vat_settle[pm] += payable

    vat_net_payable_total = float(vat_settle.sum())

    # ---------- CASH COSTS (GROSS) ----------
    costs_cash = np.zeros(months)
    for m in range(months):
        net_cost = float(costs_net[m])
        if vat_cfg.enabled:
            gross, _ = add_vat_to_net(net_cost, vr)
            costs_cash[m] = gross
        else:
            costs_cash[m] = net_cost

    costs_cash[land_month] += land_gross_entered + friction_net
    vat_cash = vat_settle.copy()

    # ---------- PRESALES GATE ----------
    for r in product_meta:
        r["presales_value_gross"] = float(r.get("gdv_gross", 0.0)) * float(r.get("presales_pct", 0.0)) if r["is_residential_sale"] else 0.0
    presales_gate_month = _compute_presales_gate_month(a, product_meta)

    # ---------- ECON TOTALS (NET) ----------
    gdv_net_total = float(sale_gdv_net_total + rental_income_net_total + rental_exit_value_net_total)
    costs_net_ex_land_ex_fin = float(build_total_net + prof_fees_net + statutory_net + marketing_net + overhead_net)

    # Debt caps use net economics (conservative)
    cost_budget_net = costs_net_ex_land_ex_fin + land_net + friction_net
    max_debt_cap = min(float(a.max_ltc) * cost_budget_net, float(a.max_ltv) * gdv_net_total) if a.use_debt else 0.0
    min_equity_required = float(a.min_equity_pct) * cost_budget_net

    # ---------- TERM DEBT SIZING (min LTV, DSCR) ----------
    refinance_month = -1
    term_loan_amt = 0.0
    term_dscr_at_refi: Optional[float] = None

    if a.use_debt and a.enable_term_debt and has_rental and a.refinance_at_stabilisation:
        refinance_month = int(max(0, min(months - 1, stabilisation_month + int(a.refinance_month_offset))))
        # Value basis (net) = sum NOI/cap (pre selling costs)
        value_net = float(rental_value_net_for_ltv)
        noi_annual = float(rental_noi_annual_total)

        # LTV cap
        loan_by_ltv = float(a.term_max_ltv) * value_net

        # DSCR cap:
        # DSCR = NOI / ADS, where ADS = annual debt service.
        # debt constant = ADS / principal for an amortising loan at (term_rate, term_n)
        payment_per_1 = _annuity_payment(1.0, term_rate_m, term_n)
        annual_ds_per_1 = payment_per_1 * 12.0
        debt_constant = annual_ds_per_1  # because principal = 1.0
        loan_by_dscr = noi_annual / max(1e-9, (float(a.term_dscr_min) * debt_constant))

        term_loan_amt = max(0.0, min(loan_by_ltv, loan_by_dscr))

        # DSCR at refi using the sized term loan
        if term_loan_amt > 0:
            pay = _annuity_payment(term_loan_amt, term_rate_m, term_n)
            ads = pay * 12.0
            term_dscr_at_refi = noi_annual / ads if ads > 0 else None
        else:
            term_dscr_at_refi = None

    # ---------- FINANCE SIM ----------
    arrangement_fee = 0.0
    exit_fee = 0.0

    debt_bal = np.zeros(months)
    debt_draw = np.zeros(months)
    debt_repay = np.zeros(months)
    interest_cap = np.zeros(months)      # construction interest capitalised
    debt_service = np.zeros(months)      # term debt service (cash)
    equity_inj = np.zeros(months)

    def simulate(arr_fee: float, ex_fee: float) -> tuple[float, float, float]:
        cash = 0.0
        debt = 0.0
        peak = 0.0
        eq_total = 0.0
        gate = presales_gate_month

        # fixed term payment once refi happens
        fixed_term_payment = _annuity_payment(term_loan_amt, term_rate_m, term_n) if (term_loan_amt > 0) else 0.0

        for m in range(months):
            net = float(revenue_gross[m]) - float(costs_cash[m]) - float(vat_cash[m])
            if m == 0:
                net -= arr_fee
            if m == months - 1:
                net -= ex_fee
            cash += net

            # 🔁 Refinance event (swap construction debt → term debt)
            if refinance_month >= 0 and m == refinance_month and a.enable_term_debt and term_loan_amt > 0:
                # pay off construction debt using term loan proceeds (no cash by default)
                # if term loan smaller than existing debt => equity injection to close the gap
                if term_loan_amt < debt:
                    shortfall = debt - term_loan_amt
                    equity_inj[m] += shortfall
                    eq_total += shortfall
                    debt = term_loan_amt
                else:
                    # if term > debt: optionally cash-out
                    surplus = term_loan_amt - debt
                    debt = term_loan_amt
                    if a.allow_cash_out_refi and surplus > 0:
                        cash += surplus

            in_build = m < last_build_end

            # If cash negative, try draw debt (construction/term) subject to constraints and gate
            if cash < 0:
                need = -cash
                can_draw = 0.0
                if a.use_debt:
                    # if we're in term phase (post refi), allow draw only up to term loan amount (no revolver)
                    if refinance_month >= 0 and m >= refinance_month and term_loan_amt > 0:
                        can_draw = 0.0
                    else:
                        if a.allow_debt_draw_before_presales or (m >= gate):
                            can_draw = max(0.0, max_debt_cap - debt)
                draw = min(need, can_draw)
                if draw > 0:
                    debt += draw
                    debt_draw[m] += draw
                    cash += draw

                if cash < 0:
                    inj = -cash
                    equity_inj[m] += inj
                    eq_total += inj
                    cash = 0.0

            # Term debt service (cash) after refi
            if refinance_month >= 0 and m >= refinance_month and term_loan_amt > 0 and debt > 0:
                # Pay scheduled payment; last payment may be smaller
                interest = debt * term_rate_m
                principal = min(debt, max(0.0, fixed_term_payment - interest))
                pay = interest + principal
                cash -= pay
                debt -= principal
                debt_service[m] += pay

                if cash < 0:
                    inj = -cash
                    equity_inj[m] += inj
                    eq_total += inj
                    cash = 0.0

            # Construction: interest-only during build, capitalised
            if a.use_debt and debt > 0 and (refinance_month < 0 or m < refinance_month):
                # capitalise interest monthly
                i = debt * construction_rate_m
                debt += i
                interest_cap[m] += i

            # Allow repayments outside build once sales start (pre refi)
            if cash > 0 and debt > 0 and (refinance_month < 0 or m < refinance_month):
                if not (a.debt_interest_only_during_build and in_build):
                    repay = min(cash, debt)
                    debt -= repay
                    debt_repay[m] += repay
                    cash -= repay

            debt_bal[m] = debt
            peak = max(peak, debt)

        # force close any remaining debt at end with equity
        if debt > 1e-6:
            equity_inj[-1] += debt
            eq_total += debt
            debt = 0.0
            debt_bal[-1] = 0.0

        # enforce min equity requirement
        if eq_total < min_equity_required:
            top = min_equity_required - eq_total
            equity_inj[0] += top
            eq_total += top

        # finance costs: construction interest capitalised + term debt service interest already in debt_service (cash)
        # We’ll count total term debt service as finance cashflow, but for economic “finance cost” we want interest+fees.
        # MVP: treat full debt service as finance cash impact; for headline finance_costs we’ll use interest_cap + fees + (debt_service - principal repaid).
        return float(peak), float(eq_total), float(interest_cap.sum())

    # iterate fees (peak debt depends on fees)
    for _ in range(max_iter):
        debt_draw[:] = 0
        debt_repay[:] = 0
        interest_cap[:] = 0
        debt_service[:] = 0
        equity_inj[:] = 0
        debt_bal[:] = 0

        peak, eq_total, constr_interest = simulate(arrangement_fee, exit_fee)
        new_arr = peak * float(a.arrangement_fee_rate) if a.use_debt else 0.0
        new_exit = peak * float(a.exit_fee_rate) if a.use_debt else 0.0

        if abs(new_arr - arrangement_fee) < 10.0 and abs(new_exit - exit_fee) < 10.0:
            arrangement_fee, exit_fee = new_arr, new_exit
            break

        arrangement_fee = 0.6 * arrangement_fee + 0.4 * new_arr
        exit_fee = 0.6 * exit_fee + 0.4 * new_exit

    # final run
    debt_draw[:] = 0
    debt_repay[:] = 0
    interest_cap[:] = 0
    debt_service[:] = 0
    equity_inj[:] = 0
    debt_bal[:] = 0
    peak_debt, equity_total, constr_interest_total = simulate(arrangement_fee, exit_fee)

    # Split term debt service into interest/principal (approx by replaying amort schedule on arrays)
    # For MVP headline finance costs: construction interest + fees + term interest portion.
    term_interest_total = 0.0
    if refinance_month >= 0 and term_loan_amt > 0:
        bal = term_loan_amt
        fixed = _annuity_payment(term_loan_amt, term_rate_m, term_n)
        for m in range(refinance_month, months):
            if bal <= 0:
                break
            intr = bal * term_rate_m
            prin = min(bal, max(0.0, fixed - intr))
            pay = intr + prin
            # only count months where we actually paid service (debt_service row)
            if debt_service[m] > 0:
                term_interest_total += intr
            bal -= prin
            if pay <= 0:
                break

    finance_costs = float(constr_interest_total + arrangement_fee + exit_fee + term_interest_total)

    # ---------- RESIDUAL LAND (OPTIONAL) ----------
    if a.solve_residual_land:
        if a.target_profit_basis == "gdv":
            target_profit = float(a.target_profit_rate) * gdv_net_total
            land_net = gdv_net_total - (costs_net_ex_land_ex_fin + friction_net + finance_costs) - target_profit
        else:
            c = costs_net_ex_land_ex_fin + friction_net + finance_costs
            land_net = (gdv_net_total / (1.0 + float(a.target_profit_rate))) - c

    total_cost_net = float(costs_net_ex_land_ex_fin + land_net + friction_net + finance_costs)
    profit_net = float(gdv_net_total - total_cost_net)
    profit_on_cost = float(profit_net / total_cost_net) if total_cost_net else 0.0
    profit_on_gdv = float(profit_net / gdv_net_total) if gdv_net_total else 0.0

    # Equity IRR (monthly)
    equity_cf = [-float(equity_inj[m]) for m in range(months)]
    equity_cf[-1] += float(equity_total + profit_net)
    irr_pa = annualize_monthly_rate(irr_monthly(equity_cf))

    # ---------- CASHFLOW ROWS ----------
    cashflow_rows: List[Dict[str, Any]] = []
    for m in range(months):
        cashflow_rows.append({
            "Month": m + 1,
            "Revenue (gross)": float(revenue_gross[m]),
            "Costs (gross incl VAT)": float(costs_cash[m]),
            "VAT settlement (+pay / -refund)": float(vat_cash[m]),
            "Debt service (term)": float(debt_service[m]),
            "Net pre-finance": float(revenue_gross[m] - costs_cash[m] - vat_cash[m] - debt_service[m]),
            "Debt balance": float(debt_bal[m]),
            "Debt draw": float(debt_draw[m]),
            "Debt repay": float(debt_repay[m]),
            "Interest (cap., construction)": float(interest_cap[m]),
            "Equity inject (auto)": float(equity_inj[m]),
        })

    # ---------- PRODUCT ROWS (UI) ----------
    ui_products: List[Dict[str, Any]] = []
    for r in product_meta:
        row = {
            "Product": r["name"],
            "Type": r["type"],
            "Phase": r["phase"],
            "Net sqm": r["net_sqm"],
            "Efficiency": r["efficiency"],
            "GBA sqm": r["gba_sqm"],
            "Build / GBA sqm": r["build_cost_per_gba_sqm"],
            "Build net": r["build_net"],
        }
        if r["is_sale"]:
            row.update({
                "Price / Net sqm": _safe_float(next((p.get("sale_price_per_net_sqm") for p in a.products if p.get("name") == r["name"]), r.get("sale_price_per_net_sqm", 0.0))),
                "GDV gross": r["gdv_gross"],
                "GDV net": r["gdv_net"],
                "IH sqm": r["ih_sqm"],
                "IH value (gross)": r["ih_value_gross"],
                "Off-plan share": r["offplan_share"],
                "Presales %": r["presales_pct"],
                "Presales month": r["presales_achieved_month"],
            })
        if r["is_rental"]:
            row.update({
                "Rent / net sqm / month": r["rent_per_net_sqm_month"],
                "Vacancy (stab.)": r["vacancy_stabilized"],
                "Opex ratio": r["opex_ratio"],
                "Letting-up months": r["letting_up_months"],
                "Hold months (post build)": r["hold_months_after_build"],
                "Exit cap": r["exit_cap_rate"],
                "NOI stabilised (annual)": float(r.get("noi_stab_annual", 0.0)),
                "Exit value (net)": float(r.get("exit_value_net", 0.0)),
            })
        ui_products.append(row)

    # ---------- AUDIT ----------
    audit.extend([
        {"section": "Revenue", "key": "Sale GDV net", "value": float(sale_gdv_net_total), "unit": ccy},
        {"section": "Revenue", "key": "Rental income net (total)", "value": float(rental_income_net_total), "unit": ccy},
        {"section": "Revenue", "key": "Rental exit value net", "value": float(rental_exit_value_net_total), "unit": ccy},
        {"section": "Revenue", "key": "Total revenue net", "value": float(gdv_net_total), "unit": ccy},

        {"section": "Overlays", "key": "IH enabled", "value": str(bool(a.inclusionary_enabled)), "unit": ""},
        {"section": "Overlays", "key": "IH rate", "value": float(a.inclusionary_rate), "unit": "ratio"},
        {"section": "Overlays", "key": "IH price / net sqm", "value": float(a.inclusionary_price_per_net_sqm), "unit": ccy},
        {"section": "Overlays", "key": "Heritage enabled", "value": str(bool(a.heritage_enabled)), "unit": ""},
        {"section": "Overlays", "key": "Heritage uplift rate", "value": float(a.heritage_cost_uplift_rate), "unit": "ratio"},

        {"section": "Finance", "key": "Construction rate p.a.", "value": float(construction_rate_pa), "unit": "ratio"},
        {"section": "Finance", "key": "Term rate p.a.", "value": float(term_rate_pa), "unit": "ratio"},
        {"section": "Finance", "key": "Refinance month (0-index)", "value": int(refinance_month), "unit": ""},
        {"section": "Finance", "key": "Term loan amount (sized by min LTV/DSCR)", "value": float(term_loan_amt), "unit": ccy},
        {"section": "Finance", "key": "Term DSCR @ refi", "value": (0.0 if term_dscr_at_refi is None else float(term_dscr_at_refi)), "unit": "ratio"},

        {"section": "Finance", "key": "Debt cap (construction: min LTC/LTV)", "value": float(max_debt_cap), "unit": ccy},
        {"section": "Finance", "key": "Arrangement fee", "value": float(arrangement_fee), "unit": ccy},
        {"section": "Finance", "key": "Exit fee", "value": float(exit_fee), "unit": ccy},
        {"section": "Finance", "key": "Finance costs (interest + fees)", "value": float(finance_costs), "unit": ccy},
        {"section": "Finance", "key": "Peak debt", "value": float(peak_debt), "unit": ccy},

        {"section": "VAT", "key": "VAT settlement total (+pay / -refund)", "value": float(vat_net_payable_total), "unit": ccy},

        {"section": "Costs", "key": "Build total net (incl uplift/escal/cont)", "value": float(build_total_net), "unit": ccy},
        {"section": "Costs", "key": "Professional fees net", "value": float(prof_fees_net), "unit": ccy},
        {"section": "Costs", "key": "Statutory net", "value": float(statutory_net), "unit": ccy},
        {"section": "Costs", "key": "Marketing net", "value": float(marketing_net), "unit": ccy},
        {"section": "Costs", "key": "Overhead net", "value": float(overhead_net), "unit": ccy},

        {"section": "Land", "key": "Land treatment", "value": land_treatment, "unit": ""},
        {"section": "Land", "key": "Land net", "value": float(land_net), "unit": ccy},
        {"section": "Land", "key": "Transfer duty", "value": float(transfer_duty), "unit": ccy},
        {"section": "Land", "key": "Legal", "value": float(legal_net), "unit": ccy},
        {"section": "Land", "key": "Other disbursements", "value": float(other_net), "unit": ccy},
        {"section": "Land", "key": "Friction net", "value": float(friction_net), "unit": ccy},

        {"section": "Gate", "key": "Presales required (resi)", "value": float(a.presales_required_pct_resi), "unit": "ratio"},
        {"section": "Gate", "key": "Presales gate month", "value": int(presales_gate_month if presales_gate_month < 10_000 else -1), "unit": ""},

        {"section": "Profit", "key": "Total cost net", "value": float(costs_net_ex_land_ex_fin + land_net + friction_net + finance_costs), "unit": ccy},
        {"section": "Profit", "key": "Profit net", "value": float(profit_net), "unit": ccy},
        {"section": "Profit", "key": "Profit % Cost", "value": float(profit_on_cost), "unit": "ratio"},
        {"section": "Profit", "key": "Profit % Revenue", "value": float(profit_on_gdv), "unit": "ratio"},
    ])

    return Output(
        currency=ccy,
        gdv_net=float(gdv_net_total),
        costs_net_ex_land_ex_fin=float(costs_net_ex_land_ex_fin),
        land_net=float(land_net),
        friction_net=float(friction_net),
        finance_costs=float(finance_costs),
        total_cost_net=float(costs_net_ex_land_ex_fin + land_net + friction_net + finance_costs),
        profit_net=float(profit_net),
        profit_on_cost=float(profit_on_cost),
        profit_on_gdv=float(profit_on_gdv),
        peak_debt=float(peak_debt),
        equity_irr_pa=None if irr_pa is None else float(irr_pa),
        months=int(months),
        vat_net_payable_total=float(vat_net_payable_total),
        presales_gate_month=int(presales_gate_month if presales_gate_month < 10_000 else -1),
        refinance_month=int(refinance_month),
        term_loan_amount=float(term_loan_amt),
        term_dscr_at_refi=term_dscr_at_refi,
        product_rows=ui_products,
        cashflow_rows=cashflow_rows,
        audit=audit,
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
            prods = []
            for p in aa.products:
                pp = dict(p)
                if pp.get("type") in ("residential_sale", "commercial_sale") and "sale_price_per_net_sqm" in pp:
                    pp["sale_price_per_net_sqm"] = _safe_float(pp.get("sale_price_per_net_sqm")) * (1 + dp)
                if pp.get("type") == "rental_yield" and "rent_per_net_sqm_month" in pp:
                    pp["rent_per_net_sqm_month"] = _safe_float(pp.get("rent_per_net_sqm_month")) * (1 + dp)
                if "build_cost_per_gba_sqm" in pp:
                    pp["build_cost_per_gba_sqm"] = _safe_float(pp.get("build_cost_per_gba_sqm")) * (1 + dc)
                prods.append(pp)
            aa.products = prods
            out = run_appraisal(aa)
            mat[i, j] = out.land_net if aa.solve_residual_land else out.profit_net
    return rows, cols, mat
