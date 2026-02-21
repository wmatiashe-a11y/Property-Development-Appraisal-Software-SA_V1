from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import copy

from .taxes_sa import VATConfig


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def default_phases() -> List[Dict[str, Any]]:
    return [
        {"name": "Phase 1", "start_month": 0, "build_months": 18, "sales_months": 12, "sales_curve": "linear"}
    ]


def default_products() -> List[Dict[str, Any]]:
    return [
        {
            "name": "Residential (Sale)",
            "type": "residential_sale",
            "phase": "Phase 1",
            "units": 60,
            "avg_unit_net_sqm": 55.0,
            "net_sqm": None,
            "efficiency_ratio": 0.83,
            "sale_price_per_net_sqm": 38000.0,
            "build_cost_per_gba_sqm": 17500.0,
            "offplan_share": 0.60,
            "deposit_pct": 0.10,
            "deposit_released_during_build": False,
            "presales_pct": 0.60,
            "presales_achieved_month": 6,
            "ih_eligible": True,
        },
        {
            "name": "Retail (Sale)",
            "type": "commercial_sale",
            "phase": "Phase 1",
            "net_sqm": 600.0,
            "efficiency_ratio": 0.90,
            "sale_price_per_net_sqm": 52000.0,
            "build_cost_per_gba_sqm": 22000.0,
            "offplan_share": 0.10,
            "deposit_pct": 0.05,
            "deposit_released_during_build": False,
            "presales_pct": 0.0,
            "presales_achieved_month": 0,
            "ih_eligible": False,
        },
        {
            "name": "Office / Resi BTL (Yield)",
            "type": "rental_yield",
            "phase": "Phase 1",
            "net_sqm": 1200.0,
            "efficiency_ratio": 0.88,
            "build_cost_per_gba_sqm": 21000.0,

            # Rental assumptions
            "rent_per_net_sqm_month": 240.0,
            "rent_is_vat_exclusive": True,
            "opex_ratio": 0.35,
            "vacancy_stabilized": 0.05,
            "letting_up_months": 12,
            "hold_months_after_build": 36,
            "selling_cost_rate": 0.02,

            # Valuation
            "exit_cap_rate": 0.095,
            "exit_cap_applies_to": "noi_stabilized",

            "ih_eligible": False,
        }
    ]


@dataclass
class Assumptions:
    currency: str = "ZAR"

    phases: List[Dict[str, Any]] = field(default_factory=default_phases)
    products: List[Dict[str, Any]] = field(default_factory=default_products)

    # Overlays
    inclusionary_enabled: bool = True
    inclusionary_rate: float = 0.0
    inclusionary_price_per_net_sqm: float = 18000.0

    heritage_enabled: bool = False
    heritage_cost_uplift_rate: float = 0.00

    # Risk / soft costs
    contingency_rate: float = 0.07
    escalation_rate_pa: float = 0.07
    professional_fees_rate: float = 0.10
    statutory_costs: float = 350000.0
    marketing_rate: float = 0.02
    overhead_per_month: float = 25000.0

    # Land + friction
    land_price: float = 15_000_000.0
    land_treatment: str = "transfer_duty"
    legal_conveyancing: float = 150_000.0
    land_other_disbursements: float = 50_000.0

    # VAT
    vat: Dict[str, Any] = field(default_factory=lambda: asdict(VATConfig()))

    # Profit target
    target_profit_basis: str = "cost"
    target_profit_rate: float = 0.18

    # Construction finance + constraints
    use_debt: bool = True
    prime_rate_pa: float = 0.1025
    debt_margin_over_prime: float = 0.015
    debt_interest_only_during_build: bool = True

    max_ltc: float = 0.70
    max_ltv: float = 0.70
    min_equity_pct: float = 0.30

    presales_required_pct_resi: float = 0.60
    allow_debt_draw_before_presales: bool = False

    arrangement_fee_rate: float = 0.01
    exit_fee_rate: float = 0.005

    # ✅ Term debt / DSCR (post-stabilisation refinance)
    enable_term_debt: bool = True
    term_margin_over_prime: float = 0.010          # Prime + 1.0% typical “term”
    term_amort_years: int = 15                     # 10–20 yrs common
    term_dscr_min: float = 1.25                    # DSCR covenant
    term_max_ltv: float = 0.65                     # term LTV cap against value
    refinance_at_stabilisation: bool = True        # refi month = end of letting-up (default)
    refinance_month_offset: int = 0                # extra months after stabilisation
    allow_cash_out_refi: bool = False              # conservative default: no cash-out

    # Residual land
    solve_residual_land: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Assumptions":
        base = Assumptions()
        merged = deep_merge(base.to_dict(), d or {})
        return Assumptions(**merged)


@dataclass
class Output:
    currency: str

    gdv_net: float
    costs_net_ex_land_ex_fin: float
    land_net: float
    friction_net: float
    finance_costs: float
    total_cost_net: float
    profit_net: float
    profit_on_cost: float
    profit_on_gdv: float

    peak_debt: float
    equity_irr_pa: Optional[float]

    months: int
    vat_net_payable_total: float
    presales_gate_month: int

    # term debt stats
    refinance_month: int
    term_loan_amount: float
    term_dscr_at_refi: Optional[float]

    product_rows: List[Dict[str, Any]] = field(default_factory=list)
    cashflow_rows: List[Dict[str, Any]] = field(default_factory=list)
    audit: List[Dict[str, Any]] = field(default_factory=list)

    def headline(self) -> Dict[str, Any]:
        return {
            "Total revenue (net)": self.gdv_net,
            "Costs (net, ex land/fin)": self.costs_net_ex_land_ex_fin,
            "Land (net)": self.land_net,
            "Friction (net)": self.friction_net,
            "Finance": self.finance_costs,
            "Total cost (net)": self.total_cost_net,
            "Profit (net)": self.profit_net,
            "Profit % Cost": self.profit_on_cost,
            "Profit % Revenue": self.profit_on_gdv,
            "Peak debt": self.peak_debt,
            "Equity IRR p.a.": self.equity_irr_pa,
            "Refinance month": self.refinance_month,
            "Term loan amount": self.term_loan_amount,
            "Term DSCR @ refi": self.term_dscr_at_refi,
            "VAT net settlement total": self.vat_net_payable_total,
            "Presales gate month": self.presales_gate_month,
        }
