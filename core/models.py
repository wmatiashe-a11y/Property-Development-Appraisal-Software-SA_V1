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
    # SA typical: one phase; you can add more in UI
    return [
        {"name": "Phase 1", "start_month": 0, "build_months": 18, "sales_months": 12, "sales_curve": "linear"}
    ]


def default_products() -> List[Dict[str, Any]]:
    """
    Key SA convention baked in:
    - Revenue is on Net area (NSA/NLA)
    - Build costs are on GBA
    - Efficiency ratio converts Net -> GBA: GBA = Net / efficiency
    """
    return [
        {
            "name": "Residential",
            "type": "residential_sale",          # residential_sale | commercial_sale (MVP)
            "phase": "Phase 1",

            # Net area (NSA)
            "units": 60,
            "avg_unit_net_sqm": 55.0,           # NSA per unit
            "net_sqm": None,                    # computed for residential
            "efficiency_ratio": 0.83,           # NSA/GBA typical 0.80–0.85

            # Prices: selling price per NET sqm (VAT treatment via vat config)
            "sale_price_per_net_sqm": 38000.0,  # ZAR per NSA sqm (typically VAT-inclusive in SA new-build sales)

            # Costs: build cost per GBA sqm
            "build_cost_per_gba_sqm": 17500.0,  # ZAR per GBA sqm

            # Sales streams
            "offplan_share": 0.60,              # % of GDV sold during build (contracts), cash collected at transfer unless deposit released
            "deposit_pct": 0.10,                # deposit % of contract value
            "deposit_released_during_build": False,  # typically held in trust until transfer
            "presales_pct": 0.60,               # bankable presales % of this line's GDV
            "presales_achieved_month": 6,       # month in which presales are achieved (contracts signed)
        },
        {
            "name": "Retail",
            "type": "commercial_sale",
            "phase": "Phase 1",

            "net_sqm": 600.0,                   # NLA
            "efficiency_ratio": 0.90,           # often higher for commercial
            "sale_price_per_net_sqm": 52000.0,
            "build_cost_per_gba_sqm": 22000.0,

            "offplan_share": 0.10,
            "deposit_pct": 0.05,
            "deposit_released_during_build": False,
            "presales_pct": 0.0,
            "presales_achieved_month": 0,
        },
    ]


@dataclass
class Assumptions:
    currency: str = "ZAR"

    phases: List[Dict[str, Any]] = field(default_factory=default_phases)
    products: List[Dict[str, Any]] = field(default_factory=default_products)

    # Cost add-ons / risk
    contingency_rate: float = 0.07
    escalation_rate_pa: float = 0.07
    heritage_cost_uplift_rate: float = 0.00

    professional_fees_rate: float = 0.10
    statutory_costs: float = 350000.0
    marketing_rate: float = 0.02
    overhead_per_month: float = 25000.0

    # Land acquisition & friction
    land_price: float = 15_000_000.0  # offer price (gross as entered)
    land_treatment: str = "transfer_duty"  # transfer_duty | vat_standard | vat_zero
    legal_conveyancing: float = 150_000.0
    land_other_disbursements: float = 50_000.0

    # VAT
    vat: Dict[str, Any] = field(default_factory=lambda: asdict(VATConfig()))

    # Profit target norms (you can choose basis)
    target_profit_basis: str = "cost"  # gdv | cost
    target_profit_rate: float = 0.18

    # Finance + constraints
    use_debt: bool = True
    prime_rate_pa: float = 0.1025  # SARB prime ~10.25% (Feb 2026)
    debt_margin_over_prime: float = 0.015  # Prime + 1.5% typical
    debt_interest_only_during_build: bool = True

    max_ltc: float = 0.70  # max loan-to-cost cap
    max_ltv: float = 0.70  # max loan-to-value cap (against GDV net)
    min_equity_pct: float = 0.30  # bank requires 30–40% equity

    presales_required_pct_resi: float = 0.60  # 50–70% typical before first draw
    allow_debt_draw_before_presales: bool = False

    arrangement_fee_rate: float = 0.01  # % of peak debt
    exit_fee_rate: float = 0.005

    # Residual land mode (optional)
    solve_residual_land: bool = False   # if True, land is solved to hit target profit; ignores land_price + friction

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

    # Econ totals (net-of-VAT if recoverable; otherwise gross)
    gdv_net: float
    costs_net_ex_land_ex_fin: float
    land_net: float
    friction_net: float  # transfer duty + legal etc
    finance_costs: float
    total_cost_net: float
    profit_net: float
    profit_on_cost: float
    profit_on_gdv: float

    # Finance + IRR
    peak_debt: float
    equity_irr_pa: Optional[float]

    # Diagnostics
    months: int
    vat_net_payable_total: float
    presales_gate_month: int

    # Tables
    product_rows: List[Dict[str, Any]] = field(default_factory=list)
    cashflow_rows: List[Dict[str, Any]] = field(default_factory=list)
    audit: List[Dict[str, Any]] = field(default_factory=list)

    def headline(self) -> Dict[str, Any]:
        return {
            "GDV (net)": self.gdv_net,
            "Costs (net, ex land/fin)": self.costs_net_ex_land_ex_fin,
            "Land (net)": self.land_net,
            "Friction (net)": self.friction_net,
            "Finance": self.finance_costs,
            "Total cost (net)": self.total_cost_net,
            "Profit (net)": self.profit_net,
            "Profit % Cost": self.profit_on_cost,
            "Profit % GDV": self.profit_on_gdv,
            "Peak debt": self.peak_debt,
            "Equity IRR p.a.": self.equity_irr_pa,
            "VAT net payable (total)": self.vat_net_payable_total,
            "Presales gate month": self.presales_gate_month,
        }
