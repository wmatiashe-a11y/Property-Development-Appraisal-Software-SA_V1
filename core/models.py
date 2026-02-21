from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import copy


def deep_merge(a: dict, b: dict) -> dict:
    """Merge b into a (recursively) without mutating inputs."""
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def default_products() -> List[Dict[str, Any]]:
    """
    Product mix lines.
    - residential: units * avg_unit_size_sqm => sellable_sqm
    - commercial: sellable_sqm input directly
    """
    return [
        {
            "name": "Residential",
            "type": "residential",  # residential | commercial
            "units": 40,
            "avg_unit_size_sqm": 60.0,
            "sellable_sqm": None,  # computed for residential
            "price_per_sqm": 38000.0,  # ZAR/sqm
            "build_cost_per_sqm": 17500.0,  # ZAR/sqm (sellable proxy)
            "inclusionary_eligible": True,  # apply IH split if enabled
        },
        {
            "name": "Retail",
            "type": "commercial",
            "units": None,
            "avg_unit_size_sqm": None,
            "sellable_sqm": 600.0,
            "price_per_sqm": 52000.0,
            "build_cost_per_sqm": 22000.0,
            "inclusionary_eligible": False,
        },
    ]


@dataclass
class Assumptions:
    # --- GLOBAL ---
    currency: str = "ZAR"

    # --- PROGRAMME ---
    build_months: int = 18
    sales_months: int = 12
    sales_curve: str = "linear"  # linear | front | back

    # --- MIX ---
    products: List[Dict[str, Any]] = field(default_factory=default_products)

    # --- COST ADD-ONS / RATES ---
    contingency_rate: float = 0.07          # on build+uplifts+escalation
    escalation_rate_pa: float = 0.06        # build cost escalation
    heritage_cost_uplift_rate: float = 0.00 # uplift on build costs

    professional_fees_rate: float = 0.10    # % of build total (incl contingency/escalation/uplifts)
    statutory_costs: float = 350000.0       # lump sum (bulk services, approvals)
    marketing_rate: float = 0.02            # % of GDV
    overhead_per_month: float = 25000.0     # monthly overhead during build+sales

    # --- POLICY / OVERLAYS ---
    inclusionary_rate: float = 0.0          # % of eligible residential sellable sqm at capped price
    inclusionary_price_per_sqm: float = 18000.0
    incentive_grant: float = 0.0            # reduces costs (month 0)

    # --- PROFIT TARGET ---
    target_profit_basis: str = "gdv"        # gdv | cost
    target_profit_rate: float = 0.18

    # --- LAND ---
    land_price_input: Optional[float] = None  # if set, compute profit; else solve residual land

    # --- FINANCE (MONTHLY, REAL CASHFLOW) ---
    use_debt: bool = True
    debt_interest_rate_pa: float = 0.12
    debt_arrangement_fee_rate: float = 0.01  # % of peak debt (MVP proxy)
    debt_exit_fee_rate: float = 0.005        # % of peak debt
    equity_injection_month0: float = 0.0     # equity injected at month 0 (reduces debt)

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

    # headline
    gdv: float
    tdc_ex_land_ex_finance: float
    finance_costs: float
    land_value: float
    total_cost_incl_land_finance: float
    profit: float
    profit_rate_on_gdv: float
    profit_rate_on_cost: float
    peak_debt: float
    equity_irr_pa: Optional[float]

    # details
    months: int
    cashflow_rows: List[Dict[str, Any]] = field(default_factory=list)
    audit: List[Dict[str, Any]] = field(default_factory=list)
    product_rows: List[Dict[str, Any]] = field(default_factory=list)

    def headline_dict(self) -> Dict[str, Any]:
        return {
            "GDV": self.gdv,
            "Costs (ex land, ex finance)": self.tdc_ex_land_ex_finance,
            "Finance costs": self.finance_costs,
            "Land": self.land_value,
            "Total cost (incl land+finance)": self.total_cost_incl_land_finance,
            "Profit": self.profit,
            "Profit % GDV": self.profit_rate_on_gdv,
            "Profit % Cost": self.profit_rate_on_cost,
            "Peak debt": self.peak_debt,
            "Equity IRR p.a.": self.equity_irr_pa,
        }
