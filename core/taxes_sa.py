from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class VATConfig:
    enabled: bool = True
    vat_rate: float = 0.15  # SARS standard rate 15%
    prices_include_vat: bool = True   # selling prices typically quoted VAT-inclusive
    costs_include_vat: bool = False   # builder bills often quoted excl VAT
    input_vat_recoverable: bool = True
    settlement_lag_months: int = 1    # cash paid/received one month later (MVP)


def split_vat_from_gross(gross: float, vat_rate: float) -> Tuple[float, float]:
    """
    Given a VAT-inclusive gross amount, return (net, vat).
    """
    if gross <= 0:
        return 0.0, 0.0
    net = gross / (1.0 + vat_rate)
    vat = gross - net
    return net, vat


def add_vat_to_net(net: float, vat_rate: float) -> Tuple[float, float]:
    """
    Given a VAT-exclusive net amount, return (gross, vat).
    """
    if net <= 0:
        return 0.0, 0.0
    vat = net * vat_rate
    gross = net + vat
    return gross, vat


def transfer_duty_za(property_value: float) -> float:
    """
    South Africa transfer duty calculation using SARS brackets
    effective 1 April 2025 (as per SARS notice/table).
    If SARS updates later, update these thresholds/rates.

    Brackets (R):
      0 – 1,210,000: 0%
      1,210,001 – 1,663,800: 3% above 1,210,000
      1,663,801 – 2,329,300: 13,614 + 6% above 1,663,800
      2,329,301 – 2,994,800: 53,544 + 8% above 2,329,300
      2,994,801 – 13,310,000: 106,784 + 11% above 2,994,800
      13,310,001+: 1,241,456 + 13% above 13,310,000
    """
    v = max(0.0, float(property_value))
    if v <= 1_210_000:
        return 0.0
    if v <= 1_663_800:
        return 0.03 * (v - 1_210_000)
    if v <= 2_329_300:
        return 13_614 + 0.06 * (v - 1_663_800)
    if v <= 2_994_800:
        return 53_544 + 0.08 * (v - 2_329_300)
    if v <= 13_310_000:
        return 106_784 + 0.11 * (v - 2_994_800)
    return 1_241_456 + 0.13 * (v - 13_310_000)
