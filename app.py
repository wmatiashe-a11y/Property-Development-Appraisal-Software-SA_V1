from __future__ import annotations

import io
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

from core.models import Assumptions
from core.engine import run_appraisal, sensitivity_grid
from data.db import init_db, list_projects, create_project, list_appraisals, save_appraisal, load_appraisal


def money(x: float, ccy: str) -> str:
    try:
        return f"{ccy} {x:,.0f}"
    except Exception:
        return f"{ccy} {x}"

def pct(x: float) -> str:
    return f"{x*100:.1f}%"

def fmt_audit(v, unit: str, ccy: str) -> str:
    if unit == "ratio":
        return pct(float(v))
    if unit == ccy:
        return money(float(v), ccy)
    return str(v)

def kpis(out):
    ccy = out.currency
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Profit (net)", money(out.profit_net, ccy), pct(out.profit_on_cost))
    a2.metric("Total revenue (net)", money(out.gdv_net, ccy))
    a3.metric("Peak debt", money(out.peak_debt, ccy))
    a4.metric("Equity IRR p.a.", "-" if out.equity_irr_pa is None else pct(out.equity_irr_pa))

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Costs (net ex land/fin)", money(out.costs_net_ex_land_ex_fin, ccy))
    b2.metric("Land (net)", money(out.land_net, ccy))
    b3.metric("Friction (net)", money(out.friction_net, ccy))
    b4.metric("Finance", money(out.finance_costs, ccy))

def build_pdf(project_name: str, project_location: str, a: Assumptions, out) -> bytes:
    ccy = out.currency
    styles = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16*mm, rightMargin=16*mm, topMargin=14*mm, bottomMargin=14*mm)

    story = []
    story.append(Paragraph(f"Feasibility Report — {project_name}", styles["Title"]))
    story.append(Paragraph(f"{project_location or ''} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 10))

    summary = [
        ["Metric", "Value"],
        ["Total revenue (net)", money(out.gdv_net, ccy)],
        ["Costs (net ex land/fin)", money(out.costs_net_ex_land_ex_fin, ccy)],
        ["Land (net)", money(out.land_net, ccy)],
        ["Friction (net)", money(out.friction_net, ccy)],
        ["Finance", money(out.finance_costs, ccy)],
        ["Profit (net)", f"{money(out.profit_net, ccy)}  ({pct(out.profit_on_cost)} on cost)"],
        ["Equity IRR p.a.", "-" if out.equity_irr_pa is None else pct(out.equity_irr_pa)],
        ["Peak debt", money(out.peak_debt, ccy)],
        ["VAT net settlement total", money(out.vat_net_payable_total, ccy)],
        ["Presales gate month", str(out.presales_gate_month)],
    ]
    t = Table(summary, colWidths=[70*mm, 100*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#D1D5DB")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(Paragraph("Summary", styles["Heading2"]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Products
    dfp = pd.DataFrame(out.product_rows)
    if not dfp.empty:
        cols = list(dfp.columns)
        table = [cols]
        for _, r in dfp.iterrows():
            row = []
            for c in cols:
                v = r[c]
                if c.lower().endswith("(net)") or c.lower().endswith("(gross)") or "value" in c.lower() or "gdv" in c.lower() or "build" in c.lower() or "noi" in c.lower():
                    try:
                        row.append(money(float(v), ccy))
                    except Exception:
                        row.append(str(v))
                elif "sqm" in c.lower():
                    try:
                        row.append(f"{float(v):,.0f}")
                    except Exception:
                        row.append(str(v))
                elif "cap" in c.lower() or "ratio" in c.lower() or "vacancy" in c.lower() or "opex" in c.lower() or "share" in c.lower():
                    try:
                        row.append(pct(float(v)))
                    except Exception:
                        row.append(str(v))
                else:
                    row.append(str(v))
            table.append(row)
        pt = Table(table, repeatRows=1)
        pt.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#111827")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#D1D5DB")),
            ("FONTSIZE", (0,0), (-1,-1), 7),
            ("PADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(Paragraph("Product mix", styles["Heading2"]))
        story.append(pt)
        story.append(Spacer(1, 12))

    # Audit
    story.append(Paragraph("Audit trail (key lines)", styles["Heading2"]))
    dfa = pd.DataFrame(out.audit).head(60)
    table = [["Section", "Key", "Value"]]
    for _, r in dfa.iterrows():
        table.append([str(r["section"]), str(r["key"]), fmt_audit(r["value"], str(r.get("unit") or ""), ccy)])
    au = Table(table, repeatRows=1, colWidths=[25*mm, 80*mm, 65*mm])
    au.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#D1D5DB")),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("PADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(au)

    doc.build(story)
    return buf.getvalue()


# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="SA Feasibility (Sale + BTL)", layout="wide")
init_db()

st.title("SA-Style Development Feasibility — Sale + BTL")
st.caption("Sale products + Rental/Yield products • IH overlay • Heritage overlay • VAT • Bank constraints")

with st.sidebar:
    st.header("Projects")
    projects = list_projects()

    with st.expander("➕ New project", expanded=False):
        new_name = st.text_input("Project name", value="")
        new_loc = st.text_input("Location", value="")
        if st.button("Create project", use_container_width=True) and new_name.strip():
            create_project(new_name.strip(), new_loc.strip())
            st.rerun()

    if not projects:
        st.info("Create a project to begin.")
        st.stop()

    labels = [f"#{p['id']} — {p['name']}" for p in projects]
    i = st.selectbox("Select project", list(range(len(labels))), format_func=lambda k: labels[k])
    project = projects[i]
    project_id = int(project["id"])

    st.divider()
    st.subheader("Saved appraisals")
    apps = list_appraisals(project_id)
    appraisal_id = None
    if apps:
        app_labels = [f"#{x['id']} — {x['version_name']} ({x['created_at']})" for x in apps]
        pick = st.selectbox("Load saved version", ["(none)"] + app_labels)
        if pick != "(none)":
            appraisal_id = int(pick.split("—")[0].strip().replace("#", ""))

if "assumptions_dict" not in st.session_state:
    st.session_state.assumptions_dict = Assumptions().to_dict()

if appraisal_id:
    loaded = load_appraisal(appraisal_id)
    if loaded:
        st.session_state.assumptions_dict = loaded[0]

a = Assumptions.from_dict(st.session_state.assumptions_dict)

left, right = st.columns([0.56, 0.44], gap="large")

with left:
    st.subheader("Inputs")

    # --------- PHASES ----------
    st.markdown("### Phasing")
    phases = list(a.phases or [])
    for idx, ph in enumerate(phases):
        with st.expander(f"{idx+1}) {ph.get('name','Phase')}", expanded=(idx == 0)):
            c1, c2, c3, c4 = st.columns(4)
            ph["name"] = c1.text_input("Name", value=str(ph.get("name","Phase")), key=f"ph_name_{idx}")
            ph["start_month"] = int(c2.number_input("Start month (0=now)", 0, 240, int(ph.get("start_month", 0)), key=f"ph_start_{idx}"))
            ph["build_months"] = int(c3.number_input("Build months", 1, 120, int(ph.get("build_months", 18)), key=f"ph_build_{idx}"))
            ph["sales_months"] = int(c4.number_input("Sales months (sale products)", 1, 120, int(ph.get("sales_months", 12)), key=f"ph_sales_{idx}"))
            ph["sales_curve"] = st.selectbox("Sales curve", ["linear","front","back"], index=["linear","front","back"].index(str(ph.get("sales_curve","linear"))), key=f"ph_curve_{idx}")

            d1, d2 = st.columns(2)
            if d1.button("Duplicate phase", key=f"ph_dup_{idx}"):
                phases.insert(idx+1, dict(ph))
                st.session_state.assumptions_dict["phases"] = phases
                st.rerun()
            if d2.button("Delete phase", key=f"ph_del_{idx}") and len(phases) > 1:
                phases.pop(idx)
                st.session_state.assumptions_dict["phases"] = phases
                st.rerun()

    if st.button("➕ Add phase", use_container_width=True):
        phases.append({"name": f"Phase {len(phases)+1}", "start_month": 0, "build_months": 18, "sales_months": 12, "sales_curve": "linear"})
        st.session_state.assumptions_dict["phases"] = phases
        st.rerun()
    a.phases = phases
    phase_names = [str(p.get("name")) for p in a.phases]

    # --------- PRODUCT MIX ----------
    st.markdown("### Product mix")
    prods = list(a.products or [])
    for i, p in enumerate(prods):
        with st.expander(f"{i+1}) {p.get('name','Product')}", expanded=(i == 0)):
            r1, r2, r3 = st.columns([0.45, 0.30, 0.25])
            p["name"] = r1.text_input("Name", value=str(p.get("name","Product")), key=f"p_name_{i}")
            p["type"] = r2.selectbox("Type", ["residential_sale","commercial_sale","rental_yield"], index=["residential_sale","commercial_sale","rental_yield"].index(str(p.get("type","residential_sale"))), key=f"p_type_{i}")
            p["phase"] = r3.selectbox("Phase", phase_names, index=phase_names.index(p.get("phase", phase_names[0])) if p.get("phase") in phase_names else 0, key=f"p_phase_{i}")

            p["efficiency_ratio"] = st.slider("Efficiency (Net/GBA)", 0.70, 0.95, float(p.get("efficiency_ratio", 0.83)), 0.01, key=f"p_eff_{i}")

            is_res = (p["type"] == "residential_sale")
            if is_res:
                c1, c2, c3 = st.columns(3)
                p["units"] = int(c1.number_input("Units", 0, 5000, int(p.get("units", 0)), 1, key=f"p_units_{i}"))
                p["avg_unit_net_sqm"] = float(c2.number_input("Avg unit NSA (m²)", 0.0, 500.0, float(p.get("avg_unit_net_sqm", 0.0)), 1.0, key=f"p_avg_{i}"))
                net = p["units"] * p["avg_unit_net_sqm"]
                c3.metric("Computed Net sqm", f"{net:,.0f}")
                p["net_sqm"] = None
            else:
                p["net_sqm"] = float(st.number_input("Net area (m²)", 0.0, 1e9, float(p.get("net_sqm", 0.0)), 10.0, key=f"p_net_{i}"))
                p["units"] = int(p.get("units") or 0)
                p["avg_unit_net_sqm"] = float(p.get("avg_unit_net_sqm") or 0.0)

            p["build_cost_per_gba_sqm"] = float(st.number_input("Build cost per GBA sqm", 0.0, 1e9, float(p.get("build_cost_per_gba_sqm", 0.0)), 500.0, key=f"p_cost_{i}"))

            if p["type"] in ("residential_sale", "commercial_sale"):
                st.markdown("**Sale assumptions**")
                p["sale_price_per_net_sqm"] = float(st.number_input("Sale price per NET sqm", 0.0, 1e9, float(p.get("sale_price_per_net_sqm", 0.0)), 500.0, key=f"p_price_{i}"))

                st.markdown("**Bankability / sales streams**")
                s1, s2, s3, s4 = st.columns(4)
                p["offplan_share"] = s1.slider("Off-plan share", 0.0, 1.0, float(p.get("offplan_share", 0.0)), 0.05, key=f"p_off_{i}")
                p["deposit_pct"] = s2.slider("Deposit %", 0.0, 0.30, float(p.get("deposit_pct", 0.0)), 0.01, key=f"p_dep_{i}")
                p["deposit_released_during_build"] = s3.toggle("Deposit released during build", value=bool(p.get("deposit_released_during_build", False)), key=f"p_dep_rel_{i}")
                p["presales_pct"] = s4.slider("Bankable presales %", 0.0, 1.0, float(p.get("presales_pct", 0.0)), 0.05, key=f"p_pre_{i}")
                p["presales_achieved_month"] = int(st.number_input("Presales achieved month", 0, 240, int(p.get("presales_achieved_month", 0)), 1, key=f"p_pre_m_{i}"))

                p["ih_eligible"] = st.toggle("IH eligible (resi sale only)", value=bool(p.get("ih_eligible", p["type"]=="residential_sale")), key=f"p_ih_{i}")

            else:
                st.markdown("**Rental / yield valuation (BTL / commercial hold)**")
                z1, z2, z3 = st.columns(3)
                p["rent_per_net_sqm_month"] = float(z1.number_input("Rent per net sqm / month", 0.0, 1e9, float(p.get("rent_per_net_sqm_month", 0.0)), 5.0, key=f"p_rent_{i}"))
                p["rent_is_vat_exclusive"] = z2.toggle("Rent + VAT (exclusive)", value=bool(p.get("rent_is_vat_exclusive", True)), key=f"p_rent_vat_{i}")
                p["opex_ratio"] = z3.slider("Opex ratio (% of rent)", 0.0, 0.80, float(p.get("opex_ratio", 0.35)), 0.01, key=f"p_opex_{i}")

                y1, y2, y3, y4 = st.columns(4)
                p["vacancy_stabilized"] = y1.slider("Vacancy (stabilised)", 0.0, 0.30, float(p.get("vacancy_stabilized", 0.05)), 0.01, key=f"p_vac_{i}")
                p["letting_up_months"] = int(y2.number_input("Letting-up months", 0, 60, int(p.get("letting_up_months", 12)), 1, key=f"p_let_{i}"))
                p["hold_months_after_build"] = int(y3.number_input("Hold months after build", 0, 240, int(p.get("hold_months_after_build", 36)), 1, key=f"p_hold_{i}"))
                p["exit_cap_rate"] = float(y4.slider("Exit cap rate", 0.03, 0.25, float(p.get("exit_cap_rate", 0.095)), 0.0025, key=f"p_cap_{i}"))

                p["selling_cost_rate"] = float(st.slider("Selling costs (% of exit value)", 0.0, 0.08, float(p.get("selling_cost_rate", 0.02)), 0.0025, key=f"p_sell_{i}"))
                p["ih_eligible"] = False

            d1, d2 = st.columns(2)
            if d1.button("Duplicate line", key=f"p_dup_{i}"):
                prods.insert(i+1, dict(p))
                st.session_state.assumptions_dict["products"] = prods
                st.rerun()
            if d2.button("Delete line", key=f"p_del_{i}") and len(prods) > 1:
                prods.pop(i)
                st.session_state.assumptions_dict["products"] = prods
                st.rerun()

    if st.button("➕ Add product line", use_container_width=True):
        prods.append(dict(Assumptions().products[0]))
        st.session_state.assumptions_dict["products"] = prods
        st.rerun()
    a.products = prods

    # --------- OVERLAYS TABS ----------
    st.markdown("### Overlays")
    t_core, t_ih, t_herit = st.tabs(["Overview", "Inclusionary Housing", "Heritage"])

    with t_core:
        st.caption("Use the tabs to apply overlay assumptions that affect revenue (IH) and costs (Heritage).")

    with t_ih:
        a.inclusionary_enabled = st.toggle("Enable Inclusionary Housing overlay", value=bool(a.inclusionary_enabled))
        a.inclusionary_rate = st.slider("IH % of eligible resi sale net area", 0.0, 0.30, float(a.inclusionary_rate), 0.01)
        a.inclusionary_price_per_net_sqm = float(st.number_input("IH capped price per net sqm", 0.0, 1e9, float(a.inclusionary_price_per_net_sqm), 500.0))
        st.caption("Applied only to product lines: **Type = residential_sale** with **IH eligible = True**.")

    with t_herit:
        a.heritage_enabled = st.toggle("Enable Heritage overlay", value=bool(a.heritage_enabled))
        a.heritage_cost_uplift_rate = st.slider("Heritage uplift on build cost base", 0.0, 0.25, float(a.heritage_cost_uplift_rate), 0.01)

    # --------- RISK / SOFT COSTS ----------
    st.markdown("### Risk & soft costs")
    c1, c2, c3 = st.columns(3)
    a.contingency_rate = c1.slider("Contingency", 0.0, 0.20, float(a.contingency_rate), 0.01)
    a.escalation_rate_pa = c2.slider("Escalation p.a.", 0.0, 0.20, float(a.escalation_rate_pa), 0.005)
    a.professional_fees_rate = c3.slider("Professional fees", 0.0, 0.25, float(a.professional_fees_rate), 0.01)

    d1, d2, d3 = st.columns(3)
    a.statutory_costs = float(d1.number_input("Statutory (lump sum)", 0.0, 1e12, float(a.statutory_costs), 50_000.0))
    a.marketing_rate = d2.slider("Marketing (% of sale GDV net)", 0.0, 0.10, float(a.marketing_rate), 0.005)
    a.overhead_per_month = float(d3.number_input("Overhead / month", 0.0, 1e12, float(a.overhead_per_month), 5_000.0))

    # --------- LAND ----------
    st.markdown("### Land + friction")
    e1, e2, e3 = st.columns(3)
    a.land_price = float(e1.number_input("Land price (as entered)", 0.0, 1e12, float(a.land_price), 250_000.0))
    a.land_treatment = e2.selectbox("Land treatment", ["transfer_duty", "vat_standard", "vat_zero"], index=["transfer_duty","vat_standard","vat_zero"].index(a.land_treatment))
    a.solve_residual_land = e3.toggle("Solve residual land to hit target", value=bool(a.solve_residual_land))

    f1, f2 = st.columns(2)
    a.legal_conveyancing = float(f1.number_input("Legal / conveyancing", 0.0, 1e12, float(a.legal_conveyancing), 10_000.0))
    a.land_other_disbursements = float(f2.number_input("Other land disbursements", 0.0, 1e12, float(a.land_other_disbursements), 10_000.0))

    # --------- VAT ----------
    st.markdown("### VAT")
    vat = dict(a.vat or {})
    v1, v2, v3, v4, v5 = st.columns(5)
    vat["enabled"] = v1.toggle("VAT enabled", value=bool(vat.get("enabled", True)))
    vat["vat_rate"] = v2.slider("VAT rate", 0.10, 0.20, float(vat.get("vat_rate", 0.15)), 0.005)
    vat["prices_include_vat"] = v3.toggle("Sale prices include VAT", value=bool(vat.get("prices_include_vat", True)))
    vat["costs_include_vat"] = v4.toggle("Costs include VAT", value=bool(vat.get("costs_include_vat", False)))
    vat["input_vat_recoverable"] = v5.toggle("Input VAT recoverable", value=bool(vat.get("input_vat_recoverable", True)))
    vat["settlement_lag_months"] = int(st.slider("VAT settlement lag (months)", 0, 3, int(vat.get("settlement_lag_months", 1)), 1))
    a.vat = vat

    # --------- PROFIT TARGET ----------
    st.markdown("### Profit target")
    p1, p2 = st.columns(2)
    a.target_profit_basis = p1.selectbox("Target basis", ["cost", "gdv"], index=0 if a.target_profit_basis == "cost" else 1)
    a.target_profit_rate = p2.slider("Target profit rate", 0.0, 0.35, float(a.target_profit_rate), 0.01)

    # --------- FINANCE ----------
    st.markdown("### Finance constraints (bank-style)")
    q1, q2, q3, q4 = st.columns(4)
    a.use_debt = q1.toggle("Use debt", value=bool(a.use_debt))
    a.prime_rate_pa = q2.slider("Prime p.a.", 0.05, 0.20, float(a.prime_rate_pa), 0.0025)
    a.debt_margin_over_prime = q3.slider("Margin over prime", 0.0, 0.05, float(a.debt_margin_over_prime), 0.0025)
    a.debt_interest_only_during_build = q4.toggle("Interest-only during build", value=bool(a.debt_interest_only_during_build))

    r1, r2, r3, r4 = st.columns(4)
    a.max_ltc = r1.slider("Max LTC", 0.30, 0.90, float(a.max_ltc), 0.01)
    a.max_ltv = r2.slider("Max LTV", 0.30, 0.90, float(a.max_ltv), 0.01)
    a.min_equity_pct = r3.slider("Min equity", 0.10, 0.60, float(a.min_equity_pct), 0.01)
    a.presales_required_pct_resi = r4.slider("Presales required (resi)", 0.0, 0.90, float(a.presales_required_pct_resi), 0.05)

    s1, s2, s3 = st.columns(3)
    a.allow_debt_draw_before_presales = s1.toggle("Allow debt draw before presales", value=bool(a.allow_debt_draw_before_presales))
    a.arrangement_fee_rate = s2.slider("Arrangement fee (% peak debt)", 0.0, 0.05, float(a.arrangement_fee_rate), 0.001)
    a.exit_fee_rate = s3.slider("Exit fee (% peak debt)", 0.0, 0.05, float(a.exit_fee_rate), 0.001)

    st.session_state.assumptions_dict = a.to_dict()

with right:
    st.subheader("Outputs")
    out = run_appraisal(a)
    kpis(out)

    st.markdown("### Product breakdown")
    dfp = pd.DataFrame(out.product_rows)
    if not dfp.empty:
        ccy = out.currency
        show = dfp.copy()
        # format common money columns if present
        money_cols = [c for c in show.columns if any(k in c.lower() for k in ["gdv", "build", "noi", "value", "rent /"])]
        for col in money_cols:
            try:
                show[col] = show[col].astype(float).map(lambda v: money(v, ccy))
            except Exception:
                pass
        # sqm
        for col in [c for c in show.columns if "sqm" in c.lower()]:
            try:
                show[col] = show[col].astype(float).map(lambda v: f"{v:,.0f}")
            except Exception:
                pass
        # ratios
        for col in [c for c in show.columns if any(k in c.lower() for k in ["efficiency","share","vacancy","opex","cap"])]:
            try:
                show[col] = show[col].astype(float).map(pct)
            except Exception:
                pass
        st.dataframe(show, use_container_width=True, hide_index=True)

    tabs = st.tabs(["Cashflow", "Sensitivity", "Scenarios & Compare", "Audit", "PDF"])

    with tabs[0]:
        df = pd.DataFrame(out.cashflow_rows)
        st.caption("Cashflow includes sale receipts, rental receipts, exit value, VAT settlement timing, and constrained debt draws.")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["Month"], y=df["Revenue (gross)"], name="Revenue (gross)"))
        fig.add_trace(go.Bar(x=df["Month"], y=-df["Costs (gross incl VAT)"], name="-Costs (gross)"))
        fig.add_trace(go.Bar(x=df["Month"], y=-df["VAT settlement (+pay / -refund)"], name="-VAT settlement"))
        fig.update_layout(barmode="relative", height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["Month"], y=df["Debt balance"], name="Debt balance", mode="lines"))
        fig2.add_trace(go.Bar(x=df["Month"], y=df["Debt draw"], name="Debt draw"))
        fig2.add_trace(go.Bar(x=df["Month"], y=-df["Debt repay"], name="-Debt repay"))
        fig2.add_trace(go.Scatter(x=df["Month"], y=df["Equity inject (auto)"], name="Equity inject (auto)", mode="lines+markers"))
        fig2.update_layout(barmode="relative", height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(df, use_container_width=True, hide_index=True)

    with tabs[1]:
        rows, cols, mat = sensitivity_grid(a)
        heat = go.Figure(data=go.Heatmap(
            z=mat,
            x=cols,
            y=rows,
            hovertemplate="Price/Rent: %{y}<br>Cost: %{x}<br>Value: %{z:,.0f}<extra></extra>"
        ))
        heat.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(heat, use_container_width=True)

        metric = "Residual land (net)" if a.solve_residual_land else "Profit (net)"
        st.caption(f"Sensitivity metric: **{metric}** (price axis uplifts sale prices and rents; cost axis uplifts build costs).")
        st.dataframe(pd.DataFrame(mat, index=rows, columns=cols).applymap(lambda v: money(float(v), out.currency)), use_container_width=True)

    with tabs[2]:
        st.caption("Save Base / Offer / Bank cases, then compare side-by-side.")
        c1, c2, c3 = st.columns(3)
        if c1.button("💾 Save as Base", use_container_width=True):
            save_appraisal(project_id, "Base", a.to_dict(), out.headline()); st.rerun()
        if c2.button("💾 Save as Offer", use_container_width=True):
            save_appraisal(project_id, "Offer", a.to_dict(), out.headline()); st.rerun()
        if c3.button("💾 Save as Bank", use_container_width=True):
            save_appraisal(project_id, "Bank", a.to_dict(), out.headline()); st.rerun()

        st.divider()
        apps = list_appraisals(project_id)
        if not apps:
            st.info("No saved appraisals yet.")
        else:
            labels = {x["id"]: f"#{x['id']} — {x['version_name']} ({x['created_at']})" for x in apps}
            ids = list(labels.keys())
            a1, a2, a3 = st.columns(3)
            pickA = a1.selectbox("Compare A", ids, format_func=lambda k: labels[k])
            pickB = a2.selectbox("Compare B", ids, format_func=lambda k: labels[k], index=min(1, len(ids)-1))
            pickC = a3.selectbox("Compare C", ids, format_func=lambda k: labels[k], index=min(2, len(ids)-1))

            def load_recalc(app_id):
                loaded = load_appraisal(int(app_id))
                if not loaded:
                    return None
                aa = Assumptions.from_dict(loaded[0])
                oo = run_appraisal(aa)
                return labels[app_id], oo

            rows_ = []
            for slot, pid in [("A", pickA), ("B", pickB), ("C", pickC)]:
                res = load_recalc(pid)
                if res:
                    label, oo = res
                    rows_.append({
                        "Slot": slot,
                        "Version": label,
                        "Revenue (net)": oo.gdv_net,
                        "Profit (net)": oo.profit_net,
                        "Profit % Cost": oo.profit_on_cost,
                        "Peak debt": oo.peak_debt,
                        "Equity IRR p.a.": oo.equity_irr_pa,
                        "VAT total": oo.vat_net_payable_total,
                        "Presales gate month": oo.presales_gate_month,
                    })

            dfc = pd.DataFrame(rows_)
            if not dfc.empty:
                ccy = out.currency
                show = dfc.copy()
                for col in ["Revenue (net)", "Profit (net)", "Peak debt", "VAT total"]:
                    show[col] = show[col].astype(float).map(lambda v: money(v, ccy))
                show["Profit % Cost"] = show["Profit % Cost"].astype(float).map(pct)
                show["Equity IRR p.a."] = show["Equity IRR p.a."].apply(lambda v: "-" if v is None else pct(float(v)))
                st.dataframe(show, use_container_width=True, hide_index=True)

    with tabs[3]:
        ccy = out.currency
        dfa = pd.DataFrame(out.audit)
        dfa["Display"] = dfa.apply(lambda r: fmt_audit(r["value"], str(r.get("unit") or ""), ccy), axis=1)
        st.dataframe(dfa[["section","key","Display"]], use_container_width=True, hide_index=True)

    with tabs[4]:
        pdf = build_pdf(project["name"], project.get("location") or "", a, out)
        st.download_button(
            "📄 Download PDF report",
            data=pdf,
            file_name=f"feasibility_{project['name'].replace(' ','_')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
