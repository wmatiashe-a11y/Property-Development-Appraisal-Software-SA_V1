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


# ----------------------------
# Formatting helpers
# ----------------------------
def money(x: float, ccy: str) -> str:
    try:
        return f"{ccy} {x:,.0f}"
    except Exception:
        return f"{ccy} {x}"

def pct(x: float) -> str:
    return f"{x*100:.1f}%"

def ratio_or_text(v):
    if isinstance(v, (int, float)):
        return v
    return str(v)

def fmt_audit_value(v, unit: str, ccy: str) -> str:
    if unit == "ratio":
        return pct(float(v))
    if unit == ccy:
        return money(float(v), ccy)
    return str(v)

def kpi_cards(out):
    ccy = out.currency
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Residual Land Value", money(out.land_value, ccy))
    c2.metric("Profit", money(out.profit, ccy), pct(out.profit_rate_on_gdv))
    c3.metric("GDV", money(out.gdv, ccy))
    c4.metric("Peak Debt", money(out.peak_debt, ccy))

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Costs (ex land, ex finance)", money(out.tdc_ex_land_ex_finance, ccy))
    d2.metric("Finance costs", money(out.finance_costs, ccy))
    d3.metric("Total cost (incl land+finance)", money(out.total_cost_incl_land_finance, ccy))
    d4.metric("Equity IRR p.a.", "-" if out.equity_irr_pa is None else pct(out.equity_irr_pa))


# ----------------------------
# PDF report
# ----------------------------
def build_pdf(project_name: str, project_location: str, a: Assumptions, out) -> bytes:
    ccy = out.currency
    styles = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16*mm, rightMargin=16*mm, topMargin=14*mm, bottomMargin=14*mm)

    story = []
    title = f"Feasibility Report — {project_name}"
    story.append(Paragraph(title, styles["Title"]))
    meta = f"{project_location or ''} &nbsp;&nbsp;|&nbsp;&nbsp; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    story.append(Paragraph(meta, styles["Normal"]))
    story.append(Spacer(1, 10))

    # Summary KPIs
    summary_data = [
        ["Metric", "Value"],
        ["GDV", money(out.gdv, ccy)],
        ["Costs (ex land, ex finance)", money(out.tdc_ex_land_ex_finance, ccy)],
        ["Finance costs", money(out.finance_costs, ccy)],
        ["Residual Land Value", money(out.land_value, ccy)],
        ["Profit", f"{money(out.profit, ccy)}  ({pct(out.profit_rate_on_gdv)} of GDV)"],
        ["Equity IRR p.a.", "-" if out.equity_irr_pa is None else pct(out.equity_irr_pa)],
        ["Peak debt", money(out.peak_debt, ccy)],
    ]
    t = Table(summary_data, colWidths=[70*mm, 100*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#D1D5DB")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(Paragraph("Summary", styles["Heading2"]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Products
    prod_df = pd.DataFrame(out.product_rows)
    prod_cols = ["Product", "Type", "Sellable sqm", "Price / sqm", "GDV", "Build cost / sqm", "Build base", "Inclusionary sqm"]
    prod_table = [prod_cols]
    for _, r in prod_df[prod_cols].iterrows():
        prod_table.append([
            str(r["Product"]),
            str(r["Type"]),
            f'{float(r["Sellable sqm"]):,.0f}',
            money(float(r["Price / sqm"]), ccy),
            money(float(r["GDV"]), ccy),
            money(float(r["Build cost / sqm"]), ccy),
            money(float(r["Build base"]), ccy),
            f'{float(r["Inclusionary sqm"]):,.0f}',
        ])
    pt = Table(prod_table, repeatRows=1)
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#D1D5DB")),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("PADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(Paragraph("Product mix", styles["Heading2"]))
    story.append(pt)
    story.append(Spacer(1, 12))

    # Assumptions (selected)
    story.append(Paragraph("Key assumptions", styles["Heading2"]))
    key_assumptions = [
        ["Currency", a.currency],
        ["Build months", str(a.build_months)],
        ["Sales months", str(a.sales_months)],
        ["Sales curve", a.sales_curve],
        ["Contingency", pct(a.contingency_rate)],
        ["Escalation p.a.", pct(a.escalation_rate_pa)],
        ["Professional fees", pct(a.professional_fees_rate)],
        ["Marketing (of GDV)", pct(a.marketing_rate)],
        ["Overhead / month", money(a.overhead_per_month, ccy)],
        ["Statutory costs", money(a.statutory_costs, ccy)],
        ["Inclusionary rate", pct(a.inclusionary_rate)],
        ["Inclusionary price / sqm", money(a.inclusionary_price_per_sqm, ccy)],
        ["Profit target", f"{pct(a.target_profit_rate)} of {a.target_profit_basis.upper()}"],
        ["Debt interest p.a.", pct(a.debt_interest_rate_pa) if a.use_debt else "N/A"],
        ["Arrangement fee", pct(a.debt_arrangement_fee_rate) if a.use_debt else "N/A"],
        ["Exit fee", pct(a.debt_exit_fee_rate) if a.use_debt else "N/A"],
        ["Equity injection (m0)", money(a.equity_injection_month0, ccy)],
        ["Land (input)", "Residual" if a.land_price_input is None else money(a.land_price_input, ccy)],
    ]
    at = Table([["Assumption", "Value"]] + key_assumptions, colWidths=[70*mm, 100*mm])
    at.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#D1D5DB")),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("PADDING", (0,0), (-1,-1), 5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
    ]))
    story.append(at)
    story.append(Spacer(1, 12))

    # Audit (trim to keep report compact)
    story.append(Paragraph("Audit trail (key lines)", styles["Heading2"]))
    audit_df = pd.DataFrame(out.audit)
    # keep most decision-useful lines only
    keep_sections = ["Revenue", "Costs", "Finance", "Land", "Profit"]
    audit_df = audit_df[audit_df["section"].isin(keep_sections)].copy()
    audit_df = audit_df.head(40)

    audit_table = [["Section", "Key", "Value"]]
    for _, r in audit_df.iterrows():
        audit_table.append([
            str(r["section"]),
            str(r["key"]),
            fmt_audit_value(r["value"], str(r.get("unit") or ""), ccy),
        ])
    au = Table(audit_table, repeatRows=1, colWidths=[25*mm, 80*mm, 65*mm])
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
st.set_page_config(page_title="SA Feasibility (MVP+)", layout="wide")
init_db()

st.title("SA-Style Development Feasibility — Streamlit MVP+")
st.caption("Product mix • Monthly finance cashflow • Scenarios (Base/Offer/Bank) • Compare • PDF report")

# Sidebar: projects + saved appraisals
with st.sidebar:
    st.header("Projects")
    projects = list_projects()

    with st.expander("➕ New project", expanded=False):
        new_name = st.text_input("Project name", value="")
        new_loc = st.text_input("Location", value="")
        if st.button("Create project", use_container_width=True) and new_name.strip():
            pid = create_project(new_name.strip(), new_loc.strip())
            st.success(f"Created project #{pid}")
            st.rerun()

    if not projects:
        st.info("Create a project to begin.")
        st.stop()

    project_labels = [f"#{p['id']} — {p['name']}" for p in projects]
    idx = st.selectbox("Select project", list(range(len(project_labels))), format_func=lambda i: project_labels[i])
    project = projects[idx]
    project_id = int(project["id"])

    st.divider()
    st.subheader("Saved appraisals")
    appraisals = list_appraisals(project_id)
    appraisal_id = None
    if appraisals:
        labels = [f"#{a['id']} — {a['version_name']} ({a['created_at']})" for a in appraisals]
        pick = st.selectbox("Load saved version", ["(none)"] + labels)
        if pick != "(none)":
            appraisal_id = int(pick.split("—")[0].strip().replace("#", ""))

# Session state assumptions
if "assumptions_dict" not in st.session_state:
    st.session_state.assumptions_dict = Assumptions().to_dict()

if appraisal_id:
    loaded = load_appraisal(appraisal_id)
    if loaded:
        st.session_state.assumptions_dict = loaded[0]

a = Assumptions.from_dict(st.session_state.assumptions_dict)

# Layout
left, right = st.columns([0.55, 0.45], gap="large")

with left:
    st.subheader("Inputs")

    # Global / programme
    g1, g2, g3, g4 = st.columns(4)
    a.currency = g1.selectbox("Currency", ["ZAR", "USD", "GBP", "EUR"], index=["ZAR","USD","GBP","EUR"].index(a.currency) if a.currency in ["ZAR","USD","GBP","EUR"] else 0)
    a.build_months = int(g2.number_input("Build months", 1, 72, int(a.build_months)))
    a.sales_months = int(g3.number_input("Sales months", 1, 72, int(a.sales_months)))
    a.sales_curve = g4.selectbox("Sales curve", ["linear", "front", "back"], index=["linear","front","back"].index(a.sales_curve) if a.sales_curve in ["linear","front","back"] else 0)

    st.markdown("### Product mix (multi-line revenue + build costs)")
    st.caption("Residential uses units × avg size. Commercial uses sellable m². Prices and build costs are per sellable m² (MVP proxy).")

    prods = list(a.products or [])
    # Ensure keys exist
    for p in prods:
        p.setdefault("name", "Product")
        p.setdefault("type", "commercial")
        p.setdefault("price_per_sqm", 0.0)
        p.setdefault("build_cost_per_sqm", 0.0)
        p.setdefault("inclusionary_eligible", (str(p.get("type")).lower() == "residential"))

    for i, p in enumerate(prods):
        with st.expander(f"{i+1}) {p.get('name','Product')}", expanded=(i == 0)):
            c1, c2, c3 = st.columns([0.45, 0.25, 0.30])
            p["name"] = c1.text_input("Name", value=str(p.get("name") or "Product"), key=f"p_name_{i}")
            p["type"] = c2.selectbox("Type", ["residential", "commercial"], index=0 if str(p.get("type")).lower()=="residential" else 1, key=f"p_type_{i}")
            p["inclusionary_eligible"] = c3.toggle("Inclusionary eligible", value=bool(p.get("inclusionary_eligible", p["type"]=="residential")), key=f"p_inc_{i}")

            if p["type"] == "residential":
                r1, r2, r3 = st.columns(3)
                p["units"] = int(r1.number_input("Units", min_value=0, value=int(p.get("units") or 0), step=1, key=f"p_units_{i}"))
                p["avg_unit_size_sqm"] = float(r2.number_input("Avg unit size (m²)", min_value=0.0, value=float(p.get("avg_unit_size_sqm") or 0.0), step=1.0, key=f"p_size_{i}"))
                sellable = (p["units"] * p["avg_unit_size_sqm"])
                r3.metric("Computed sellable m²", f"{sellable:,.0f}")
                p["sellable_sqm"] = None
            else:
                p["sellable_sqm"] = float(st.number_input("Sellable area (m²)", min_value=0.0, value=float(p.get("sellable_sqm") or 0.0), step=10.0, key=f"p_sqm_{i}"))
                p["units"] = None
                p["avg_unit_size_sqm"] = None

            k1, k2 = st.columns(2)
            p["price_per_sqm"] = float(k1.number_input("Sales price (ZAR/m²)", min_value=0.0, value=float(p.get("price_per_sqm") or 0.0), step=500.0, key=f"p_price_{i}"))
            p["build_cost_per_sqm"] = float(k2.number_input("Build cost (ZAR/m²)", min_value=0.0, value=float(p.get("build_cost_per_sqm") or 0.0), step=500.0, key=f"p_cost_{i}"))

            d1, d2 = st.columns(2)
            if d1.button("Duplicate line", key=f"dup_{i}"):
                prods.insert(i+1, dict(p))
                st.session_state.assumptions_dict["products"] = prods
                st.rerun()
            if d2.button("Delete line", key=f"del_{i}") and len(prods) > 1:
                prods.pop(i)
                st.session_state.assumptions_dict["products"] = prods
                st.rerun()

    if st.button("➕ Add product line", use_container_width=True):
        prods.append({
            "name": "New line",
            "type": "commercial",
            "sellable_sqm": 100.0,
            "price_per_sqm": 30000.0,
            "build_cost_per_sqm": 15000.0,
            "inclusionary_eligible": False,
        })
        st.session_state.assumptions_dict["products"] = prods
        st.rerun()

    a.products = prods

    st.markdown("### Development costs & overlays")
    cA, cB, cC, cD = st.columns(4)
    a.contingency_rate = cA.slider("Contingency", 0.0, 0.20, float(a.contingency_rate), 0.01)
    a.escalation_rate_pa = cB.slider("Escalation p.a.", 0.0, 0.20, float(a.escalation_rate_pa), 0.005)
    a.professional_fees_rate = cC.slider("Professional fees", 0.0, 0.25, float(a.professional_fees_rate), 0.01)
    a.heritage_cost_uplift_rate = cD.slider("Heritage uplift on build", 0.0, 0.25, float(a.heritage_cost_uplift_rate), 0.01)

    dA, dB, dC, dD = st.columns(4)
    a.statutory_costs = float(dA.number_input("Statutory (lump sum)", min_value=0.0, value=float(a.statutory_costs), step=50000.0))
    a.marketing_rate = dB.slider("Marketing (% of GDV)", 0.0, 0.10, float(a.marketing_rate), 0.005)
    a.overhead_per_month = float(dC.number_input("Overhead / month", min_value=0.0, value=float(a.overhead_per_month), step=5000.0))
    a.incentive_grant = float(dD.number_input("Incentive / grant (reduces cost)", value=float(a.incentive_grant), step=50000.0))

    st.markdown("### Inclusionary (SA policy toggle)")
    i1, i2 = st.columns(2)
    a.inclusionary_rate = i1.slider("Inclusionary % (eligible lines)", 0.0, 0.30, float(a.inclusionary_rate), 0.01)
    a.inclusionary_price_per_sqm = float(i2.number_input("Inclusionary price (ZAR/m²)", min_value=0.0, value=float(a.inclusionary_price_per_sqm), step=500.0))

    st.markdown("### Profit target")
    p1, p2 = st.columns(2)
    a.target_profit_basis = p1.selectbox("Target basis", ["gdv", "cost"], index=0 if a.target_profit_basis == "gdv" else 1)
    a.target_profit_rate = p2.slider("Target profit", 0.0, 0.35, float(a.target_profit_rate), 0.01)

    st.markdown("### Land")
    land_in = st.number_input("Land price input (0 = solve residual)", value=float(a.land_price_input or 0.0), step=250000.0)
    a.land_price_input = None if land_in == 0.0 else float(land_in)

    st.markdown("### Finance (monthly cashflow)")
    f1, f2, f3, f4 = st.columns(4)
    a.use_debt = f1.toggle("Use debt", value=bool(a.use_debt))
    a.debt_interest_rate_pa = f2.slider("Interest p.a.", 0.0, 0.30, float(a.debt_interest_rate_pa), 0.005)
    a.debt_arrangement_fee_rate = f3.slider("Arrangement fee (% peak debt)", 0.0, 0.05, float(a.debt_arrangement_fee_rate), 0.001)
    a.debt_exit_fee_rate = f4.slider("Exit fee (% peak debt)", 0.0, 0.05, float(a.debt_exit_fee_rate), 0.001)

    a.equity_injection_month0 = float(st.number_input("Equity injection at Month 0 (reduces debt)", value=float(a.equity_injection_month0), step=250000.0))

    # Persist
    st.session_state.assumptions_dict = a.to_dict()

with right:
    st.subheader("Outputs")
    out = run_appraisal(a)
    kpi_cards(out)

    # Product breakdown table
    st.markdown("### Product breakdown")
    df_prod = pd.DataFrame(out.product_rows)
    ccy = out.currency
    if not df_prod.empty:
        df_show = df_prod.copy()
        for col in ["Price / sqm", "GDV", "Build cost / sqm", "Build base"]:
            if col in df_show.columns:
                df_show[col] = df_show[col].astype(float).map(lambda v: money(v, ccy))
        for col in ["Sellable sqm", "Inclusionary sqm"]:
            if col in df_show.columns:
                df_show[col] = df_show[col].astype(float).map(lambda v: f"{v:,.0f}")
        st.dataframe(df_show, use_container_width=True, hide_index=True)

    tabs = st.tabs(["Cashflow", "Sensitivity", "Scenarios & Compare", "Audit", "PDF"])
    with tabs[0]:
        st.caption("Monthly cashflow lines include debt draws/repayments + capitalised interest.")
        df_cf = pd.DataFrame(out.cashflow_rows)

        # Plot: revenue vs cost (ex land/fin)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_cf["Month"], y=df_cf["Revenue"], name="Revenue"))
        fig.add_trace(go.Bar(x=df_cf["Month"], y=-df_cf["Costs (ex land, ex fin)"], name="-Costs (ex land/fin)"))
        fig.update_layout(barmode="relative", height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Plot: debt draws/repay and interest
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=df_cf["Month"], y=df_cf["Debt draw"], name="Debt draw"))
        fig2.add_trace(go.Bar(x=df_cf["Month"], y=-df_cf["Debt repay"], name="-Debt repay"))
        fig2.add_trace(go.Scatter(x=df_cf["Month"], y=df_cf["Interest (cap.)"], name="Interest (cap.)", mode="lines+markers"))
        fig2.update_layout(barmode="relative", height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(df_cf, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.caption("2D sensitivity: product prices vs product build costs → residual land value (ZAR).")
        rows, cols, mat = sensitivity_grid(a)
        heat = go.Figure(data=go.Heatmap(
            z=mat,
            x=cols,
            y=rows,
            hovertemplate="Price: %{y}<br>Cost: %{x}<br>Land: %{z:,.0f}<extra></extra>"
        ))
        heat.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(heat, use_container_width=True)
        st.dataframe(pd.DataFrame(mat, index=rows, columns=cols).applymap(lambda v: money(float(v), ccy)), use_container_width=True)

    with tabs[2]:
        st.caption("Save Base / Offer / Bank cases, then compare side-by-side.")
        colS1, colS2, colS3 = st.columns(3)
        with colS1:
            if st.button("💾 Save as Base", use_container_width=True):
                save_appraisal(project_id, "Base", a.to_dict(), out.headline_dict())
                st.success("Saved Base")
                st.rerun()
        with colS2:
            if st.button("💾 Save as Offer", use_container_width=True):
                save_appraisal(project_id, "Offer", a.to_dict(), out.headline_dict())
                st.success("Saved Offer")
                st.rerun()
        with colS3:
            if st.button("💾 Save as Bank", use_container_width=True):
                save_appraisal(project_id, "Bank", a.to_dict(), out.headline_dict())
                st.success("Saved Bank")
                st.rerun()

        st.divider()
        appraisals = list_appraisals(project_id)
        if not appraisals:
            st.info("No saved appraisals yet.")
        else:
            labels = {a_["id"]: f"#{a_['id']} — {a_['version_name']} ({a_['created_at']})" for a_ in appraisals}
            ids = list(labels.keys())

            cA, cB, cC = st.columns(3)
            pickA = cA.selectbox("Compare slot A", ids, format_func=lambda i: labels[i])
            pickB = cB.selectbox("Compare slot B", ids, format_func=lambda i: labels[i], index=min(1, len(ids)-1))
            pickC = cC.selectbox("Compare slot C", ids, format_func=lambda i: labels[i], index=min(2, len(ids)-1))

            def load_out(app_id):
                loaded = load_appraisal(int(app_id))
                if not loaded:
                    return None
                aa = Assumptions.from_dict(loaded[0])
                oo = run_appraisal(aa)  # recompute to ensure consistent engine
                return aa, oo

            comps = []
            for slot, pid in [("A", pickA), ("B", pickB), ("C", pickC)]:
                res = load_out(pid)
                if res:
                    aa, oo = res
                    comps.append({
                        "Slot": slot,
                        "Version": labels[pid],
                        "GDV": oo.gdv,
                        "Costs ex land/fin": oo.tdc_ex_land_ex_finance,
                        "Finance": oo.finance_costs,
                        "Land": oo.land_value,
                        "Profit": oo.profit,
                        "Profit % GDV": oo.profit_rate_on_gdv,
                        "Equity IRR p.a.": oo.equity_irr_pa if oo.equity_irr_pa is not None else None,
                        "Peak debt": oo.peak_debt,
                    })

            df_cmp = pd.DataFrame(comps)
            if not df_cmp.empty:
                df_show = df_cmp.copy()
                for col in ["GDV", "Costs ex land/fin", "Finance", "Land", "Profit", "Peak debt"]:
                    df_show[col] = df_show[col].astype(float).map(lambda v: money(v, ccy))
                df_show["Profit % GDV"] = df_show["Profit % GDV"].astype(float).map(pct)
                df_show["Equity IRR p.a."] = df_show["Equity IRR p.a."].apply(lambda v: "-" if v is None else pct(float(v)))
                st.dataframe(df_show, use_container_width=True, hide_index=True)

    with tabs[3]:
        df_a = pd.DataFrame(out.audit)
        df_a["Display"] = df_a.apply(lambda r: fmt_audit_value(r["value"], str(r.get("unit") or ""), ccy), axis=1)
        st.dataframe(df_a[["section", "key", "Display"]], use_container_width=True, hide_index=True)

    with tabs[4]:
        st.caption("Generate a PDF report for the current scenario (summary + product mix + assumptions + key audit lines).")
        pdf_bytes = build_pdf(project["name"], project.get("location") or "", a, out)
        st.download_button(
            "📄 Download PDF report",
            data=pdf_bytes,
            file_name=f"feasibility_{project['name'].replace(' ','_')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
