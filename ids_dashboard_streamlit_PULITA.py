
import os
import io
from pathlib import Path
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

st.set_page_config(page_title="IDS/Q8 Fleet Intelligence", layout="wide")

ALIASES = {
    "date": ["TrsDate", "Transaction Date", "Date"],
    "time": ["TrsTime", "Transaction Time", "Time"],
    "product": ["Product", "ProductName", "Fuel Product"],
    "volume": ["Volume", "Qty", "Quantity", "Liters", "Litres"],
    "net_amount_eur": ["EUR NetAmount", "NetAmount", "Net Amount", "Amount EUR", "Net Amount EUR"],
    "gross_amount_eur": ["EUR GrossAmount", "GrossAmount", "Gross Amount"],
    "net_unit_price_eur": ["EUR NetUnitPrice", "NetUnitPrice", "Net Unit Price EUR", "EUR Net Unit Price"],
    "gross_unit_price_eur": ["EUR GrossUnitPrice", "GrossUnitPrice", "Gross Unit Price EUR", "EUR Gross Unit Price"],
    "plate": ["PlateNr", "Plate", "Vehicle", "Registration"],
    "driver": ["DriverName", "Driver", "Employee"],
    "card": ["CardNr", "Card", "Card Number"],
    "station": ["StationName", "SiteName", "MerchantName", "Station"],
    "country": ["Country", "CountryName"],
    "odometer": ["Odometer", "Mileage", "Km", "KmInsertion"],
}

def euro(x):
    if pd.isna(x):
        return "n.d."
    return f"€ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def liters(x):
    if pd.isna(x):
        return "n.d."
    return f"{x:,.1f} L".replace(",", "X").replace(".", ",").replace("X", ".")

def percent(x):
    if pd.isna(x):
        return "n.d."
    return f"{100*x:.1f}%".replace(".", ",")

def check_login():
    default_user = os.environ.get("IDS_APP_USER", "admin")
    default_pass = os.environ.get("IDS_APP_PASSWORD", "1234")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        return

    st.title("🔐 Accesso dashboard IDS/Q8")
    st.caption("Inserisci le credenziali per accedere alla dashboard.")

    c1, c2 = st.columns(2)
    with c1:
        user = st.text_input("Username")
    with c2:
        password = st.text_input("Password", type="password")

    if st.button("Accedi", type="primary"):
        if user == default_user and password == default_pass:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Credenziali non corrette.")

    st.stop()

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for std, names in ALIASES.items():
        for name in names:
            if name in df.columns:
                rename_map[name] = std
                break
    return df.rename(columns=rename_map)

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_dataframe(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise FileNotFoundError("Carica un file Excel IDS/Q8 per avviare l'analisi.")

    df = pd.read_excel(uploaded_file)
    source_name = uploaded_file.name

    df = standardize_columns(df)
    numeric_cols = [
        "volume", "net_amount_eur", "gross_amount_eur",
        "net_unit_price_eur", "gross_unit_price_eur", "odometer"
    ]
    df = coerce_numeric(df, numeric_cols)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "time" in df.columns:
        df["time"] = df["time"].astype(str)

    df.attrs["source_name"] = source_name
    return df

def prepare_kpi_dataframe(df: pd.DataFrame):
    work = df.copy()
    initial_rows = len(work)

    if "net_unit_price_eur" in work.columns and work["net_unit_price_eur"].notna().any():
        work["price_per_liter"] = work["net_unit_price_eur"]
        price_source = "EUR NetUnitPrice"
    elif {"net_amount_eur", "volume"}.issubset(work.columns):
        work["price_per_liter"] = np.where(work["volume"] > 0, work["net_amount_eur"] / work["volume"], np.nan)
        price_source = "EUR NetAmount / Volume"
    elif "gross_unit_price_eur" in work.columns and work["gross_unit_price_eur"].notna().any():
        work["price_per_liter"] = work["gross_unit_price_eur"]
        price_source = "EUR GrossUnitPrice"
    elif {"gross_amount_eur", "volume"}.issubset(work.columns):
        work["price_per_liter"] = np.where(work["volume"] > 0, work["gross_amount_eur"] / work["volume"], np.nan)
        price_source = "EUR GrossAmount / Volume"
    else:
        work["price_per_liter"] = np.nan
        price_source = "non disponibile"

    if "net_amount_eur" in work.columns and work["net_amount_eur"].notna().any():
        work["amount_eur"] = work["net_amount_eur"]
    elif "gross_amount_eur" in work.columns and work["gross_amount_eur"].notna().any():
        work["amount_eur"] = work["gross_amount_eur"]
    else:
        work["amount_eur"] = np.nan

    work["excluded_reason"] = np.nan
    if "volume" in work.columns:
        work.loc[work["volume"].isna(), "excluded_reason"] = "Volume mancante"
        work.loc[work["volume"] <= 0, "excluded_reason"] = "Volume nullo o negativo"
    else:
        work["excluded_reason"] = "Colonna Volume assente"

    work.loc[work["price_per_liter"].isna() & work["excluded_reason"].isna(), "excluded_reason"] = "Prezzo/L non disponibile"
    work.loc[(work["price_per_liter"] < 0.20) & work["excluded_reason"].isna(), "excluded_reason"] = "Prezzo/L troppo basso"
    work.loc[(work["price_per_liter"] > 3.50) & work["excluded_reason"].isna(), "excluded_reason"] = "Prezzo/L troppo alto"

    valid = work[work["excluded_reason"].isna()].copy()
    excluded = work[work["excluded_reason"].notna()].copy()

    meta = {
        "initial_rows": initial_rows,
        "valid_rows": len(valid),
        "excluded_rows": len(excluded),
        "price_source": price_source,
    }
    return valid, excluded, meta

def build_pdf_buffer(summary_rows, bullet_points, title="Report executive IDS/Q8"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generato il {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("KPI principali", styles["Heading2"]))
    table_data = [["Indicatore", "Valore"]] + summary_rows
    table = Table(table_data, colWidths=[220, 220])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Messaggi chiave", styles["Heading2"]))
    for item in bullet_points:
        story.append(Paragraph(f"• {item}", styles["Normal"]))
        story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer

check_login()

st.title("🚛 IDS/Q8 Fleet Intelligence")
st.caption("Dashboard executive: carica il file Excel, ottieni KPI affidabili, alert automatici e report PDF.")

uploaded = st.file_uploader(
    "Carica un file Excel IDS/Q8",
    type=["xlsx"]
)

try:
    raw = load_dataframe(uploaded)
except Exception as e:
    st.info(str(e))
    st.stop()

valid, excluded, meta = prepare_kpi_dataframe(raw)

with st.sidebar:
    st.header("Filtri")
    if "date" in valid.columns and valid["date"].notna().any():
        min_d = valid["date"].min().date()
        max_d = valid["date"].max().date()
        date_range = st.date_input("Periodo", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_d, end_d = date_range
            valid = valid[(valid["date"].dt.date >= start_d) & (valid["date"].dt.date <= end_d)]

    for col, label in [("product", "Prodotto"), ("country", "Paese"), ("plate", "Targa"), ("station", "Stazione")]:
        if col in valid.columns:
            values = sorted([x for x in valid[col].dropna().astype(str).unique().tolist() if x.strip() != ""])
            selected = st.multiselect(label, values)
            if selected:
                valid = valid[valid[col].astype(str).isin(selected)]

    st.divider()
    st.subheader("Qualità del dato")
    st.write(f"Righe originali: **{meta['initial_rows']}**")
    st.write(f"Righe valide KPI: **{len(valid)}**")
    st.write(f"Righe escluse: **{meta['excluded_rows']}**")
    st.write(f"Prezzo/L da: **{meta['price_source']}**")

if valid.empty:
    st.warning("Nessuna riga valida nel perimetro selezionato.")
    st.stop()

total_cost = valid["amount_eur"].sum()
total_volume = valid["volume"].sum()
avg_price = total_cost / total_volume if total_volume > 0 else np.nan
transactions = len(valid)

st.success(f"File in uso: {Path(raw.attrs.get('source_name', 'n.d.')).name}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Spesa totale", euro(total_cost))
k2.metric("Volume totale", liters(total_volume))
k3.metric("Prezzo medio / litro", euro(avg_price))
k4.metric("Transazioni valide", f"{transactions:,}".replace(",", "."))

alerts = []
if pd.notna(avg_price) and avg_price > 2.00:
    alerts.append("Prezzo medio/L sopra la soglia di attenzione.")
if len(excluded) / meta["initial_rows"] > 0.10:
    alerts.append("Incidenza righe escluse superiore al 10%.")
if "station" in valid.columns:
    station_tbl = (
        valid.groupby("station")
             .agg(spesa=("amount_eur", "sum"), volume=("volume", "sum"), transazioni=("station", "size"))
             .query("transazioni >= 3 and volume > 50")
             .assign(prezzo_medio_l=lambda x: x["spesa"] / x["volume"])
             .sort_values("prezzo_medio_l", ascending=False)
    )
    if not station_tbl.empty and station_tbl["prezzo_medio_l"].max() - station_tbl["prezzo_medio_l"].min() > 0.15:
        alerts.append("Dispersione prezzo/L significativa tra stazioni: esiste potenziale saving.")
if alerts:
    for a in alerts:
        st.warning(f"🚨 {a}")

st.subheader("Executive summary")
bullets = []
if "product" in valid.columns:
    by_product = valid.groupby("product")["amount_eur"].sum().sort_values(ascending=False)
    if not by_product.empty:
        bullets.append(f"Il prodotto con maggiore spesa è {by_product.index[0]} con {euro(by_product.iloc[0])}.")
if "plate" in valid.columns:
    by_plate = valid.groupby("plate")["amount_eur"].sum().sort_values(ascending=False)
    if not by_plate.empty:
        bullets.append(f"La targa con maggiore spesa è {by_plate.index[0]} con {euro(by_plate.iloc[0])}.")
if "station" in valid.columns:
    station_tbl = (
        valid.groupby("station")
             .agg(spesa=("amount_eur", "sum"), volume=("volume", "sum"), transazioni=("station", "size"))
             .query("transazioni >= 3 and volume > 50")
             .assign(prezzo_medio_l=lambda x: x["spesa"] / x["volume"])
    )
    if not station_tbl.empty:
        expensive = station_tbl.sort_values("prezzo_medio_l", ascending=False).head(1)
        cheap = station_tbl.sort_values("prezzo_medio_l", ascending=True).head(1)
        diff = float(expensive["prezzo_medio_l"].iloc[0] - cheap["prezzo_medio_l"].iloc[0])
        saving = max(0, diff * total_volume * 0.10)
        bullets.append(f"La stazione più cara è {expensive.index[0]} con {euro(expensive['prezzo_medio_l'].iloc[0])}/L.")
        bullets.append(f"Spostando il 10% dei volumi dalla stazione più cara a quella più conveniente, il saving teorico è circa {euro(saving)}.")
if len(excluded) > 0:
    bullets.append(f"Sono state escluse {len(excluded)} righe non affidabili dai KPI principali.")

for b in bullets:
    st.markdown(f"- {b}")

st.subheader("Download report")
summary_rows = [
    ["Spesa totale", euro(total_cost)],
    ["Volume totale", liters(total_volume)],
    ["Prezzo medio / litro", euro(avg_price)],
    ["Transazioni valide", f"{transactions:,}".replace(",", ".")],
    ["Righe escluse", f"{len(excluded):,}".replace(",", ".")],
]
pdf_buffer = build_pdf_buffer(summary_rows, bullets)
st.download_button(
    "📄 Scarica report PDF",
    data=pdf_buffer,
    file_name=f"report_ids_q8_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
    mime="application/pdf"
)

c1, c2 = st.columns(2)
if "product" in valid.columns:
    mix = valid.groupby("product")["amount_eur"].sum().reset_index()
    c1.plotly_chart(px.pie(mix, names="product", values="amount_eur", title="Mix spesa per prodotto"), use_container_width=True)
if "date" in valid.columns:
    daily = valid.groupby(valid["date"].dt.date)["amount_eur"].sum().reset_index()
    daily.columns = ["date", "amount_eur"]
    c2.plotly_chart(px.line(daily, x="date", y="amount_eur", title="Trend giornaliero spesa"), use_container_width=True)

c3, c4 = st.columns(2)
if "plate" in valid.columns:
    top_plate = valid.groupby("plate")["amount_eur"].sum().sort_values(ascending=False).head(10).reset_index()
    c3.plotly_chart(px.bar(top_plate, x="plate", y="amount_eur", title="Top 10 targhe per spesa"), use_container_width=True)
if "station" in valid.columns:
    top_station = (
        valid.groupby("station")
             .agg(spesa=("amount_eur", "sum"), volume=("volume", "sum"), transazioni=("station", "size"))
             .query("transazioni >= 3 and volume > 50")
             .assign(prezzo_medio_l=lambda x: x["spesa"] / x["volume"])
             .sort_values("prezzo_medio_l", ascending=False)
             .head(10)
             .reset_index()
    )
    if not top_station.empty:
        c4.plotly_chart(px.bar(top_station, x="station", y="prezzo_medio_l", title="Top 10 stazioni per prezzo medio €/L"), use_container_width=True)

st.subheader("Controlli di qualità del dato")
dq1, dq2 = st.columns(2)
dq_summary = pd.DataFrame({
    "Indicatore": ["Righe originali", "Righe valide", "Righe escluse", "Incidenza righe escluse"],
    "Valore": [
        meta["initial_rows"],
        len(valid),
        len(excluded),
        percent(len(excluded) / meta["initial_rows"] if meta["initial_rows"] else np.nan),
    ],
})
dq1.dataframe(dq_summary, use_container_width=True)
if not excluded.empty:
    reasons = excluded["excluded_reason"].value_counts().reset_index()
    reasons.columns = ["Motivo esclusione", "Righe"]
    dq2.dataframe(reasons, use_container_width=True)
else:
    dq2.success("Nessuna riga esclusa dai KPI.")

st.caption("Nota metodologica: versione pulita con solo upload manuale del file Excel, ideale per uso online e condivisione.")
