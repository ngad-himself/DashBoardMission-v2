import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import io
import tempfile
import os
from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")

MOIS_FR = {
    'January': 'Janvier', 'February': 'Février', 'March': 'Mars',
    'April': 'Avril', 'May': 'Mai', 'June': 'Juin',
    'July': 'Juillet', 'August': 'Août', 'September': 'Septembre',
    'October': 'Octobre', 'November': 'Novembre', 'December': 'Décembre'
}

st.title("Dashboard Missions Occitanie Est - Synthèse mensuelle")

@st.cache_data
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1omlbZlKb_gpKW3K5GgbIB977T7hFajpyrMmcfCsld3Y/export?format=csv&gid=0"
    df = pd.read_csv(url)

    def clean_offrandes(val):
        if pd.isna(val):
            return 0.0
        val = str(val).replace('€', '').replace(',', '.').strip()
        try:
            return float(val) if val else 0.0
        except:
            return 0.0

    df['Offrandes'] = df['Offrandes'].apply(clean_offrandes)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Modification clé: utiliser des strings au lieu de Period
    df['Mois'] = df['Date'].dt.strftime('%Y-%m')
    mois_complet = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='MS').strftime('%Y-%m').tolist()
    df['Mois'] = pd.Categorical(df['Mois'], categories=mois_complet, ordered=True)

    return df, mois_complet

df, mois_complet = load_data()

def format_mois_fr(month_str):
    dt = datetime.strptime(month_str, '%Y-%m')
    mois_annee = dt.strftime('%B %Y')
    for en, fr in MOIS_FR.items():
        mois_annee = mois_annee.replace(en, fr)
    return mois_annee

missions = sorted(df['Mission'].dropna().unique())
mois_str = [format_mois_fr(m) for m in mois_complet]

def get_previous_month_index():
    today = pd.Timestamp.today()
    prev_month = (today - pd.DateOffset(months=1)).strftime('%Y-%m')
    try:
        return mois_complet.index(prev_month)
    except ValueError:
        return len(mois_complet) - 1 if mois_complet else 0

# Sélections
default_month_idx = get_previous_month_index()
default_missions = [m for m in ['Alès', 'Béziers', 'MNO'] if m in missions]
selected_missions = st.multiselect("Sélectionnez la/les mission(s)", options=missions, default=default_missions)
selected_months = st.multiselect("Sélectionnez le(s) mois", options=mois_str, default=[mois_str[default_month_idx]] if mois_str else [])

if not selected_missions or not selected_months:
    st.warning("❗ Veuillez sélectionner au moins une mission et un mois pour afficher les données.")
    st.stop()

# Vérification des données disponibles
try:
    selected_months_str = [mois_complet[mois_str.index(m)] for m in selected_months]
    df_filtered = df[(df['Mission'].isin(selected_missions)) & (df['Mois'].isin(selected_months_str))]
    
    # Vérification stricte
    if df_filtered.empty or df_filtered[['Hommes', 'Femmes', 'Adultes', 'NA', 'NC', 'Offrandes']].isnull().all().all():
        st.info("⏳ Encore un peu de patience, il n'y a pas de statistiques pour ce(s) mois sélectionné(s).")
        st.stop()
        
except Exception as e:
    st.info("⏳ Encore un peu de patience, il n'y a pas de statistiques pour ce(s) mois sélectionné(s).")
    st.stop()

# === Moyennes ===
cols_moyenne = ['Hommes', 'Femmes', 'Adultes']
grouped_moyenne = df_filtered.groupby(['Mission', 'Mois'], observed=True)[cols_moyenne].mean().round().reset_index()
grouped_moyenne['Mois'] = grouped_moyenne['Mois'].apply(format_mois_fr)

st.subheader("📊 Moyenne de fréquentation au culte des missions et par mois")
st.dataframe(grouped_moyenne, use_container_width=True, hide_index=True)

fig_moy = px.bar(grouped_moyenne, x='Mois', y='Adultes', color='Mission',
                 barmode='group', text='Mission', title="Fréquentation des Adultes par Mission")
fig_moy.update_layout(bargap=0.2, bargroupgap=0.04)
fig_moy.update_traces(marker_line_width=0, textposition='auto', textfont=dict(color='white'))
st.plotly_chart(fig_moy, use_container_width=True)

# === NA / NC ===
grouped_nanac = df_filtered.groupby(['Mission', 'Mois'], observed=True)[['NA', 'NC']].sum().reset_index()
grouped_nanac['Mois'] = grouped_nanac['Mois'].apply(format_mois_fr)

st.subheader("🧾 Nombre de NA | NC des missions par mois")
st.dataframe(grouped_nanac, use_container_width=True, hide_index=True)

fig_na = px.bar(grouped_nanac, x='Mois', y='NA', color='Mission',
                barmode='group', text='Mission', title="Nombre de NA par Mission")
fig_na.update_layout(bargap=0.2, bargroupgap=0.07)
fig_na.update_traces(marker_line_width=0, textposition='auto', textfont=dict(color='white'))

fig_nc = px.bar(grouped_nanac, x='Mois', y='NC', color='Mission',
                barmode='group', text='Mission', title="Nombre de NC par Mission")
fig_nc.update_layout(bargap=0.2, bargroupgap=0.07)
fig_nc.update_traces(marker_line_width=0, textposition='auto', textfont=dict(color='white'))

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_na, use_container_width=True)
with col2:
    st.plotly_chart(fig_nc, use_container_width=True)

# === Offrandes ===
grouped_offrandes = df_filtered.groupby(['Mission', 'Mois'], observed=True)[['Offrandes']].sum().reset_index()
grouped_offrandes_graph = grouped_offrandes.copy()
grouped_offrandes['Mois'] = grouped_offrandes['Mois'].apply(format_mois_fr)
grouped_offrandes_graph['Mois'] = grouped_offrandes_graph['Mois'].apply(format_mois_fr)
grouped_offrandes['Offrandes'] = grouped_offrandes['Offrandes'].apply(lambda x: f"{x:,.2f} €".replace(",", " ").replace(".", ","))

st.subheader("💶 Somme des Offrandes des missions par mois")
st.dataframe(grouped_offrandes, use_container_width=True, hide_index=True)

fig_off = px.bar(grouped_offrandes_graph, x='Mois', y='Offrandes', color='Mission',
                 barmode='group', text='Mission', title="Offrandes par Mission")
fig_off.update_layout(bargap=0.2, bargroupgap=0.04)
fig_off.update_traces(marker_line_width=0, textposition='auto', textfont=dict(color='white'))
st.plotly_chart(fig_off, use_container_width=True)

# === Export Excel ===
excel_output = io.BytesIO()
with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
    grouped_moyenne.to_excel(writer, index=False, sheet_name='Moyennes')
    grouped_nanac.to_excel(writer, index=False, sheet_name='NA_NC')
    grouped_offrandes.to_excel(writer, index=False, sheet_name='Offrandes')
excel_output.seek(0)

st.download_button("📥 Télécharger Excel", data=excel_output,
                   file_name='rapport_missions.xlsx',
                   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# === PDF ===
def plot_grouped_bar(df, y_col, title, ylabel, path):
    plt.figure(figsize=(10, 5))
    missions = df['Mission'].unique()
    mois = df['Mois'].unique()
    x = np.arange(len(mois))

    group_gap = 0.15
    total_width = 0.8
    n = len(missions)
    bar_width = (total_width - group_gap * (n - 1)) / n

    for i, mission in enumerate(missions):
        d = df[df['Mission'] == mission]
        bar_positions = x - total_width/2 + i * (bar_width + group_gap) + bar_width / 2
        bars = plt.bar(bar_positions, d[y_col], width=bar_width, label=mission)

        if y_col in ['Adultes', 'Offrandes']:
            for xi, yi in zip(bar_positions, d[y_col]):
                plt.text(xi, yi + 0.5, str(int(yi)), ha='center', va='bottom', fontsize=8, color='black')

    plt.grid(axis='y', linestyle='--', alpha=0.2)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x, mois, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, "Rapport Mensuel de Mission - RMM", 0, 1, 'C')
        self.set_font('Arial', '', 10)
        date_str = datetime.today().strftime('%d/%m/%Y')
        self.cell(0, 10, f"Ce rapport a été généré à la date : {date_str}", 0, 1, 'L')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def add_table(pdf, df):
    pdf.set_font('Arial', '', 8)
    col_widths = [35] + [30] * (len(df.columns) - 1)
    table_width = sum(col_widths)
    x_start = (pdf.w - table_width) / 2
    pdf.set_x(x_start)
    pdf.set_fill_color(200, 220, 255)

    for i, col in enumerate(df.columns):
        pdf.cell(col_widths[i], 8, str(col), 1, 0, 'C', 1)
    pdf.ln()

    for _, row in df.iterrows():
        pdf.set_x(x_start)
        for i, col in enumerate(df.columns):
            val = row[col]
            if isinstance(val, float) and val.is_integer():
                val = int(val)
            txt = str(val).replace('€', 'Euros')
            pdf.cell(col_widths[i], 7, txt, 1, 0, 'C')
        pdf.ln()

def create_pdf(df_moy, df_nanac, df_off, imgs, mois_label):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f"Fréquentation moyenne au culte - {', '.join(mois_label)}", 0, 1, 'C')
    add_table(pdf, df_moy)
    pdf.ln(5)
    pdf.image(imgs['moyenne'], x=10, w=190)

    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f"Nombre de NA | NC - {', '.join(mois_label)}", 0, 1, 'C')
    add_table(pdf, df_nanac)
    pdf.ln(5)
    pdf.image(imgs['na'], x=10, w=190, h=90)
    pdf.ln(3)
    pdf.image(imgs['nc'], x=10, w=190, h=90)

    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f"Somme des offrandes - {', '.join(mois_label)}", 0, 1, 'C')
    add_table(pdf, df_off)
    pdf.ln(5)
    pdf.image(imgs['offrandes'], x=10, w=190)

    return pdf

if st.button("🖨️ Générer le rapport PDF"):
    with st.spinner("Création du PDF en cours..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = {
                "moyenne": os.path.join(temp_dir, "moyenne.png"),
                "na": os.path.join(temp_dir, "na.png"),
                "nc": os.path.join(temp_dir, "nc.png"),
                "offrandes": os.path.join(temp_dir, "offrandes.png")
            }

            plot_grouped_bar(grouped_moyenne, 'Adultes', "Fréquentation des Adultes par Mission", "Adultes", paths['moyenne'])
            plot_grouped_bar(grouped_nanac, 'NA', "Nombre de NA par Mission", "NA", paths['na'])
            plot_grouped_bar(grouped_nanac, 'NC', "Nombre de NC par Mission", "NC", paths['nc'])
            plot_grouped_bar(grouped_offrandes_graph, 'Offrandes', "Offrandes par Mission", "Euros", paths['offrandes'])

            pdf = create_pdf(grouped_moyenne, grouped_nanac, grouped_offrandes, paths, selected_months)
            pdf_path = os.path.join(temp_dir, "rapport_missions.pdf")
            pdf.output(pdf_path)

            with open(pdf_path, "rb") as f:
                st.success("✅ Rapport PDF généré avec succès !")
                st.download_button("📄 Télécharger le rapport PDF", data=f,
                                   file_name="rapport_missions.pdf", mime="application/pdf")