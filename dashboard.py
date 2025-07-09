import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import io
import base64
import tempfile
import os
import plotly.io as pio
import platform
import ctypes.util
from ctypes.util import find_library
from weasyprint import HTML


# Installation des dépendances système pour Streamlit Cloud
if platform.system() == "Linux" and "STREAMLIT_SERVER_RUNNING" in os.environ:
    os.system('apt-get update && apt-get install -y libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0')
    
# Configuration GTK
GTK_PATH = r"C:\Program Files\GTK3-Runtime Win64\bin"
os.environ['PATH'] = GTK_PATH + os.pathsep + os.environ['PATH']

required_dlls = ['libgobject-2.0-0.dll', 'libcairo-2.dll']
missing_dlls = [dll for dll in required_dlls if not os.path.exists(os.path.join(GTK_PATH, dll))]

if missing_dlls:
    st.error(f"DLL manquantes : {', '.join(missing_dlls)}. Réinstallez GTK.")
    st.stop()

# Dictionnaire de traduction des mois
MOIS_FR = {
    'January': 'Janvier', 'February': 'Février', 'March': 'Mars',
    'April': 'Avril', 'May': 'Mai', 'June': 'Juin',
    'July': 'Juillet', 'August': 'Août', 'September': 'Septembre',
    'October': 'Octobre', 'November': 'Novembre', 'December': 'Décembre'
}

st.title("Dashboard Missions - Synthèse mensuelle avec filtres avancés")

@st.cache_data
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1omlbZlKb_gpKW3K5GgbIB977T7hFajpyrMmcfCsld3Y/export?format=csv&gid=0"
    df = pd.read_csv(url)
    
    def clean_offrandes(val):
        if pd.isna(val):
            return 0.0
        val = str(val).replace('€', '').replace(',', '.')
        parts = ''.join([c if c.isdigit() or c == '.' else ' ' for c in val]).split()
        try:
            return float(parts[0]) if parts else 0.0
        except:
            return 0.0

    df['Offrandes'] = df['Offrandes'].apply(clean_offrandes)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Date'])
    
    annee_min = df['Date'].dt.year.min()
    annee_max = df['Date'].dt.year.max()
    df['Mois'] = df['Date'].dt.to_period('M')
    mois_complet = pd.period_range(start=f'{annee_min}-01', end=f'{annee_max}-12', freq='M')
    df['Mois'] = pd.Categorical(df['Mois'], categories=mois_complet, ordered=True)
    
    return df, mois_complet

df, mois_complet = load_data()

missions = sorted(df['Mission'].dropna().unique())
default_missions = [m for m in ['Alès', 'Béziers', 'MNO'] if m in missions]

# Formatage des mois en français
def format_mois_fr(period):
    mois_annee = period.strftime('%B %Y')
    for en, fr in MOIS_FR.items():
        mois_annee = mois_annee.replace(en, fr)
    return mois_annee

mois_str = [format_mois_fr(m) for m in mois_complet]

def get_previous_month_index():
    today = pd.Timestamp.today()
    prev_month = (today - pd.DateOffset(months=1)).to_period('M')
    try:
        return mois_complet.get_loc(prev_month)
    except KeyError:
        return len(mois_complet) - 1

def create_pdf_figure(df, x_col, y_cols, title):
    if isinstance(y_cols, list):
        fig = px.bar(df, x=x_col, y=y_cols, color='Mission', 
                    barmode='group', title=title, text='Mission')
    else:
        fig = px.bar(df, x=x_col, y=y_cols, color='Mission',
                    title=title, text='Mission')
    
    fig.update_traces(
        texttemplate='<b>%{text}</b>',
        textposition='inside',
        insidetextfont=dict(color='white', size=10),
        marker=dict(line_width=0),
        opacity=1.0
    )
    
    fig.update_layout(
        uniformtext_minsize=10,
        bargap=0.3,
        bargroupgap=0.1,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def save_fig_as_svg(fig, filename):
    fig.update_layout(
        template='plotly_white',
        colorway=['#FF0000', '#00FF00', '#0000FF', '#FFA500', '#800080'],
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black', size=12),
    )
    
    fig.update_traces(
        marker=dict(line=dict(width=1, color='black')),
        opacity=1.0
    )
    
    fig.write_image(filename, format='svg', engine='kaleido')
    with open(filename, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def generate_pdf_html(df_moyenne, df_nanac, df_offrandes, base64_svgs, mois_selection):
    mois_titre = ', '.join(mois_selection)
    date_aujourdhui = datetime.today().strftime('%d/%m/%Y')
    
    def svg(b64):
        svg_content = base64.b64decode(b64).decode('utf-8')
        return f'<div style="width:100%; margin:10px 0; border:1px solid #eee">{svg_content}</div>'
    
    html = f"""
    <html>
    <head><meta charset='utf-8'>
    <style>
        @page {{ size: A4; margin: 1cm; }}
        body {{ font-family: Arial; padding: 20px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; page-break-inside: avoid; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        .svg-container {{ margin: 20px 0; page-break-inside: avoid; }}
        .footer {{ margin-top: 30px; font-size: 0.9em; color: #7f8c8d; text-align: center; }}
    </style>
    </head>
    <body>
        <h1>Rapport de mission - {mois_titre}</h1>
        <p><strong>Date de génération :</strong> {date_aujourdhui}</p>
        <p><strong>Missions incluses :</strong> {', '.join(df_moyenne['Mission'].unique())}</p>

        <h2>Moyennes Hommes / Femmes / Adultes</h2>
        {df_moyenne.to_html(index=False, border=0)}
        <div class="svg-container">{svg(base64_svgs['moyenne'])}</div>

        <h2>Sommes NA / NC</h2>
        {df_nanac.to_html(index=False, border=0)}
        <div class="svg-container">{svg(base64_svgs['nanac'])}</div>

        <h2>Sommes Offrandes</h2>
        {df_offrandes.to_html(index=False, border=0)}
        <div class="svg-container">{svg(base64_svgs['offrandes'])}</div>
    </body>
    </html>
    """
    return html

# Interface Streamlit
default_month_idx = get_previous_month_index()
selected_missions = st.multiselect("Sélectionnez la/les mission(s)", options=missions, default=default_missions)
selected_months = st.multiselect("Sélectionnez le(s) mois", options=mois_str, default=[mois_str[default_month_idx]])
selected_months_period = [mois_complet[mois_str.index(m)] for m in selected_months]

df_filtered = df[(df['Mission'].isin(selected_missions)) & (df['Mois'].isin(selected_months_period))]

# Section Moyennes
cols_moyenne = ['Hommes', 'Femmes', 'Adultes']
grouped_moyenne = df_filtered.groupby(['Mission', 'Mois'], observed=True)[cols_moyenne].mean().round().reset_index()
grouped_moyenne = grouped_moyenne.dropna(subset=cols_moyenne)
grouped_moyenne['Mois'] = grouped_moyenne['Mois'].apply(format_mois_fr)

st.subheader("📊 Moyenne des Hommes, Femmes, Adultes par Mission et Mois")
st.dataframe(grouped_moyenne, use_container_width=True, hide_index=True)

fig_moy = px.bar(grouped_moyenne, x='Mois', y=cols_moyenne, color='Mission', 
                barmode='group', title="Moyennes par Mission")
fig_moy.update_layout(bargap=0.3, bargroupgap=0.1)
st.plotly_chart(fig_moy)

# Section NA/NC
grouped_nanac = df_filtered.groupby(['Mission', 'Mois'], observed=True)[['NA', 'NC']].sum().reset_index()
grouped_nanac = grouped_nanac.dropna(subset=['NA', 'NC'])
grouped_nanac['Mois'] = grouped_nanac['Mois'].apply(format_mois_fr)

st.subheader("🧾 Somme des NA et NC par Mission et Mois")
st.dataframe(grouped_nanac, use_container_width=True, hide_index=True)

fig_nanac = px.bar(grouped_nanac, x='Mois', y=['NA', 'NC'], color='Mission', 
                  barmode='group', title="NA/NC par Mission")
fig_nanac.update_layout(bargap=0.3, bargroupgap=0.1)
st.plotly_chart(fig_nanac)

# Section Offrandes
grouped_offrandes = df_filtered.groupby(['Mission', 'Mois'], observed=True)[['Offrandes']].sum().reset_index()
grouped_offrandes_graph = grouped_offrandes.copy()
grouped_offrandes_graph['Mois'] = grouped_offrandes_graph['Mois'].apply(format_mois_fr)
grouped_offrandes['Offrandes'] = grouped_offrandes['Offrandes'].apply(lambda x: f"{x:,.2f} €".replace(",", " ").replace(".", ","))
grouped_offrandes['Mois'] = grouped_offrandes['Mois'].apply(format_mois_fr)

st.subheader("💶 Somme des Offrandes par Mission et Mois")
st.dataframe(grouped_offrandes, use_container_width=True, hide_index=True)

fig_off = px.bar(grouped_offrandes_graph, x='Mois', y='Offrandes', 
                color='Mission', barmode='group', title="Offrandes par Mission")
fig_off.update_layout(bargap=0.3, bargroupgap=0.1)
st.plotly_chart(fig_off)

# Export Excel
excel_output = io.BytesIO()
with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
    grouped_moyenne.to_excel(writer, index=False, sheet_name='Moyennes')
    grouped_nanac.to_excel(writer, index=False, sheet_name='NA_NC')
    grouped_offrandes.to_excel(writer, index=False, sheet_name='Offrandes')
excel_output.seek(0)

st.download_button("📥 Télécharger les statistiques en Excel", data=excel_output,
                  file_name='rapport_missions.xlsx',
                  mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# Export PDF
if st.button("🖨️ Générer le rapport PDF"):
    with tempfile.TemporaryDirectory() as temp_dir:
        moy_path = os.path.join(temp_dir, "moy.svg")
        nanac_path = os.path.join(temp_dir, "nanac.svg")
        off_path = os.path.join(temp_dir, "off.svg")

        b64_svgs = {
            'moyenne': save_fig_as_svg(fig_moy, moy_path),
            'nanac': save_fig_as_svg(fig_nanac, nanac_path),
            'offrandes': save_fig_as_svg(fig_off, off_path)
        }

        html_report = generate_pdf_html(grouped_moyenne, grouped_nanac, grouped_offrandes, b64_svgs, selected_months)

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                HTML(string=html_report).write_pdf(tmp_file.name)
                with open(tmp_file.name, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                    st.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>', unsafe_allow_html=True)
                    st.download_button("📄 Télécharger rapport PDF", data=f, file_name="rapport_missions.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Erreur PDF : {str(e)}")


