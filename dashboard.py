import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import io
import re
import tempfile
import os
from dotenv import load_dotenv
import json
from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import hashlib
import time

st.set_page_config(layout="centered")

# CSS pour la page d'authentification
st.markdown(
    """
    <style>
    .auth-container {
        max-width: 400px;
        margin: 0px auto;
        padding: 30px;
        border-radius: 12px;
        background: linear-gradient(135deg, #7f7fd5, #86a8e7, #91eae4);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .auth-title {
        font-size: 2rem;
        margin-top: 30px;
        margin-bottom: 20px;
        font-weight: bold;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .auth-input input {
        border-radius: 8px;
        border: none;
        padding: 10px;
        width: 400px;
        margin-bottom: 15px;
    }
    .stButton {
        display: flex;
        justify-content: center;
    }
    .stButton > button {
        background: white;
        color: #4a4a4a;
        border: 2px solid grey;
        border-radius: 25px;
        padding: 6px 15px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #7f7fd5, #86a8e7, #91eae4);
        color: black;
        border-color: #7f7fd5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === Authentification ===
load_dotenv()
SUPER_ADMIN_EMAIL = os.getenv("SUPER_ADMIN_EMAIL")
AUTHORIZED_USERS = json.loads(st.secrets["AUTHORIZED_USERS_JSON"])

# Connexion SQLite
conn = sqlite3.connect("users.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, password TEXT)''')
conn.commit()

# Authentification utilisateur
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_credentials(email, password):
    c.execute("SELECT password FROM users WHERE email = ?", (email,))
    result = c.fetchone()
    return result and result[0] == hash_password(password)

def user_exists(email):
    c.execute("SELECT 1 FROM users WHERE email = ?", (email,))
    return c.fetchone() is not None

def create_user(email, password):
    c.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hash_password(password)))
    conn.commit()

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    st.markdown('<div class="auth-container">Mission Occitanie-Est </div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">🔐 Authentification requise</div>', unsafe_allow_html=True)

    email = st.text_input("Adresse e-mail")
    if email:
        if email not in AUTHORIZED_USERS:
            st.error("⛔ Cet e-mail n'est pas autorisé.")
            st.stop()

        if not user_exists(email):
            st.info("👋 Premier accès détecté. Veuillez définir un mot de passe.")
            new_password = st.text_input("Créer un mot de passe", type="password")
            if st.button("Créer mon compte", key="create_account"):
                if len(new_password) < 4:
                    st.warning("🔒 Mot de passe trop court.")
                else:
                    create_user(email, new_password)
                    st.success("✅ Compte créé. Vous pouvez maintenant vous connecter.")
                    st.rerun()
        else:
            password = st.text_input("Mot de passe", type="password")
            if st.button("Se connecter", key="login"):
                if check_credentials(email, password):
                    st.session_state['authenticated'] = True
                    st.session_state['email'] = email
                    st.rerun()
                else:
                    st.error("❌ Mot de passe incorrect.")
            st.stop()
    st.markdown('</div>', unsafe_allow_html=True)

# === Interface après authentification réussie ===
if st.session_state['authenticated'] and 'email' in st.session_state:
    with st.sidebar:
        if st.session_state['email'] == SUPER_ADMIN_EMAIL:
            st.header("Super Admin Dashboard")
        else:
            st.info(f"✅ Connecté en tant qu'utilisateur : {st.session_state['email']}")
        if st.button("🔓 Se déconnecter", key=f"logout_{st.session_state['email']}"):
            st.session_state['authenticated'] = False
            del st.session_state['email']
            st.rerun()

    st.set_page_config(layout="wide")

    MOIS_FR = {
        'January': 'Janvier', 'February': 'Février', 'March': 'Mars',
        'April': 'Avril', 'May': 'Mai', 'June': 'Juin',
        'July': 'Juillet', 'August': 'Août', 'September': 'Septembre',
        'October': 'Octobre', 'November': 'Novembre', 'December': 'Décembre'
    }

    st.title("Dashboard Missions Occitanie Est - Synthèse mensuelle")

    @st.cache_data(ttl=0)  # pas de cache, données rechargées à chaque exécution
    def load_data():
        url = "https://docs.google.com/spreadsheets/d/1omlbZlKb_gpKW3K5GgbIB977T7hFajpyrMmcfCsld3Y/export?format=csv&gid=0"
        df = pd.read_csv(url)

        def clean_offrandes(val):
            if pd.isna(val):
                return 0.0
            val = str(val)
            val = re.sub(r'[ \u00A0\s]', '', val)  # supprime espaces normaux, insécables et \u00A0
            val = val.replace('€', '').replace(',', '.')
            try:
                return float(val) if val else 0.0
            except:
                return 0.0

        df['Offrandes'] = df['Offrandes'].apply(clean_offrandes)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna(subset=['Date'])

        df['Mois'] = df['Date'].dt.to_period('M')
        df['Mois'] = pd.Categorical(df['Mois'], ordered=True)
        return df

    df = load_data()

    missions = sorted(df['Mission'].dropna().unique())

    # --- Gestion accès missions selon user ---
    user_info = AUTHORIZED_USERS.get(st.session_state["email"], {})
    role = user_info.get("role", "user")
    missions_autorisees = missions if role == "superadmin" else user_info.get("missions", [])

    if not missions_autorisees:
        st.error("⚠️ Aucune mission n’est associée à ce compte.")
        st.stop()

    st.success(f"🎉 Bienvenue ! Vous êtes connecté au compte de : {', '.join(missions_autorisees)}")

    # Filtrage du dataframe global selon mission(s) autorisée(s)
    df = df[df['Mission'].isin(missions_autorisees)]

    MOIS_AVEC_DONNEES = sorted(df['Mois'].dropna().unique())

    def format_mois_fr(period):
        mois_annee = period.strftime('%B %Y')
        for en, fr in MOIS_FR.items():
            mois_annee = mois_annee.replace(en, fr)
        return mois_annee

    mois_str = [format_mois_fr(m) for m in MOIS_AVEC_DONNEES]

    def get_previous_month_index():
        today = pd.Timestamp.today()
        prev_month = (today - pd.DateOffset(months=1)).to_period('M')
        try:
            return MOIS_AVEC_DONNEES.index(prev_month)
        except:
            return len(MOIS_AVEC_DONNEES) - 1

    # === Sélection mois uniquement (plus de sélection mission visible) ===
    default_month_idx = get_previous_month_index()
    selected_months = st.multiselect("Sélectionnez le(s) mois", options=mois_str, default=[mois_str[default_month_idx]])

    if not selected_months:
        st.warning("❗ Veuillez sélectionner au moins un mois pour afficher les données.")
        st.stop()

    try:
        selected_months_period = [MOIS_AVEC_DONNEES[mois_str.index(m)] for m in selected_months]
    except ValueError:
        st.info("⏳ Encore un peu de patience, il n'y a pas de statistiques pour ce(s) mois sélectionné(s).")
        st.stop()

    df_filtered = df[df['Mois'].isin(selected_months_period)]

    if df_filtered.empty:
        st.info("⏳ Encore un peu de patience, il n'y a pas de statistiques pour ce(s) mois sélectionné(s).")
        st.stop()

    # === Moyennes ===
    cols_moyenne = ['Hommes', 'Femmes', 'Adultes']
    grouped_moyenne = df_filtered.groupby(['Mission', 'Mois'], observed=True)[cols_moyenne].mean().round().reset_index()
    if not grouped_moyenne.empty:
        grouped_moyenne['Mois'] = grouped_moyenne['Mois'].apply(format_mois_fr)
        st.subheader("📊 Moyenne de fréquentation aux rencontres des missions par mois")
        st.dataframe(grouped_moyenne, use_container_width=True, hide_index=True)
        fig_moy = px.bar(grouped_moyenne, x='Mois', y='Adultes', color='Mission',
                        barmode='group', text='Mission', title="Fréquentation des adultes par mission")
        fig_moy.update_layout(bargap=0.2, bargroupgap=0.04)
        fig_moy.update_traces(marker_line_width=0, textposition='auto', textfont=dict(color='white'))
        st.plotly_chart(fig_moy, use_container_width=True)

    # === NA / NC ===
    grouped_nanac = df_filtered.groupby(['Mission', 'Mois'], observed=True)[['NA', 'NC']].sum().reset_index()
    if not grouped_nanac.empty:
        grouped_nanac['Mois'] = grouped_nanac['Mois'].apply(format_mois_fr)
        st.subheader("🧾 Nombre de NA | NC des missions par mois")
        st.dataframe(grouped_nanac, use_container_width=True, hide_index=True)
        fig_na = px.bar(grouped_nanac, x='Mois', y='NA', color='Mission',
                        barmode='group', text='Mission', title="Nombre de NA par mission")
        fig_na.update_layout(bargap=0.2, bargroupgap=0.07)
        fig_na.update_traces(marker_line_width=0, textposition='auto', textfont=dict(color='white'))
        fig_nc = px.bar(grouped_nanac, x='Mois', y='NC', color='Mission',
                        barmode='group', text='Mission', title="Nombre de NC par mission")
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
    if not grouped_offrandes.empty:
        grouped_offrandes['Mois'] = grouped_offrandes['Mois'].apply(format_mois_fr)
        grouped_offrandes_graph['Mois'] = grouped_offrandes_graph['Mois'].apply(format_mois_fr)
        grouped_offrandes['Offrandes'] = grouped_offrandes['Offrandes'].apply(lambda x: f"{x:,.2f} €".replace(",", " ").replace(".", ","))
        st.subheader("💶 Somme des dons des missions par mois")
        st.dataframe(grouped_offrandes, use_container_width=True, hide_index=True)
        fig_off = px.bar(grouped_offrandes_graph, x='Mois', y='Offrandes', color='Mission',
                        barmode='group', text='Mission', title="Dons par mission")
        fig_off.update_layout(bargap=0.2, bargroupgap=0.04)
        fig_off.update_traces(marker_line_width=0, textposition='auto', textfont=dict(color='white'))
        st.plotly_chart(fig_off, use_container_width=True)

    # === Export Excel ===
    excel_output = io.BytesIO()
    with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
        grouped_moyenne.to_excel(writer, index=False, sheet_name='Moyennes')
        grouped_nanac.to_excel(writer, index=False, sheet_name='NA_NC')
        grouped_offrandes.to_excel(writer, index=False, sheet_name='Dons')
    excel_output.seek(0)

    st.download_button("📥 Télécharger Excel", data=excel_output,
                    file_name='rapport_missions.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    # === PDF ===
    def plot_grouped_bar(df, y_col, title, ylabel, path):
        plt.figure(figsize=(10, 5))
        missions_uniq = df['Mission'].unique()
        mois = df['Mois'].unique()
        x = np.arange(len(mois))

        group_gap = 0.15
        total_width = 0.8
        n = len(missions_uniq)
        bar_width = (total_width - group_gap * (n - 1)) / n if n > 0 else 0.8

        for i, mission in enumerate(missions_uniq):
            d = df[df['Mission'] == mission]
            bar_positions = x - total_width/2 + i * (bar_width + group_gap) + bar_width / 2
            bars = plt.bar(bar_positions, d[y_col], width=bar_width, label=mission)

            if y_col in ['Adultes', 'Dons']:
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
        pdf.cell(0, 10, f"Fréquentation moyenne aux rencontres - {', '.join(mois_label)}", 0, 1, 'C')
        add_table(pdf, df_moy)
        pdf.ln(5)
        pdf.image(imgs['moyenne'], x=10, w=190)

        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f"Nombre de NA | NC - {', '.join(mois_label)}", 0, 1, 'C')
        add_table(pdf, df_nanac)
        pdf.ln(5)
        pdf.image(imgs['na'], x=10, w=190, h=60)
        pdf.add_page()
        pdf.image(imgs['nc'], x=10, w=190, h=60)

        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f"Offrandes - {', '.join(mois_label)}", 0, 1, 'C')
        add_table(pdf, df_off)
        pdf.ln(5)
        pdf.image(imgs['offrandes'], x=10, w=190)

        output_pdf = io.BytesIO()
        pdf.output(output_pdf)
        output_pdf.seek(0)
        return output_pdf

    with tempfile.TemporaryDirectory() as tmpdir:
        path_moy = os.path.join(tmpdir, "moyenne.png")
        path_na = os.path.join(tmpdir, "na.png")
        path_nc = os.path.join(tmpdir, "nc.png")
        path_off = os.path.join(tmpdir, "offrandes.png")

        plot_grouped_bar(grouped_moyenne, 'Adultes', 'Fréquentation des adultes', 'Nombre d’adultes', path_moy)
        plot_grouped_bar(grouped_nanac, 'NA', 'Nombre de NA', 'Nombre de NA', path_na)
        plot_grouped_bar(grouped_nanac, 'NC', 'Nombre de NC', 'Nombre de NC', path_nc)
        plot_grouped_bar(grouped_offrandes_graph, 'Offrandes', 'Offrandes par mission', 'Euros', path_off)

        pdf_data = create_pdf(grouped_moyenne, grouped_nanac, grouped_offrandes, {
            'moyenne': path_moy,
            'na': path_na,
            'nc': path_nc,
            'offrandes': path_off
        }, [format_mois_fr(m) for m in selected_months_period])

    st.download_button("📥 Télécharger rapport PDF", data=pdf_data,
                       file_name="rapport_mission.pdf",
                       mime="application/pdf")
