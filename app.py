import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Configuration de la page Streamlit
st.set_page_config(page_title="Annette K. - Dashboard NPS", layout="wide")

# Fonction pour charger les données
@st.cache_data
def load_data():
    # Chargement direct du CSV
    df = pd.read_csv("data/NPS ANNETTE K. Sauvegarde - anonymes.csv")
    # Afficher les noms des colonnes pour debug
    st.write("Colonnes disponibles:", df.columns.tolist())
    # Conversion de l'horodateur en datetime
    df['Horodateur'] = pd.to_datetime(df['Horodateur'], format='%d/%m/%Y %H:%M:%S')
    return df

# Chargement des données
try:
    df = load_data()
except Exception as e:
    st.error(f"Erreur lors du chargement des données: {str(e)}")
    st.stop()

# Fonction pour calculer le NPS
def calculate_nps(scores):
    promoters = sum(scores >= 9)
    detractors = sum(scores <= 6)
    total = len(scores)
    return ((promoters - detractors) / total) * 100 if total > 0 else 0

# Header
st.title("🏊‍♂️ Annette K. - Dashboard NPS et Satisfaction")

# Pour debugger, affichons les premières lignes du DataFrame
st.write("Aperçu des données:", df.head())

# Trouver la colonne NPS (elle peut avoir un nom légèrement différent)
nps_column = [col for col in df.columns if 'Recommandation' in col][0]
retention_column = [col for col in df.columns if 'probabilité' in col][0]

# Métriques principales
col1, col2, col3 = st.columns(3)

with col1:
    nps_score = calculate_nps(df[nps_column].dropna())
    st.metric("NPS Score", f"{nps_score:.1f}%")

with col2:
    retention_score = df[retention_column].mean()
    st.metric("Score de Rétention Moyen", f"{retention_score:.1f}/10")

with col3:
    responses_count = len(df)
    st.metric("Nombre de Réponses", responses_count)

# Évolution du NPS dans le temps
st.subheader("Évolution du NPS dans le temps")
df['Month'] = df['Horodateur'].dt.to_period('M')
monthly_nps = df.groupby('Month').agg({
    nps_column: lambda x: calculate_nps(x)
}).reset_index()
monthly_nps['Month'] = monthly_nps['Month'].astype(str)

fig_nps = px.line(monthly_nps, 
                  x='Month', 
                  y=nps_column,
                  title="Évolution mensuelle du NPS",
                  labels={nps_column: 'NPS Score (%)',
                         'Month': 'Mois'})
st.plotly_chart(fig_nps, use_container_width=True)

# Ajout du graphique des volumes promoteurs/neutres/détracteurs
st.subheader("Répartition mensuelle des répondants")
def get_nps_category(score):
    if score >= 9:
        return 'Promoteurs'
    elif score <= 6:
        return 'Détracteurs'
    else:
        return 'Neutres'

df['NPS_Category'] = df[nps_column].apply(get_nps_category)
monthly_volumes = df.groupby(['Month', 'NPS_Category']).size().reset_index(name='count')

fig_volumes = px.bar(monthly_volumes,
                    x='Month',
                    y='count',
                    color='NPS_Category',
                    title="Répartition mensuelle des répondants",
                    labels={'count': 'Nombre de répondants',
                           'Month': 'Mois',
                           'NPS_Category': 'Catégorie'},
                    category_orders={'NPS_Category': ['Détracteurs', 'Neutres', 'Promoteurs']},
                    color_discrete_map={'Promoteurs': '#00CC96',
                                      'Neutres': '#FFA15A',
                                      'Détracteurs': '#EF553B'})
st.plotly_chart(fig_volumes, use_container_width=True)

# Satisfaction par catégorie
st.subheader("Satisfaction par catégorie")
satisfaction_cols = [col for col in df.columns if "sur une echelle de 1 à 5" in col.lower()]
satisfaction_data = df[satisfaction_cols]
clean_names = [col.split("notez votre satisfaction concernant ")[-1].strip() for col in satisfaction_cols]
satisfaction_means = satisfaction_data.mean()

fig_satisfaction = go.Figure(go.Bar(
    y=clean_names,
    x=satisfaction_means,
    orientation='h',
    text=satisfaction_means.round(2),
    textposition='auto',
))

fig_satisfaction.update_layout(
    title="Satisfaction moyenne par catégorie",
    xaxis_title="Score moyen (1-5)",
    yaxis_title="Catégorie",
    height=600
)

st.plotly_chart(fig_satisfaction, use_container_width=True)

# Analyse des commentaires
st.subheader("Analyse des suggestions d'amélioration")
comments_column = "Si vous étiez manager chez Annette K, Quelles améliorations proposeriez vous ?"

# Fonction pour nettoyer le texte
def clean_text(text):
    if isinstance(text, str):
        return text.lower().strip()
    return ""

comments = df[comments_column].dropna()
comments_text = " ".join(comments.apply(clean_text))

wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(comments_text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

# Section filtrable pour voir les commentaires bruts
st.subheader("Commentaires détaillés")
if st.checkbox("Afficher tous les commentaires"):
    st.dataframe(df[["Horodateur", comments_column]].dropna())

# Footer avec dernière mise à jour
st.markdown("---")
st.markdown(f"*Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y %H:%M')}*")
