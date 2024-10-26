import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from textblob import TextBlob
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Configuration de la page Streamlit
st.set_page_config(page_title="Annette K. - Dashboard NPS", layout="wide")

# Fonction pour charger les données
@st.cache_data
def load_data():
    # Dans un cas réel, on utiliserait l'API Google Sheets ici
    # Pour l'exemple, on va charger directement le CSV
    df = pd.read_csv("NPS ANNETTE K. Sauvegarde - anonymes.csv")
    df['Horodateur'] = pd.to_datetime(df['Horodateur'], format='%d/%m/%Y %H:%M:%S')
    return df

# Chargement des données
df = load_data()

# Fonction pour calculer le NPS
def calculate_nps(scores):
    promoters = sum(scores >= 9)
    detractors = sum(scores <= 6)
    total = len(scores)
    return ((promoters - detractors) / total) * 100 if total > 0 else 0

# Header
st.title("🏊‍♂️ Annette K. - Dashboard NPS et Satisfaction")

# Métriques principales
col1, col2, col3 = st.columns(3)

with col1:
    nps_score = calculate_nps(df['Recommandation\nSur une échelle de 1 à 10'].dropna())
    st.metric("NPS Score", f"{nps_score:.1f}%")

with col2:
    retention_score = df['Sur une échelle de 1 à 10, \nQuelle est la probabilité que vous soyez toujours abonné chez Annette K. dans 6 mois ?'].mean()
    st.metric("Score de Rétention Moyen", f"{retention_score:.1f}/10")

with col3:
    responses_count = len(df)
    st.metric("Nombre de Réponses", responses_count)

# Évolution du NPS dans le temps
st.subheader("Évolution du NPS dans le temps")
df['Month'] = df['Horodateur'].dt.to_period('M')
monthly_nps = df.groupby('Month').agg({
    'Recommandation\nSur une échelle de 1 à 10': lambda x: calculate_nps(x)
}).reset_index()
monthly_nps['Month'] = monthly_nps['Month'].astype(str)

fig_nps = px.line(monthly_nps, 
                  x='Month', 
                  y='Recommandation\nSur une échelle de 1 à 10',
                  title="Évolution mensuelle du NPS",
                  labels={'Recommandation\nSur une échelle de 1 à 10': 'NPS Score (%)',
                         'Month': 'Mois'})
st.plotly_chart(fig_nps, use_container_width=True)

# Satisfaction par catégorie
st.subheader("Satisfaction par catégorie")

# Sélection des colonnes de satisfaction (échelle 1-5)
satisfaction_cols = [col for col in df.columns if "sur une echelle de 1 à 5" in col.lower()]
satisfaction_data = df[satisfaction_cols]

# Nettoyage des noms de colonnes pour l'affichage
clean_names = [col.split("notez votre satisfaction concernant ")[-1].strip() for col in satisfaction_cols]
satisfaction_means = satisfaction_data.mean()

# Création du graphique en barres horizontales
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

# Fonction pour nettoyer le texte
def clean_text(text):
    if isinstance(text, str):
        return text.lower().strip()
    return ""

# Agrégation des commentaires
comments = df["Si vous étiez manager chez Annette K, Quelles améliorations proposeriez vous ?"].dropna()
comments_text = " ".join(comments.apply(clean_text))

# Création du nuage de mots
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(comments_text)

# Affichage du nuage de mots
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

# Section filtrable pour voir les commentaires bruts
st.subheader("Commentaires détaillés")
if st.checkbox("Afficher tous les commentaires"):
    st.dataframe(df[["Horodateur", "Si vous étiez manager chez Annette K, Quelles améliorations proposeriez vous ?"]].dropna())

# Footer avec dernière mise à jour
st.markdown("---")
st.markdown(f"*Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y %H:%M')}*")

