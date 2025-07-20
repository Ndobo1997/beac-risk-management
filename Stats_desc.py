import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import lognorm, gamma, weibull_min, norm, expon, pareto, poisson
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Analyse Exploratoire - Risque Opérationnel", 
    page_icon="📊", 
    layout="wide"
)

def formater_montant(montant, format_court=False):
    """Formate un montant en FCFA avec les unités appropriées (k, M, Md)"""
    if pd.isna(montant) or montant == 0:
        return "0" if format_court else "0 FCFA"
    elif montant >= 1_000_000_000:
        return f"{montant/1_000_000_000:.1f} Md" if format_court else f"{montant/1_000_000_000:.1f} Md FCFA"
    elif montant >= 1_000_000:
        return f"{montant/1_000_000:.1f} M" if format_court else f"{montant/1_000_000:.1f} M FCFA"
    elif montant >= 1_000:
        return f"{montant/1_000:.1f} k" if format_court else f"{montant/1_000:.1f} k FCFA"
    else:
        return f"{montant:.0f}" if format_court else f"{montant:.0f} FCFA"

def analyse_exploratoire_complete():
    """
    Analyse exploratoire complète des données de risque opérationnel
    Adaptée à la structure exacte du fichier base_incidents.xlsx
    Avec affichage des vraies valeurs en FCFA
    """
    
    st.title("📊 Analyse Exploratoire - Risque Opérationnel")
    st.markdown("---")
    st.write("*Analyse complète des caractéristiques, tendances et distributions des incidents*")
    
    # =================== CHARGEMENT DES DONNÉES ===================
    
    @st.cache_data
    def charger_donnees():
        """Chargement et nettoyage des données avec conversion en FCFA réels"""
        try:
            # Chargement du fichier Excel avec la structure exacte
            df = pd.read_excel("data/base_incidents.xlsx", sheet_name="Incidents_DOF_augmente")
            
            st.info(f"📁 **Fichier chargé** - Colonnes disponibles : {list(df.columns)}")
            
            # Colonnes principales identifiées
            cout_col = 'Cout_total_estime(en 10 000 FCFA)'
            entite_col = 'Entité'
            date_col = 'Date de survenance'
            categorie_col = 'Catégorie_Risque'
            gravite_col = 'Gravité'
            
            # Nettoyage et CONVERSION DES COÛTS EN FCFA RÉELS
            cout_unite_10k = pd.to_numeric(df[cout_col], errors='coerce')
            cout_fcfa_reel = cout_unite_10k * 10_000  # Conversion en FCFA réels
            
            # Nettoyage des entités
            entites = df[entite_col].astype(str)
            
            # Conversion des dates du format DD-MM-YYYY
            dates_str = df[date_col].astype(str)
            dates = pd.to_datetime(dates_str, format='%d-%m-%Y', errors='coerce')
            
            # Types/catégories de risque
            categories = df[categorie_col].astype(str)
            
            # Gravités
            gravites = df[gravite_col].astype(str)
            
            # Masque pour données valides
            mask = (
                (~pd.isna(cout_fcfa_reel)) & 
                (cout_fcfa_reel > 0) & 
                (~pd.isna(dates)) & 
                (entites != 'nan') & 
                (categories != 'nan')
            )
            
            # Construction du dataframe nettoyé avec vraies valeurs FCFA
            df_clean = pd.DataFrame({
                'Severite': cout_fcfa_reel[mask],  # Maintenant en FCFA réels
                'Entite': entites[mask],
                'Date': dates[mask],
                'Categorie_Risque': categories[mask],
                'Gravite': gravites[mask],
                'Annee': dates[mask].dt.year,
                'Mois': dates[mask].dt.month,
                'Code': df['Code'][mask]
            })
            
            return {
                'df_original': df,
                'severites': cout_fcfa_reel[mask].values,  # En FCFA réels
                'entites': entites[mask].values,
                'dates': dates[mask],
                'categories': categories[mask].values,
                'gravites': gravites[mask].values,
                'df_clean': df_clean
            }
            
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement des données: {e}")
            st.info("🔍 Vérifiez que le fichier 'base_incidents.xlsx' est présent avec la feuille 'Incidents_DOF_augmente'")
            return None
    
    # Chargement des données
    data = charger_donnees()
    if data is None:
        st.stop()
    
    df = data['df_clean']
    severites = data['severites']
    entites = data['entites']
    dates = data['dates']
    categories = data['categories']
    gravites = data['gravites']
    
    # Informations générales
    st.success(f"✅ **{len(severites):,} observations** chargées avec succès sur **{len(data['df_original'])} incidents** totaux")
    
    # =================== 1. CARACTÉRISTIQUES GÉNÉRALES ===================
    
    st.header("1️⃣ Caractéristiques Générales des Données")
    
    # Vue d'ensemble générale
    col1, col2, col3, col4, col5 = st.columns(5)
    
    periode_debut = df['Date'].min().strftime('%Y-%m-%d')
    periode_fin = df['Date'].max().strftime('%Y-%m-%d')
    nb_annees = (df['Date'].max() - df['Date'].min()).days / 365.25
    
    col1.metric("📊 Total Incidents", f"{len(severites):,}")
    col2.metric("🏢 Entités Uniques", f"{df['Entite'].nunique()}")
    col3.metric("📅 Période Analyse", f"{nb_annees:.1f} ans")
    col4.metric("💰 Perte Totale", f"{np.sum(severites)/1_000_000_000:.1f}", delta="Milliards FCFA")
    col5.metric("📈 Fréquence Moyenne", f"{len(severites)/nb_annees:.1f}/an")
    
    st.info(f"**Période d'analyse :** {periode_debut} → {periode_fin}")
    st.caption("💡 **Note :** La perte totale est affichée en milliards de FCFA dans la métrique principale")
    
    # Vue des données brutes
    with st.expander("🔍 Aperçu des Données Brutes"):
        # Création d'un dataframe d'affichage avec formatage
        df_display_raw = df.copy()
        df_display_raw['Severite_Formatee'] = df_display_raw['Severite'].apply(formater_montant)
        
        # Colonnes à afficher
        colonnes_display = ['Code', 'Entite', 'Date', 'Categorie_Risque', 'Gravite', 'Severite_Formatee']
        df_show = df_display_raw[colonnes_display].head(10)
        df_show = df_show.rename(columns={'Severite_Formatee': 'Sévérité'})
        
        st.dataframe(df_show, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Catégories de Risque Uniques:**")
            st.write(df['Categorie_Risque'].value_counts())
        
        with col2:
            st.write("**Niveaux de Gravité Uniques:**")
            st.write(df['Gravite'].value_counts())
    
    # =================== 2. CARACTÉRISTIQUES PAR ENTITÉ ===================
    
    st.subheader("🏢 Analyse Détaillée par Entité")
    
    # Calcul des statistiques par entité avec formatage
    def calcul_stats_entites_avec_formatage(df):
        """Calcul des statistiques par entité avec formatage approprié"""
        try:
            stats_list = []
            for entite in df['Entite'].unique():
                subset = df[df['Entite'] == entite]
                severites_entite = subset['Severite'].values  # Déjà en FCFA réels
                
                if len(severites_entite) > 0:
                    # Catégorie et gravité principales
                    cat_principale = subset['Categorie_Risque'].mode().iloc[0] if len(subset['Categorie_Risque'].mode()) > 0 else 'N/A'
                    grav_principale = subset['Gravite'].mode().iloc[0] if len(subset['Gravite'].mode()) > 0 else 'N/A'
                    
                    stats_entite = {
                        'Entite': entite,
                        'Nb_Incidents': len(severites_entite),
                        'Severite_Moyenne': np.mean(severites_entite),
                        'Severite_Mediane': np.median(severites_entite),
                        'Ecart_Type': np.std(severites_entite, ddof=1) if len(severites_entite) > 1 else 0,
                        'Severite_Min': np.min(severites_entite),
                        'Severite_Max': np.max(severites_entite),
                        'Perte_Totale': np.sum(severites_entite),
                        'Premiere_Date': subset['Date'].min(),
                        'Derniere_Date': subset['Date'].max(),
                        'Categorie_Principale': cat_principale,
                        'Gravite_Principale': grav_principale
                    }
                    stats_list.append(stats_entite)
            
            return pd.DataFrame(stats_list).set_index('Entite')
            
        except Exception as e:
            st.error(f"Erreur dans calcul_stats_entites_avec_formatage : {e}")
            return None
    
    stats_entites = calcul_stats_entites_avec_formatage(df)
    
    if stats_entites is not None:
        # Ajout de métriques dérivées
        total_incidents = len(df)
        total_pertes = df['Severite'].sum()
        
        stats_entites['Part_Incidents_%'] = (stats_entites['Nb_Incidents'] / total_incidents * 100).round(1)
        stats_entites['Part_Pertes_%'] = (stats_entites['Perte_Totale'] / total_pertes * 100).round(1)
        stats_entites['Frequence_Annuelle'] = (stats_entites['Nb_Incidents'] / nb_annees).round(1)
        
        # CV sécurisé
        cv_values = []
        for _, row in stats_entites.iterrows():
            if row['Severite_Moyenne'] > 0:
                cv = (row['Ecart_Type'] / row['Severite_Moyenne'])
            else:
                cv = 0
            cv_values.append(cv)
        
        stats_entites['CV_Severite'] = np.array(cv_values).round(2)
        
        # Nettoyage des noms d'entités pour l'affichage
        stats_entites['Entite_Court'] = [nom.replace('DGEFRI-DOF : ', '') for nom in stats_entites.index]
        
        # Ajout des colonnes formatées
        stats_entites['Severite_Moyenne_Format'] = stats_entites['Severite_Moyenne'].apply(formater_montant)
        stats_entites['Severite_Mediane_Format'] = stats_entites['Severite_Mediane'].apply(formater_montant)
        stats_entites['Severite_Max_Format'] = stats_entites['Severite_Max'].apply(formater_montant)
        stats_entites['Perte_Totale_Format'] = stats_entites['Perte_Totale'].apply(formater_montant)
        
        # Tri par nombre d'incidents
        stats_entites = stats_entites.sort_values('Nb_Incidents', ascending=False)
        
        # Vérification de cohérence des montants
        total_calcule = np.sum(severites)
        total_par_entites = stats_entites['Perte_Totale'].sum()
        difference = abs(total_calcule - total_par_entites)
        
        if difference < 0.01:  # Tolérance pour les erreurs d'arrondi
            st.success(f"✅ **Vérification cohérence :** Total général = Somme par entités = {formater_montant(total_calcule)}")
        else:
            st.warning(f"⚠️ **Incohérence détectée :** Total général ({formater_montant(total_calcule)}) ≠ Somme entités ({formater_montant(total_par_entites)})")
    else:
        st.error("❌ Impossible de calculer les statistiques par entité")
    
    # Affichage du tableau avec formatage
    if stats_entites is not None:
        colonnes_affichage = [
            'Entite_Court', 'Nb_Incidents', 'Frequence_Annuelle', 
            'Severite_Moyenne_Format', 'Severite_Mediane_Format', 'Severite_Max_Format',
            'Part_Incidents_%', 'Part_Pertes_%', 'CV_Severite',
            'Categorie_Principale', 'Gravite_Principale'
        ]
    
        df_display = stats_entites[colonnes_affichage].copy()
        df_display = df_display.rename(columns={
            'Entite_Court': 'Entité',
            'Severite_Moyenne_Format': 'Sévérité Moyenne',
            'Severite_Mediane_Format': 'Sévérité Médiane',
            'Severite_Max_Format': 'Sévérité Max',
            'Part_Incidents_%': 'Part Incidents (%)',
            'Part_Pertes_%': 'Part Pertes (%)'
        })
        
        st.dataframe(
            df_display.style.format({
                'Part Incidents (%)': '{:.1f}%',
                'Part Pertes (%)': '{:.1f}%',
                'Frequence_Annuelle': '{:.1f}',
                'CV_Severite': '{:.2f}'
            }),
            use_container_width=True
        )
    
        # Graphiques par entité
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 10 entités par nombre d'incidents
            top_entites = stats_entites.head(10)
            entites_courtes = [nom[:25] + '...' if len(nom) > 25 else nom for nom in top_entites['Entite_Court']]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(entites_courtes)), top_entites['Nb_Incidents'])
            ax.set_xticks(range(len(entites_courtes)))
            ax.set_xticklabels(entites_courtes, rotation=45, ha='right')
            ax.set_title("Top 10 - Nombre d'Incidents par Entité")
            ax.set_ylabel("Nombre d'Incidents")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            # Top 10 entités par pertes totales
            fig, ax = plt.subplots(figsize=(10, 6))
            pertes_millions = top_entites['Perte_Totale'] / 1_000_000  # Conversion en millions pour l'affichage
            ax.bar(range(len(entites_courtes)), pertes_millions)
            ax.set_xticks(range(len(entites_courtes)))
            ax.set_xticklabels(entites_courtes, rotation=45, ha='right')
            ax.set_title("Top 10 - Pertes Totales par Entité")
            ax.set_ylabel("Pertes Totales (Millions FCFA)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.warning("⚠️ Impossible d'afficher les graphiques par entité")
    
    # =================== 4. DISTRIBUTION PAR CATÉGORIE DE RISQUE ===================
    
    st.header("2️⃣ Distribution des Incidents par Catégorie de Risque")
    
    st.subheader("📋 Analyse par Catégorie de Risque")
    
    # Calcul des statistiques par catégorie avec formatage
    def calcul_stats_categories_avec_formatage(df):
        """Calcul des statistiques par catégorie avec formatage approprié"""
        try:
            stats_list = []
            for categorie in df['Categorie_Risque'].unique():
                subset = df[df['Categorie_Risque'] == categorie]
                severites_cat = subset['Severite'].values  # Déjà en FCFA réels
                
                if len(severites_cat) > 0:
                    stats_cat = {
                        'Categorie': categorie,
                        'Nb_Incidents': len(severites_cat),
                        'Severite_Moyenne': np.mean(severites_cat),
                        'Severite_Mediane': np.median(severites_cat),
                        'Perte_Totale': np.sum(severites_cat),
                        'Ecart_Type': np.std(severites_cat, ddof=1) if len(severites_cat) > 1 else 0,
                        'Nb_Entites': len(subset['Entite'].unique())
                    }
                    stats_list.append(stats_cat)
            
            df_stats = pd.DataFrame(stats_list).set_index('Categorie')
            
            # Calculs dérivés
            total_incidents = len(df)
            total_pertes = df['Severite'].sum()
            df_stats['Part_Incidents_%'] = (df_stats['Nb_Incidents'] / total_incidents * 100).round(1)
            df_stats['Part_Pertes_%'] = (df_stats['Perte_Totale'] / total_pertes * 100).round(1)
            
            # Ajout des colonnes formatées
            df_stats['Severite_Moyenne_Format'] = df_stats['Severite_Moyenne'].apply(formater_montant)
            df_stats['Severite_Mediane_Format'] = df_stats['Severite_Mediane'].apply(formater_montant)
            df_stats['Perte_Totale_Format'] = df_stats['Perte_Totale'].apply(formater_montant)
            
            return df_stats.sort_values('Nb_Incidents', ascending=False)
            
        except Exception as e:
            st.error(f"Erreur dans calcul_stats_categories_avec_formatage : {e}")
            return None
    
    stats_categories = calcul_stats_categories_avec_formatage(df)
    
    # Définition des noms complets des catégories
    if stats_categories is not None:
        categories_noms = {
            'RF': 'Risques de Fraude',
            'RSI': 'Risques Systèmes d\'Information',
            'RH': 'Risques Humains',
            'RC': 'Risques de Crédit',
            'RO': 'Risques Opérationnels',
            'RM': 'Risques de Marché'
        }
        
        stats_categories['Nom_Complet'] = [categories_noms.get(cat, cat) for cat in stats_categories.index]
    
    # Affichage avec formatage
    colonnes_cat_display = ['Nom_Complet', 'Nb_Incidents', 'Severite_Moyenne_Format', 'Severite_Mediane_Format', 
                           'Perte_Totale_Format', 'Part_Incidents_%', 'Part_Pertes_%', 'Nb_Entites']
    
    df_cat_display = stats_categories[colonnes_cat_display].copy()
    df_cat_display = df_cat_display.rename(columns={
        'Nom_Complet': 'Catégorie',
        'Severite_Moyenne_Format': 'Sévérité Moyenne',
        'Severite_Mediane_Format': 'Sévérité Médiane',
        'Perte_Totale_Format': 'Perte Totale',
        'Part_Incidents_%': 'Part Incidents (%)',
        'Part_Pertes_%': 'Part Pertes (%)',
        'Nb_Entites': 'Nb Entités'
    })
    
    st.dataframe(
        df_cat_display.style.format({
            'Part Incidents (%)': '{:.1f}%',
            'Part Pertes (%)': '{:.1f}%'
        }),
        use_container_width=True
    )
    
    # Graphiques par catégorie
    if stats_categories is not None and len(stats_categories) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Répartition des incidents par catégorie
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(stats_categories['Nb_Incidents'], labels=stats_categories['Nom_Complet'], autopct='%1.1f%%')
            ax.set_title("Répartition des Incidents par Catégorie de Risque")
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            # Sévérité moyenne par catégorie
            fig, ax = plt.subplots(figsize=(8, 6))
            severites_millions = stats_categories['Severite_Moyenne'] / 1_000_000
            ax.bar(range(len(stats_categories)), severites_millions)
            ax.set_xticks(range(len(stats_categories)))
            ax.set_xticklabels(stats_categories['Nom_Complet'], rotation=45, ha='right')
            ax.set_title("Sévérité Moyenne par Catégorie de Risque")
            ax.set_ylabel("Sévérité Moyenne (Millions FCFA)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.warning("⚠️ Impossible d'afficher les graphiques par catégorie")
    
    # =================== 5. ANALYSE PAR GRAVITÉ ===================
    
    st.subheader("⚡ Analyse par Niveau de Gravité")
    
    # Calcul des statistiques par gravité avec formatage
    def calcul_stats_gravites_avec_formatage(df):
        """Calcul des statistiques par gravité avec formatage approprié"""
        try:
            stats_list = []
            for gravite in df['Gravite'].unique():
                if pd.isna(gravite) or gravite == 'nan':
                    continue
                    
                subset = df[df['Gravite'] == gravite]
                severites_grav = subset['Severite'].values  # Déjà en FCFA réels
                
                if len(severites_grav) > 0:
                    stats_grav = {
                        'Gravite': gravite,
                        'Nb_Incidents': len(severites_grav),
                        'Severite_Moyenne': np.mean(severites_grav),
                        'Severite_Mediane': np.median(severites_grav),
                        'Perte_Totale': np.sum(severites_grav),
                        'Nb_Entites': len(subset['Entite'].unique())
                    }
                    stats_list.append(stats_grav)
            
            df_stats = pd.DataFrame(stats_list).set_index('Gravite')
            
            # Calculs dérivés
            total_incidents = len(df)
            total_pertes = df['Severite'].sum()
            df_stats['Part_Incidents_%'] = (df_stats['Nb_Incidents'] / total_incidents * 100).round(1)
            df_stats['Part_Pertes_%'] = (df_stats['Perte_Totale'] / total_pertes * 100).round(1)
            
            # Ajout des colonnes formatées
            df_stats['Severite_Moyenne_Format'] = df_stats['Severite_Moyenne'].apply(formater_montant)
            df_stats['Severite_Mediane_Format'] = df_stats['Severite_Mediane'].apply(formater_montant)
            df_stats['Perte_Totale_Format'] = df_stats['Perte_Totale'].apply(formater_montant)
            
            # Ordre logique des gravités
            ordre_gravite = ['Très faible', 'Faible', 'Moyen', 'Fort', 'Très fort']
            gravites_presentes = [g for g in ordre_gravite if g in df_stats.index]
            
            return df_stats.reindex(gravites_presentes)
            
        except Exception as e:
            st.error(f"Erreur dans calcul_stats_gravites_avec_formatage : {e}")
            return None
    
    stats_gravites = calcul_stats_gravites_avec_formatage(df)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if stats_gravites is not None and len(stats_gravites) > 0:
            # Affichage avec formatage
            colonnes_grav_display = ['Nb_Incidents', 'Severite_Moyenne_Format', 'Severite_Mediane_Format',
                                   'Perte_Totale_Format', 'Part_Incidents_%', 'Part_Pertes_%', 'Nb_Entites']
            
            df_grav_display = stats_gravites[colonnes_grav_display].copy()
            df_grav_display = df_grav_display.rename(columns={
                'Severite_Moyenne_Format': 'Sévérité Moyenne',
                'Severite_Mediane_Format': 'Sévérité Médiane',
                'Perte_Totale_Format': 'Perte Totale',
                'Part_Incidents_%': 'Part Incidents (%)',
                'Part_Pertes_%': 'Part Pertes (%)',
                'Nb_Entites': 'Nb Entités'
            })
            
            st.dataframe(
                df_grav_display.style.format({
                    'Part Incidents (%)': '{:.1f}%',
                    'Part Pertes (%)': '{:.1f}%'
                }),
                use_container_width=True
            )
        else:
            st.error("❌ Impossible de calculer les statistiques par gravité")
    
    with col2:
        if stats_gravites is not None and len(stats_gravites) > 0:
            # Graphique gravité vs sévérité
            fig, ax = plt.subplots(figsize=(8, 6))
            severites_millions = stats_gravites['Severite_Moyenne'] / 1_000_000
            scatter = ax.scatter(range(len(stats_gravites.index)), severites_millions, 
                               s=stats_gravites['Nb_Incidents']*10, alpha=0.6)
            ax.set_xticks(range(len(stats_gravites.index)))
            ax.set_xticklabels(stats_gravites.index, rotation=45)
            ax.set_title("Gravité vs Sévérité Moyenne")
            ax.set_xlabel("Niveau de Gravité")
            ax.set_ylabel("Sévérité Moyenne (Millions FCFA)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("⚠️ Impossible d'afficher le graphique par gravité")
    
    # =================== 6. MATRICE DE CORRÉLATION ===================
    
    st.subheader("🔗 Matrice Catégorie × Entité")
    
    # Heatmap des incidents par catégorie et entité (top 10 entités)
    if stats_entites is not None and len(stats_entites) > 0:
        top_10_entites = stats_entites.head(10).index.tolist()
        df_top = df[df['Entite'].isin(top_10_entites)]
        
        # Noms courts pour l'affichage
        df_top_display = df_top.copy()
        df_top_display['Entite_Court'] = [nom.replace('DGEFRI-DOF : ', '') for nom in df_top_display['Entite']]
        
        try:
            matrice_cat_entite = pd.crosstab(df_top_display['Categorie_Risque'], df_top_display['Entite_Court'])
            
            # Heatmap avec seaborn
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(matrice_cat_entite, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
            ax.set_title("Heatmap Incidents: Catégorie × Entité (Top 10)")
            ax.set_xlabel("Entité")
            ax.set_ylabel("Catégorie de Risque")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"⚠️ Impossible d'afficher la heatmap: {e}")
    else:
        st.warning("⚠️ Impossible d'afficher la matrice de corrélation")
    
    # =================== 7. ANALYSE DES TENDANCES TEMPORELLES ===================
    
    st.header("3️⃣ Analyse des Tendances Temporelles")
    
    # Évolution annuelle
    st.subheader("📅 Évolution Annuelle")
    
    # Calcul des tendances annuelles avec formatage
    def calcul_trends_annuelles_avec_formatage(df):
        """Calcul des tendances annuelles avec formatage approprié"""
        try:
            trends_list = []
            for annee in sorted(df['Annee'].unique()):
                subset = df[df['Annee'] == annee]
                severites_annee = subset['Severite'].values  # Déjà en FCFA réels
                
                if len(severites_annee) > 0:
                    trend_annee = {
                        'Annee': annee,
                        'Nb_Incidents': len(severites_annee),
                        'Pertes_Totales': np.sum(severites_annee),
                        'Severite_Moyenne': np.mean(severites_annee),
                        'Severite_Mediane': np.median(severites_annee)
                    }
                    trends_list.append(trend_annee)
            
            df_trends = pd.DataFrame(trends_list).set_index('Annee')
            
            # Ajout des colonnes formatées
            df_trends['Pertes_Totales_Format'] = df_trends['Pertes_Totales'].apply(formater_montant)
            df_trends['Severite_Moyenne_Format'] = df_trends['Severite_Moyenne'].apply(formater_montant)
            df_trends['Severite_Mediane_Format'] = df_trends['Severite_Mediane'].apply(formater_montant)
            
            return df_trends
            
        except Exception as e:
            st.error(f"Erreur dans calcul_trends_annuelles_avec_formatage : {e}")
            return None
    
    trends_annuelles = calcul_trends_annuelles_avec_formatage(df)
    
    # Graphiques temporels
    if trends_annuelles is not None and len(trends_annuelles) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Évolution du nombre d'incidents
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(trends_annuelles.index, trends_annuelles['Nb_Incidents'], marker='o')
            ax.set_title("Évolution du Nombre d'Incidents par Année")
            ax.set_xlabel("Année")
            ax.set_ylabel("Nombre d'Incidents")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            # Évolution des pertes totales
            fig, ax = plt.subplots(figsize=(10, 6))
            pertes_millions = trends_annuelles['Pertes_Totales'] / 1_000_000
            ax.plot(trends_annuelles.index, pertes_millions, marker='o', color='red')
            ax.set_title("Évolution des Pertes Totales par Année")
            ax.set_xlabel("Année")
            ax.set_ylabel("Pertes Totales (Millions FCFA)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.warning("⚠️ Impossible d'afficher les tendances annuelles")
    
    # Analyse mensuelle (saisonnalité)
    st.subheader("🌓 Analyse de Saisonnalité")
    
    # Calcul des tendances mensuelles avec formatage
    def calcul_trends_mensuelles_avec_formatage(df):
        """Calcul des tendances mensuelles avec formatage approprié"""
        try:
            trends_list = []
            for mois in range(1, 13):
                subset = df[df['Mois'] == mois]
                severites_mois = subset['Severite'].values  # Déjà en FCFA réels
                
                if len(severites_mois) > 0:
                    # Moyenne des incidents par mois sur toutes les années
                    nb_annees_mois = len(subset['Annee'].unique())
                    nb_incidents_moyen = len(severites_mois) / nb_annees_mois if nb_annees_mois > 0 else 0
                    
                    trend_mois = {
                        'Mois': mois,
                        'Nb_Incidents_Moyen': nb_incidents_moyen,
                        'Severite_Moyenne': np.mean(severites_mois)
                    }
                else:
                    trend_mois = {
                        'Mois': mois,
                        'Nb_Incidents_Moyen': 0,
                        'Severite_Moyenne': 0
                    }
                trends_list.append(trend_mois)
            
            return pd.DataFrame(trends_list).set_index('Mois')
            
        except Exception as e:
            st.error(f"Erreur dans calcul_trends_mensuelles_avec_formatage : {e}")
            return None
    
    trends_mensuelles = calcul_trends_mensuelles_avec_formatage(df)
    
    # Graphique saisonnalité
    if trends_mensuelles is not None and len(trends_mensuelles) > 0:
        mois_noms = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 
                    'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Mois')
        ax1.set_ylabel('Nb Incidents Moyens', color=color)
        ax1.plot(mois_noms, trends_mensuelles['Nb_Incidents_Moyen'], color=color, marker='o', label='Incidents Moyens')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_title("Saisonnalité des Incidents et Sévérités")
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Sévérité Moyenne (Millions FCFA)', color=color)
        severites_moy_millions = trends_mensuelles['Severite_Moyenne'] / 1_000_000
        ax2.plot(mois_noms, severites_moy_millions, color=color, marker='s', label='Sévérité Moyenne')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("⚠️ Impossible d'afficher l'analyse de saisonnalité")
    
    # Tests de tendance
    if trends_annuelles is not None and len(trends_annuelles) > 0:
        st.subheader("📈 Tests Statistiques de Tendance")
        
        # Test de Mann-Kendall pour tendance
        from scipy.stats import kendalltau
        
        try:
            annees = trends_annuelles.index.values
            incidents = trends_annuelles['Nb_Incidents'].values
            pertes = trends_annuelles['Pertes_Totales'].values
            
            tau_incidents, p_incidents = kendalltau(annees, incidents)
            tau_pertes, p_pertes = kendalltau(annees, pertes)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Tendance Incidents",
                    f"τ = {tau_incidents:.3f}",
                    f"p = {p_incidents:.3f} {'📈' if tau_incidents > 0 else '📉' if tau_incidents < 0 else '➡️'}"
                )
            
            with col2:
                st.metric(
                    "Tendance Pertes",
                    f"τ = {tau_pertes:.3f}",
                    f"p = {p_pertes:.3f} {'📈' if tau_pertes > 0 else '📉' if tau_pertes < 0 else '➡️'}"
                )
            
            # Interprétation
            interpretation_incidents = "Tendance croissante significative" if p_incidents < 0.05 and tau_incidents > 0 else \
                                    "Tendance décroissante significative" if p_incidents < 0.05 and tau_incidents < 0 else \
                                    "Pas de tendance significative"
            
            interpretation_pertes = "Tendance croissante significative" if p_pertes < 0.05 and tau_pertes > 0 else \
                                  "Tendance décroissante significative" if p_pertes < 0.05 and tau_pertes < 0 else \
                                  "Pas de tendance significative"
            
            st.info(f"""
            **🔍 Interprétation des Tendances :**
            - **Incidents :** {interpretation_incidents}
            - **Pertes :** {interpretation_pertes}
            """)
        except Exception as e:
            st.warning(f"⚠️ Impossible de calculer les tests de tendance: {e}")
    else:
        st.warning("⚠️ Pas de données pour les tests de tendance")
    
    # =================== 8. STATISTIQUES DE SÉVÉRITÉ AVANCÉES ===================
    
    st.header("4️⃣ Statistiques de Sévérité Détaillées")
    
    # Statistiques descriptives complètes
    st.subheader("📊 Statistiques Descriptives Complètes")
    
    def calculer_stats_completes(data):
        """Calcul de toutes les statistiques descriptives"""
        try:
            # Conversion en array numpy pour éviter les problèmes d'Index
            data_array = np.array(data).flatten()
            data_clean = data_array[~np.isnan(data_array)]
            
            if len(data_clean) == 0:
                return {'Erreur': 'Pas de données valides'}
            
            # Calcul du mode de façon sécurisée
            try:
                mode_result = stats.mode(data_clean, keepdims=False)
                mode_val = float(mode_result.mode) if hasattr(mode_result, 'mode') else float(mode_result[0])
            except:
                mode_val = np.nan
            
            return {
                'Observations': int(len(data_clean)),
                'Moyenne': float(np.mean(data_clean)),
                'Médiane': float(np.median(data_clean)),
                'Mode': mode_val,
                'Écart-Type': float(np.std(data_clean, ddof=1)),
                'Variance': float(np.var(data_clean, ddof=1)),
                'CV (%)': float((np.std(data_clean, ddof=1) / np.mean(data_clean)) * 100),
                'Minimum': float(np.min(data_clean)),
                'Q1 (25%)': float(np.percentile(data_clean, 25)),
                'Q3 (75%)': float(np.percentile(data_clean, 75)),
                'Maximum': float(np.max(data_clean)),
                'IQR': float(np.percentile(data_clean, 75) - np.percentile(data_clean, 25)),
                'Étendue': float(np.max(data_clean) - np.min(data_clean)),
                'Asymétrie': float(stats.skew(data_clean)),
                'Aplatissement': float(stats.kurtosis(data_clean)),
                'Erreur_Standard': float(stats.sem(data_clean))
            }
        except Exception as e:
            st.error(f"Erreur dans le calcul des statistiques : {e}")
            return {'Erreur': str(e)}
    
    # Calcul pour toutes les données
    stats_globales = calculer_stats_completes(severites)
    
    # Affichage en colonnes avec formatage
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🎯 Tendance Centrale**")
        # Affichage intelligent selon la taille des montants
        moyenne_val = stats_globales['Moyenne']
        mediane_val = stats_globales['Médiane'] 
        mode_val = stats_globales['Mode']
        
        if moyenne_val >= 1_000_000_000:
            st.metric("Moyenne", f"{moyenne_val/1_000_000_000:.2f}", delta="Milliards FCFA")
        else:
            st.metric("Moyenne", f"{moyenne_val/1_000_000:.1f}", delta="Millions FCFA")
            
        if mediane_val >= 1_000_000_000:
            st.metric("Médiane", f"{mediane_val/1_000_000_000:.2f}", delta="Milliards FCFA")
        else:
            st.metric("Médiane", f"{mediane_val/1_000_000:.1f}", delta="Millions FCFA")
            
        if mode_val >= 1_000_000_000:
            st.metric("Mode", f"{mode_val/1_000_000_000:.2f}", delta="Milliards FCFA")
        else:
            st.metric("Mode", f"{mode_val/1_000_000:.1f}", delta="Millions FCFA")
        
    with col2:
        st.markdown("**📏 Dispersion**")
        ecart_type_val = stats_globales['Écart-Type']
        iqr_val = stats_globales['IQR']
        erreur_std_val = stats_globales['Erreur_Standard']
        
        if ecart_type_val >= 1_000_000_000:
            st.metric("Écart-Type", f"{ecart_type_val/1_000_000_000:.2f}", delta="Milliards FCFA")
        else:
            st.metric("Écart-Type", f"{ecart_type_val/1_000_000:.1f}", delta="Millions FCFA")
            
        st.metric("CV (%)", f"{stats_globales['CV (%)']:,.1f}%")
        
        if iqr_val >= 1_000_000_000:
            st.metric("IQR", f"{iqr_val/1_000_000_000:.2f}", delta="Milliards FCFA")
        else:
            st.metric("IQR", f"{iqr_val/1_000_000:.1f}", delta="Millions FCFA")
        
    with col3:
        st.markdown("**📐 Forme**")
        st.metric("Asymétrie", f"{stats_globales['Asymétrie']:,.2f}")
        st.metric("Aplatissement", f"{stats_globales['Aplatissement']:,.2f}")
        
        if erreur_std_val >= 1_000_000_000:
            st.metric("Erreur Standard", f"{erreur_std_val/1_000_000_000:.2f}", delta="Milliards FCFA")
        else:
            st.metric("Erreur Standard", f"{erreur_std_val/1_000_000:.1f}", delta="Millions FCFA")
    
    # Interprétation des statistiques
    st.subheader("🔍 Interprétation des Statistiques")
    
    asymetrie = stats_globales['Asymétrie']
    aplatissement = stats_globales['Aplatissement']
    cv = stats_globales['CV (%)']
    
    # Analyse de l'asymétrie
    if asymetrie > 2:
        interp_asym = "Très fortement asymétrique à droite - Distribution très déformée"
    elif asymetrie > 1:
        interp_asym = "Fortement asymétrique à droite - Concentration sur faibles valeurs"
    elif asymetrie > 0.5:
        interp_asym = "Modérément asymétrique à droite - Légère concentration sur faibles valeurs"
    elif asymetrie > -0.5:
        interp_asym = "Quasi-symétrique - Distribution équilibrée"
    else:
        interp_asym = "Asymétrique à gauche - Concentration sur fortes valeurs"
    
    # Analyse de l'aplatissement
    if aplatissement > 3:
        interp_aplat = "Queues très lourdes - Événements extrêmes fréquents"
    elif aplatissement > 0:
        interp_aplat = "Queues lourdes - Plus d'événements extrêmes que la normale"
    elif aplatissement > -1:
        interp_aplat = "Queues normales - Distribution classique"
    else:
        interp_aplat = "Queues légères - Peu d'événements extrêmes"
    
    # Analyse du coefficient de variation
    if cv > 100:
        interp_cv = "Très forte variabilité - Données très dispersées"
    elif cv > 50:
        interp_cv = "Forte variabilité - Données dispersées"
    elif cv > 25:
        interp_cv = "Variabilité modérée - Dispersion acceptable"
    else:
        interp_cv = "Faible variabilité - Données homogènes"
    
    st.info(f"""
    **📋 Caractéristiques de la Distribution :**
    
    **🔸 Asymétrie ({asymetrie:.2f}):** {interp_asym}
    
    **🔸 Aplatissement ({aplatissement:.2f}):** {interp_aplat}
    
    **🔸 Coefficient de Variation ({cv:.1f}%):** {interp_cv}
    
    **🎯 Conclusion :** Cette distribution est typique du risque opérationnel avec de nombreuses petites pertes et quelques pertes extrêmes.
    """)
    
    # Percentiles de risque
    st.subheader("📈 Analyse des Percentiles de Risque")
    
    percentiles = [50, 75, 80, 85, 90, 95, 97.5, 99, 99.5, 99.9]
    valeurs_percentiles = [np.percentile(severites, p) for p in percentiles]
    
    df_percentiles = pd.DataFrame({
        'Percentile': [f"{p}%" for p in percentiles],
        'Valeur': valeurs_percentiles,
        'VaR': [f"VaR_{p}%" for p in percentiles],
        'Dépassement_%': [100-p for p in percentiles]
    })
    
    # Formatage
    df_percentiles['Valeur_Formatée'] = df_percentiles['Valeur'].apply(formater_montant)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(
            df_percentiles[['Percentile', 'Valeur_Formatée', 'Dépassement_%']],
            use_container_width=True
        )
    
    with col2:
        # Graphique des percentiles
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(percentiles, valeurs_percentiles, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Percentiles (%)')
        ax.set_ylabel('Sévérité FCFA (échelle log)')
        ax.set_title('Courbe des Percentiles (VaR)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    # Distribution et histogramme
    st.subheader("📊 Distribution des Sévérités")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogramme
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(severites, bins=50, alpha=0.7, edgecolor='black')
        ax.set_title("Histogramme des Sévérités")
        ax.set_xlabel("Sévérité (FCFA)")
        ax.set_ylabel("Fréquence")
        ax.grid(True, alpha=0.3)
        
        # Formatage de l'axe x en notation scientifique
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        # Box plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(severites)
        ax.set_title("Box Plot des Sévérités")
        ax.set_ylabel("Sévérité (FCFA)")
        ax.grid(True, alpha=0.3)
        
        # Formatage de l'axe y en notation scientifique
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    # QQ-Plot et tests de normalité
    st.subheader("🧪 Tests de Normalité et QQ-Plots")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tests de normalité
        try:
            # Shapiro-Wilk (échantillon si trop grand)
            sample_size = min(5000, len(severites))
            sample = np.random.choice(severites, sample_size, replace=False)
            _, p_shapiro = stats.shapiro(sample)
            
            # D'Agostino
            _, p_dagostino = stats.normaltest(severites)
            
            # Anderson-Darling
            ad_stat, ad_crit, ad_sig = stats.anderson(severites, dist='norm')
            p_anderson = 0.05 if ad_stat > ad_crit[2] else 0.10
            
            tests_normalite = pd.DataFrame({
                'Test': ['Shapiro-Wilk', 'D\'Agostino-Pearson', 'Anderson-Darling'],
                'Statistique': ['-', f"{stats.normaltest(severites)[0]:.3f}", f"{ad_stat:.3f}"],
                'P-Value': [f"{p_shapiro:.6f}", f"{p_dagostino:.6f}", f"{p_anderson:.3f}"],
                'Conclusion': [
                    'Normale' if p_shapiro > 0.05 else 'Non-Normale',
                    'Normale' if p_dagostino > 0.05 else 'Non-Normale',
                    'Normale' if p_anderson > 0.05 else 'Non-Normale'
                ]
            })
            
            st.dataframe(tests_normalite, use_container_width=True)
            
        except:
            st.warning("Tests de normalité non disponibles pour ces données")
    
    with col2:
        # QQ-Plot normal
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            stats.probplot(severites, dist="norm", plot=ax)
            ax.set_title("QQ-Plot vs Distribution Normale")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"QQ-Plot non disponible: {e}")
    
    # Résumé final avec formatage
    st.subheader("📋 Résumé Exécutif")
    
    # Calculs sécurisés pour le résumé
    try:
        top_entite = stats_entites.index[0].replace('DGEFRI-DOF : ', '') if stats_entites is not None and len(stats_entites) > 0 else "Non disponible"
        top_categorie = stats_categories.index[0] if stats_categories is not None and len(stats_categories) > 0 else "Non disponible"
        part_top_cat = stats_categories.iloc[0]['Part_Incidents_%'] if stats_categories is not None and len(stats_categories) > 0 else 0
        
        annee_critique = "Non disponible"
        incidents_max = 0
        if trends_annuelles is not None and len(trends_annuelles) > 0:
            annee_critique = trends_annuelles['Nb_Incidents'].idxmax()
            incidents_max = trends_annuelles['Nb_Incidents'].max()
        
        st.success(f"""
        **🎯 SYNTHÈSE DE L'ANALYSE EXPLORATOIRE**
        
        **📊 Volume des Données :**
        - {len(severites):,} incidents valides sur {nb_annees:.1f} années
        - {df['Entite'].nunique()} entités différentes
        - {df['Categorie_Risque'].nunique()} catégories de risque
        
        **💰 Profil de Risque :**
        - Perte moyenne : {formater_montant(stats_globales['Moyenne'])}
        - Perte médiane : {formater_montant(stats_globales['Médiane'])}
        - Perte maximale : {formater_montant(stats_globales['Maximum'])}
        - VaR 95% : {formater_montant(np.percentile(severites, 95))}
        
        **📈 Caractéristiques Clés :**
        - Distribution asymétrique à droite (skewness = {asymetrie:.2f})
        - Queues lourdes (kurtosis = {aplatissement:.2f})
        - Forte variabilité (CV = {cv:.1f}%)
        - Distribution typique du risque opérationnel
        
        **🏆 Top Risques :**
        - Entité la plus impactée : {top_entite}
        - Catégorie principale : {top_categorie} ({part_top_cat:.1f}% des incidents)
        - Période critique : {annee_critique} ({incidents_max} incidents)
        
        **✅ Recommandations :**
        - Utiliser des distributions spécialisées (Weibull, Log-Normale)
        - Approche LDA pour modélisation fréquence/sévérité
        - Focus sur la gestion des risques extrêmes
        - Surveillance renforcée des entités à fort impact
        """)
    except Exception as e:
        st.error(f"❌ Erreur dans le résumé exécutif: {e}")
        st.info("Les données de base sont disponibles, mais certains calculs dérivés ont échoué.")
    
    return df, stats_globales

# =================== INTERFACE PRINCIPALE ===================

def main():
    """Interface principale de l'application"""
    
    st.sidebar.title("🎛️ Configuration")
    st.sidebar.markdown("---")
    
    # Options d'affichage
    show_details = st.sidebar.checkbox("Affichage détaillé", value=True)
    show_graphs = st.sidebar.checkbox("Graphiques avancés", value=True)
    
    # Bouton de lancement
    if st.sidebar.button("🚀 LANCER L'ANALYSE", type="primary"):
        with st.spinner("Analyse en cours..."):
            try:
                df, stats = analyse_exploratoire_complete()
                
                if df is not None:
                    st.balloons()
                    st.sidebar.success("✅ Analyse terminée !")
                    
                    # Téléchargement des résultats
                    if st.sidebar.button("📥 Télécharger Rapport"):
                        # Conversion en CSV pour téléchargement
                        csv = df.to_csv(index=False)
                        st.sidebar.download_button(
                            label="📄 Télécharger CSV",
                            data=csv,
                            file_name=f"analyse_oprisk_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse: {e}")
                st.info("🔍 Vérifiez que le fichier base_incidents.xlsx est présent dans le bon format")
    else:
        st.info("👈 Utilisez le panneau latéral pour lancer l'analyse")
        
        # Informations sur l'application
        st.markdown("""
        ## 📋 À propos de cette analyse
        
        Cette application fournit une **analyse exploratoire complète** des données de risque opérationnel, avec affichage des vraies valeurs en FCFA :
        
        **🗂️ Structure de Données Détectée :**
        - Code d'incident
        - Entité (DGEFRI-DOF)
        - Gravité (Très faible, Faible, Fort, etc.)
        - Catégorie de Risque (RF, RSI, RH, etc.)
        - Coût total estimé (converti automatiquement en FCFA réels)
        - Date de survenance (format DD-MM-YYYY)
        
        **💰 Formatage Intelligent :**
        - Affichage automatique en k FCFA, M FCFA, ou Md FCFA
        - Conversion transparente des données stockées en unités de 10 000 FCFA
        - Graphiques et tableaux avec vraies valeurs
        
        **1️⃣ Caractéristiques Générales**
        - Vue d'ensemble des données avec vraies valeurs
        - Statistiques détaillées par entité
        - Métriques de concentration du risque
        
        **2️⃣ Distribution par Catégorie et Gravité**
        - Analyse par catégorie de risque (RF, RSI, RH, etc.)
        - Répartition par niveau de gravité
        - Matrices de corrélation
        
        **3️⃣ Tendances Temporelles**
        - Évolution annuelle et mensuelle
        - Tests de saisonnalité
        - Détection statistique de tendances
        
        **4️⃣ Statistiques de Sévérité**
        - Statistiques descriptives complètes avec formatage
        - Percentiles de risque (VaR) en vraies valeurs
        - Tests de normalité et QQ-Plots
        
        **📊 Données Attendues :**
        - Fichier : `base_incidents.xlsx`
        - Feuille : `Incidents_DOF_augmente`
        - Format détecté automatiquement
        - Conversion automatique en FCFA réels
        """)

if __name__ == "__main__":
    main()