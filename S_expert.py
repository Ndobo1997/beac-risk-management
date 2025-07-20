import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import torch

# Suppression des warnings TensorFlow et PyTorch
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TORCH_HOME'] = './torch_cache'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class EnhancedOpRiskABC:
    def __init__(self):
        self.salaires_beac = {
            'aes': 800_000,
            'aem': 500_000,
            'accg': 300_000
        }
        self.charges_sociales = 0.12
        self.heures_reelles_mois = 185
        self.indirect_factor_enhanced = 1.35
        self.cout_caches_beac = {
            'frais_medicaux': 0.048,
            'fonds_special': 0.032,
            'allocations_familiales': 0.075,
            'formation_continue': 0.025,
            'anciennete_premium': 0.020
        }
        self.facteurs_satisfaction = {
            'aes': 0.92,
            'aem': 1.00,
            'accg': 1.15
        }
        self.facteurs_anciennete = {
            'aes': 1.18,
            'aem': 1.10,
            'accg': 1.05
        }

    def calculate_enhanced_hourly_rates(self):
        enhanced_rates = {}
        for role, salaire_base in self.salaires_beac.items():
            cout_horaire_base = (salaire_base * (1 + self.charges_sociales)) / self.heures_reelles_mois
            facteur_caches = sum(self.cout_caches_beac.values())
            cout_horaire_ajuste = cout_horaire_base * (1 + facteur_caches)
            cout_final = cout_horaire_ajuste * self.facteurs_anciennete[role]
            enhanced_rates[role] = cout_final
        return enhanced_rates

    def classify_incident_enhanced(self, row):
        entite = str(row.get('Entité', '')).lower()
        gravite = str(row.get('Gravité', '')).lower()
        description = str(row.get('Description', '')).lower()
        ligne_metier = str(row.get('Ligne Metier', '')).lower()
        
        if 'comptabilité' in ligne_metier:
            personnel = 'accg'
        elif 'confirmation, règlements et livraison' in ligne_metier:
            if any(term in description for term in ['complexe', 'analyse', 'validation', 'swift']):
                personnel = 'aem'
            else:
                personnel = 'accg'
        elif 'conformité' in ligne_metier or 'risques' in ligne_metier:
            personnel = 'aes'
        elif 'études et stratégies' in ligne_metier:
            personnel = 'aes'
        elif 'trésorerie' in ligne_metier:
            personnel = 'aem'
        else:
            if 'direction' in entite or 'gouverneur' in entite:
                personnel = 'aes'
            elif 'trésorerie' in entite or 'risques de marchés' in entite:
                personnel = 'aem'
            elif 'comptabilité' in entite or 'règlements' in entite:
                if any(term in description for term in ['complexe', 'analyse', 'validation']):
                    personnel = 'aem'
                else:
                    personnel = 'accg'
            elif 'conformité' in entite or 'contrôle' in entite:
                personnel = 'aes'
            elif 'ressources humaines' in entite:
                personnel = 'aem'
            else:
                personnel = 'accg'
        
        if any(g in gravite for g in ['fort', 'très fort', 'critique']):
            if personnel == 'accg':
                personnel = 'aem'
            elif personnel == 'aem' and 'très fort' in gravite:
                personnel = 'aes'
                
        return personnel

    def estimate_enhanced_costs(self, incidents_df):
        hourly_rates = self.calculate_enhanced_hourly_rates()
        resultats_detailles = []
        for _, incident in incidents_df.iterrows():
            temps_reel_h = float(incident.get('Temps_H', 0)) if pd.notna(incident.get('Temps_H')) else 0
            personnel = self.classify_incident_enhanced(incident)
            cout_direct = temps_reel_h * hourly_rates[personnel]
            facteur_comportemental = self.facteurs_satisfaction[personnel]
            cout_ajuste_comportement = cout_direct * facteur_comportemental
            cout_total = cout_ajuste_comportement * self.indirect_factor_enhanced
            breakdown = {
                'Code': incident.get('Code'),
                'Entite': incident.get('Entité'),
                'Gravite': incident.get('Gravité'),
                'Temps_reel_h': round(temps_reel_h, 2),
                'Personnel_type': personnel,
                'Taux_horaire_enhanced': round(hourly_rates[personnel], 0),
                'Cout_direct': round(cout_direct, 0),
                'Facteur_satisfaction': facteur_comportemental,
                'Cout_ajuste_comportement': round(cout_ajuste_comportement, 0),
                'Cout_total_enhanced': round(cout_total, 0),
                'Ligne Metier': incident.get('Ligne Metier'),
                'Categorie_Risque': incident.get('Categorie_Risque', 'Non déterminée')
            }
            resultats_detailles.append(breakdown)
        return pd.DataFrame(resultats_detailles)

@st.cache_data
def load_data():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        training_path = os.path.join(script_dir, "data", "Base_entrainement.xlsx")
        recommendations_path = os.path.join(script_dir, "data", "Recommandations_risques_supplementaires_4.csv")
        if not os.path.exists(training_path):
            st.error(f"Fichier non trouvé : {training_path}")
            return None, None
        if not os.path.exists(recommendations_path):
            st.error(f"Fichier non trouvé : {recommendations_path}")
            return None, None
            
        incidents_df = pd.read_excel(training_path, sheet_name='Feuil1', engine='openpyxl')
        recommendations_df = pd.read_csv(recommendations_path, encoding='utf-8-sig')
        
        return incidents_df, recommendations_df
    except Exception as e:
        st.error(f"Erreur lors du chargement des fichiers : {e}")
        return None, None

@st.cache_resource
def load_model():
    try:
        # Suppression des logs TensorFlow temporairement
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.warning(f"Erreur lors du chargement du modèle principal : {e}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except Exception as e2:
            st.error(f"Erreur lors du chargement du modèle de fallback : {e2}")
            return None

def load_or_create_base_incidents():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    base_path = os.path.join(data_dir, 'base_incidents.xlsx')
    
    if os.path.exists(base_path):
        try:
            df = pd.read_excel(base_path, engine='openpyxl')
            return df
        except Exception as e:
            st.error(f"Erreur lors du chargement de la base : {e}")
            return create_empty_base()
    else:
        return create_empty_base()

def create_empty_base():
    columns = [
        'Code', 'Entité', 'Gravité', 'Temps_H', 'Catégorie_Risque', 
        'Cout_total_estime(en 10 000 FCFA)', 'Date de survenance'
    ]
    return pd.DataFrame(columns=columns)

def add_incident_to_base(new_incident_data):
    base_df = load_or_create_base_incidents()
    
    incident_row = {
        'Code': new_incident_data['Identifiant'],
        'Entité': new_incident_data['Ligne Metier'],
        'Gravité': new_incident_data['Niveau de gravité'],
        'Temps_H': new_incident_data.get('Temps_H', 0),
        'Catégorie_Risque': new_incident_data.get('Categorie_Risque_Code', 'RF'),
        'Cout_total_estime(en 10 000 FCFA)': round(new_incident_data['Coût total estimé (FCFA)'] / 10000, 0),
        'Date de survenance': new_incident_data['Ouverture'].split(' ')[0]
    }
    
    new_row_df = pd.DataFrame([incident_row])
    updated_base = pd.concat([base_df, new_row_df], ignore_index=True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, 'base_incidents.xlsx')
    updated_base.to_excel(base_path, index=False, engine='openpyxl')
    
    # Sauvegarde aussi du ticket complet avec les deux recommandations
    ticket_data = {
        'Identifiant': new_incident_data['Identifiant'],
        'Libellé': new_incident_data['Libellé'],
        'Ligne Metier': new_incident_data['Ligne Metier'],
        'Niveau de gravité': new_incident_data['Niveau de gravité'],
        'Ouverture': new_incident_data['Ouverture'],
        'Fin': new_incident_data['Fin'],
        'Recommandation_Principale': new_incident_data.get('Recommandation_1', ''),
        'Recommandation_Alternative': new_incident_data.get('Recommandation_2', ''),
        'Coût total estimé (FCFA)': new_incident_data['Coût total estimé (FCFA)'],
        'Personnel_type': new_incident_data['Personnel_type'],
        'Categorie_Risque': new_incident_data['Categorie_Risque']
    }
    
    # Sauvegarder le ticket individuel complet
    ticket_df = pd.DataFrame([ticket_data])
    ticket_file = os.path.join(script_dir, f"Ticket_{new_incident_data['Identifiant']}.csv")
    ticket_df.to_csv(ticket_file, index=False, encoding='utf-8-sig')
    
    return updated_base

def detect_service_from_description(description):
    description_lower = description.lower()
    
    services_keywords = {
        'DGEFRI-DOF : Comptabilité': [
            'comptabilité', 'comptable', 'écritures', 'bilan', 'comptes', 'journal',
            'balance', 'états financiers', 'provisions', 'amortissements'
        ],
        'DGEFRI-DOF : Confirmation, Règlements et Livraison': [
            'confirmation', 'règlements', 'livraison', 'swift', 'crl', 'règlement',
            'confirmation de paiement', 'livraison de titres', 'messagerie swift',
            'clearing', 'settlement', 'dénouement'
        ],
        'DGEFRI-DOF : Conformité, Risques de crédit et Risques Opérationnels': [
            'conformité', 'risques de crédit', 'risques opérationnels', 'compliance',
            'contrôle interne', 'audit', 'réglementation', 'supervision',
            'risque de crédit', 'risque opérationnel'
        ],
        'DGEFRI-DOF : Etudes et stratégies des marchés financiers': [
            'études', 'stratégies', 'marchés financiers', 'analyse financière',
            'recherche', 'stratégie', 'marché financier', 'analyse de marché',
            'veille financière', 'études économiques'
        ],
        'DGEFRI-DOF : Risques de Marchés et Performance': [
            'risques de marchés', 'performance', 'var', 'value at risk',
            'risque de marché', 'mesure de performance', 'indicateurs',
            'benchmarking', 'reporting performance'
        ],
        'DGEFRI-DOF : Trésorerie et Position de Change': [
            'trésorerie', 'position de change', 'change', 'devises', 'forex',
            'liquidité', 'cash management', 'gestion de trésorerie',
            'position de change', 'opérations de change'
        ],
        'DGEFRI-DOF : Trésorerie Pôle Clientèle': [
            'trésorerie pôle clientèle', 'pôle clientèle', 'clientèle',
            'relation client', 'service clientèle', 'gestion clientèle'
        ]
    }
    
    service_scores = {}
    for service, keywords in services_keywords.items():
        score = sum(1 for keyword in keywords if keyword in description_lower)
        if score > 0:
            service_scores[service] = score
    
    if service_scores:
        return max(service_scores, key=service_scores.get)
    else:
        return 'DGEFRI-DOF : Confirmation, Règlements et Livraison'

def determine_risk_and_gravity(neighbors, similarities, description):
    ligne_metier = detect_service_from_description(description)
    
    if ligne_metier == 'DGEFRI-DOF : Confirmation, Règlements et Livraison':
        risk_counts = {}
        for idx, sim in zip(neighbors.index, similarities):
            neighbor_ligne_metier = neighbors.loc[idx, 'Ligne Metier']
            if pd.notna(neighbor_ligne_metier) and neighbor_ligne_metier != '':
                risk_counts[neighbor_ligne_metier] = risk_counts.get(neighbor_ligne_metier, 0) + float(sim)
        
        if risk_counts:
            old_to_new_mapping = {
                'Back Office': 'DGEFRI-DOF : Confirmation, Règlements et Livraison',
                'Middle Office': 'DGEFRI-DOF : Risques de Marchés et Performance',
                'Front Office': 'DGEFRI-DOF : Trésorerie et Position de Change'
            }
            
            best_old_ligne = max(risk_counts, key=risk_counts.get)
            ligne_metier = old_to_new_mapping.get(best_old_ligne, ligne_metier)

    risk_categories = {
        'RF': ['fraude', 'détournement', 'corruption'],
        'RH': ['erreur humaine', 'formation', 'compétence'],
        'RJ': ['juridique', 'litige', 'contrat'],
        'RSI': ['système', 'informatique', 'logiciel', 'module', 'connexion', 'summit ft', 'swift'],
        'RSP': ['sécurité', 'physique', 'vol']
    }
    description_lower = description.lower()
    max_score = -1
    categorie_risque = 'Non déterminée'
    for cat, keywords in risk_categories.items():
        score = sum(1 for keyword in keywords if keyword in description_lower)
        if score > max_score:
            max_score = score
            categorie_risque = cat
    
    if categorie_risque == 'Non déterminée' and any(kw in description_lower for kw in ['module', 'connexion', 'summit ft', 'swift', 'accès']):
        categorie_risque = 'RSI'

    gravity_map = {
        'Très faible': 1, 'Faible': 2, 'Moyen': 3, 'Fort': 4, 'Très fort': 5
    }
    gravity_scores = []
    for _, row in neighbors.iterrows():
        date_ouverture = row['Ouverture']
        date_fin = row['Fin']
        try:
            date_ouverture = pd.to_datetime(date_ouverture, errors='coerce', dayfirst=True)
            date_fin = pd.to_datetime(date_fin, errors='coerce', dayfirst=True)
            if pd.isna(date_ouverture) or pd.isna(date_fin):
                duration = 12
            else:
                duration = (date_fin - date_ouverture).total_seconds() / 3600
            if duration <= 2:
                gravity = 'Très faible'
            elif duration <= 8:
                gravity = 'Faible'
            elif duration <= 24:
                gravity = 'Moyen'
            elif duration <= 48:
                gravity = 'Fort'
            else:
                gravity = 'Très fort'
            gravity_scores.append(gravity_map[gravity])
        except:
            gravity_scores.append(3)
    gravity_score = int(np.mean(gravity_scores))
    gravity = next(key for key, value in gravity_map.items() if value == gravity_score)

    return ligne_metier, gravity, categorie_risque

def map_risk_category_to_description(categorie_risque):
    risk_mapping = {
        'RF': 'Risque de fraude',
        'RH': 'Risque humain',
        'RJ': 'Risque juridique',
        'RSI': 'Risque du système d\'information',
        'RSP': 'Risque de sécurité des personnes et des biens',
        'Non déterminée': 'Risque du système d\'information'
    }
    
    flexible_mapping = {
        'Système d\'Information': 'Risque du système d\'information',
        'RSI': 'Risque du système d\'information'
    }
    return flexible_mapping.get(categorie_risque, risk_mapping.get(categorie_risque, 'Risque du système d\'information'))

def get_recommendations(ligne_metier, gravite, categorie_risque, recommendations_df):
    """Retourne deux recommandations : une principale et une alternative"""
    categorie_risque_desc = map_risk_category_to_description(categorie_risque)
    
    # Recherche de la recommandation principale
    matching_recommendations = recommendations_df[
        (recommendations_df['Catégorie de risque'] == categorie_risque_desc) &
        (recommendations_df['Niveau de gravité'] == gravite)
    ]
    
    recommendation_1 = "Aucune recommandation disponible pour cette catégorie de risque ou gravité."
    recommendation_2 = "Suivre les procédures standard de la DGEFRI-DOF et documenter l'incident pour amélioration continue."
    
    if not matching_recommendations.empty:
        recommendation_1 = matching_recommendations['Recommandation'].iloc[0]
        
        # Si il y a plusieurs recommandations dans la même catégorie/gravité
        if len(matching_recommendations) > 1:
            recommendation_2 = matching_recommendations['Recommandation'].iloc[1]
        else:
            # Chercher une recommandation alternative avec même catégorie mais gravité différente
            alternative_recommendations = recommendations_df[
                (recommendations_df['Catégorie de risque'] == categorie_risque_desc) &
                (recommendations_df['Niveau de gravité'] != gravite)
            ]
            if not alternative_recommendations.empty:
                recommendation_2 = alternative_recommendations['Recommandation'].iloc[0]
            else:
                # Chercher dans une catégorie de risque similaire
                similar_categories = [
                    'Risque du système d\'information',
                    'Risque humain',
                    'Risque juridique'
                ]
                for cat in similar_categories:
                    if cat != categorie_risque_desc:
                        similar_recs = recommendations_df[
                            (recommendations_df['Catégorie de risque'] == cat) &
                            (recommendations_df['Niveau de gravité'] == gravite)
                        ]
                        if not similar_recs.empty:
                            recommendation_2 = f"Alternative ({cat}) : " + similar_recs['Recommandation'].iloc[0]
                            break
    else:
        # Recherche de fallback si aucune correspondance exacte
        matching_recommendations = recommendations_df[
            (recommendations_df['Catégorie de risque'].str.lower() == categorie_risque_desc.lower()) &
            (recommendations_df['Niveau de gravité'] == gravite)
        ]
        
        if not matching_recommendations.empty:
            recommendation_1 = matching_recommendations['Recommandation'].iloc[0]
        else:
            # Dernière tentative : par service métier
            risk_mapping = {
                'DGEFRI-DOF : Comptabilité': 'Risque du système d\'information',
                'DGEFRI-DOF : Confirmation, Règlements et Livraison': 'Risque du système d\'information',
                'DGEFRI-DOF : Conformité, Risques de crédit et Risques Opérationnels': 'Risque juridique',
                'DGEFRI-DOF : Etudes et stratégies des marchés financiers': 'Risque du système d\'information',
                'DGEFRI-DOF : Risques de Marchés et Performance': 'Risque du système d\'information',
                'DGEFRI-DOF : Trésorerie et Position de Change': 'Risque du système d\'information',
                'DGEFRI-DOF : Trésorerie Pôle Clientèle': 'Risque du système d\'information'
            }
            fallback_categorie = risk_mapping.get(ligne_metier, 'Risque du système d\'information')
            fallback_recommendations = recommendations_df[
                (recommendations_df['Catégorie de risque'].str.lower() == fallback_categorie.lower()) &
                (recommendations_df['Niveau de gravité'] == gravite)
            ]
            
            if not fallback_recommendations.empty:
                recommendation_1 = fallback_recommendations['Recommandation'].iloc[0]
    
    return recommendation_1, recommendation_2

def find_nearest_neighbors(description, incidents_df, model, k=5):
    descriptions = incidents_df['Libellé'].astype(str).tolist()
    embeddings = model.encode(descriptions, convert_to_tensor=True)
    query_embedding = model.encode([description], convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, embeddings)[0]
    indices = similarities.argsort(descending=True)[:k]
    return incidents_df.iloc[indices], similarities[indices]

def generate_incident_id():
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    return f"Incident{timestamp}"

def create_historique_page():
    """Page d'historique des déclarations"""
    
    # Header de l'historique
    st.markdown("""
    <div style="background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%); 
                padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1 style="margin: 0; font-size: 2.5rem;">📚 Historique des Déclarations</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Suivi et Analyse des Incidents Passés</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement de la base d'incidents
    base_df = load_or_create_base_incidents()
    
    if base_df.empty:
        st.warning("📝 Aucun incident déclaré dans l'historique")
        st.info("💡 Commencez par déclarer des incidents via la page **'🆕 Nouvel Incident'**")
        return
    
    # Conversion des dates
    base_df['Date de survenance'] = pd.to_datetime(base_df['Date de survenance'], errors='coerce')
    base_df = base_df.sort_values('Date de survenance', ascending=False)
    
    # 📊 STATISTIQUES RAPIDES
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 Total Incidents", len(base_df))
    
    with col2:
        incidents_30j = len(base_df[base_df['Date de survenance'] >= datetime.now() - timedelta(days=30)])
        st.metric("📅 Derniers 30 jours", incidents_30j)
    
    with col3:
        cout_total = base_df['Cout_total_estime(en 10 000 FCFA)'].sum()
        st.metric("💰 Coût Total", f"{cout_total:,.0f} (10k FCFA)")
    
    with col4:
        gravite_forte = len(base_df[base_df['Gravité'].isin(['Fort', 'Très fort'])])
        st.metric("⚠️ Incidents Critiques", gravite_forte)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 🔍 FILTRES
    st.subheader("🔍 Filtres de Recherche")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filtre par période
        date_debut = st.date_input(
            "📅 Date de début",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now().date()
        )
        date_fin = st.date_input(
            "📅 Date de fin",
            value=datetime.now().date(),
            max_value=datetime.now().date()
        )
    
    with col2:
        # Filtre par gravité
        gravites_disponibles = base_df['Gravité'].unique().tolist()
        gravites_selectionnees = st.multiselect(
            "⚠️ Niveau de gravité",
            options=gravites_disponibles,
            default=gravites_disponibles
        )
    
    with col3:
        # Filtre par service
        services_disponibles = base_df['Entité'].unique().tolist()
        services_selectionnes = st.multiselect(
            "🏢 Services",
            options=services_disponibles,
            default=services_disponibles
        )
    
    # Application des filtres
    base_filtree = base_df[
        (base_df['Date de survenance'].dt.date >= date_debut) &
        (base_df['Date de survenance'].dt.date <= date_fin) &
        (base_df['Gravité'].isin(gravites_selectionnees)) &
        (base_df['Entité'].isin(services_selectionnes))
    ]
    
    st.markdown(f"**📊 {len(base_filtree)} incidents trouvés**")
    
    # 📈 GRAPHIQUES D'ANALYSE
    if not base_filtree.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Évolution temporelle
            st.subheader("📈 Évolution Temporelle")
            base_filtree['Mois'] = base_filtree['Date de survenance'].dt.to_period('M')
            evolution_mensuelle = base_filtree.groupby('Mois').size().reset_index(name='Nombre')
            evolution_mensuelle['Mois'] = evolution_mensuelle['Mois'].astype(str)
            
            fig_evolution = px.line(
                evolution_mensuelle, 
                x='Mois', 
                y='Nombre',
                title="Incidents par Mois",
                markers=True
            )
            fig_evolution.update_layout(height=300)
            st.plotly_chart(fig_evolution, use_container_width=True)
        
        with col2:
            # Répartition par gravité
            st.subheader("⚠️ Répartition par Gravité")
            repartition_gravite = base_filtree['Gravité'].value_counts()
            
            fig_gravite = px.pie(
                values=repartition_gravite.values,
                names=repartition_gravite.index,
                title="Distribution des Niveaux de Gravité"
            )
            fig_gravite.update_layout(height=300)
            st.plotly_chart(fig_gravite, use_container_width=True)
        
        # Répartition par service (graphique en barres horizontal)
        st.subheader("🏢 Répartition par Service")
        repartition_service = base_filtree['Entité'].value_counts()
        
        fig_service = px.bar(
            x=repartition_service.values,
            y=repartition_service.index,
            orientation='h',
            title="Nombre d'Incidents par Service",
            labels={'x': 'Nombre d\'Incidents', 'y': 'Services'}
        )
        fig_service.update_layout(height=400)
        st.plotly_chart(fig_service, use_container_width=True)
    
    # 📋 TABLEAU DÉTAILLÉ
    st.subheader("📋 Détail des Incidents")
    
    # Formatage du tableau
    if not base_filtree.empty:
        display_df = base_filtree.copy()
        display_df['Date de survenance'] = display_df['Date de survenance'].dt.strftime('%Y-%m-%d')
        display_df['Coût (10k FCFA)'] = display_df['Cout_total_estime(en 10 000 FCFA)'].round(0)
        
        # Colonnes à afficher
        colonnes_affichage = [
            'Code', 'Date de survenance', 'Entité', 'Gravité', 
            'Catégorie_Risque', 'Temps_H', 'Coût (10k FCFA)'
        ]
        
        # Affichage avec couleurs selon la gravité
        def colorier_gravite(val):
            if val == 'Très fort':
                return 'background-color: #ff6b6b; color: white'
            elif val == 'Fort':
                return 'background-color: #feca57; color: black'
            elif val == 'Moyen':
                return 'background-color: #48dbfb; color: black'
            elif val == 'Faible':
                return 'background-color: #1dd1a1; color: white'
            elif val == 'Très faible':
                return 'background-color: #55efc4; color: black'
            return ''
        
        styled_df = display_df[colonnes_affichage].copy()
        
        # Conversion des types pour éviter les erreurs Arrow
        for col in styled_df.columns:
            if styled_df[col].dtype == 'object':
                styled_df[col] = styled_df[col].fillna('').astype(str)
            elif pd.api.types.is_numeric_dtype(styled_df[col]):
                styled_df[col] = styled_df[col].fillna(0)
        
        # Application du style seulement si la colonne existe
        try:
            if 'Gravité' in styled_df.columns:
                styled_display = styled_df.style.applymap(
                    colorier_gravite, subset=['Gravité']
                )
                st.dataframe(styled_display, use_container_width=True, height=400)
            else:
                st.dataframe(styled_df, use_container_width=True, height=400)
        except Exception as e:
            # En cas d'erreur avec le style, afficher sans style
            st.dataframe(styled_df, use_container_width=True, height=400)
        
        # 📥 EXPORT DES DONNÉES
        st.subheader("📥 Export des Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export CSV
            csv_data = base_filtree.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📄 Télécharger CSV",
                data=csv_data,
                file_name=f"historique_incidents_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export Excel
            try:
                from io import BytesIO
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    base_filtree.to_excel(writer, sheet_name='Historique_Incidents', index=False)
                
                st.download_button(
                    label="📊 Télécharger Excel",
                    data=buffer.getvalue(),
                    file_name=f"historique_incidents_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except ImportError:
                st.info("📝 Export Excel non disponible (openpyxl requis)")
    
    # 🔍 RECHERCHE AVANCÉE
    st.subheader("🔍 Recherche Avancée")
    
    recherche_texte = st.text_input("🔎 Rechercher dans les codes d'incidents...")
    
    if recherche_texte:
        incidents_recherches = base_filtree[
            base_filtree['Code'].str.contains(recherche_texte, case=False, na=False)
        ]
        
        if not incidents_recherches.empty:
            st.write(f"**🎯 {len(incidents_recherches)} incidents trouvés :**")
            for _, incident in incidents_recherches.iterrows():
                with st.expander(f"📋 {incident['Code']} - {incident['Gravité']} - {incident['Date de survenance'].strftime('%Y-%m-%d') if pd.notna(incident['Date de survenance']) else 'Date inconnue'}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**🏢 Service :** {incident['Entité']}")
                        st.write(f"**⚠️ Gravité :** {incident['Gravité']}")
                        st.write(f"**🔍 Catégorie :** {incident['Catégorie_Risque']}")
                    with col2:
                        st.write(f"**⏱️ Durée :** {incident['Temps_H']} heures")
                        st.write(f"**💰 Coût :** {incident['Cout_total_estime(en 10 000 FCFA)']} (10k FCFA)")
        else:
            st.warning(f"🔍 Aucun incident trouvé pour : '{recherche_texte}'")

def create_dashboard(base_df):
    """Dashboard reproduisant exactement le style de l'image fournie"""
    
    # Header principal du dashboard
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1 style="margin: 0; font-size: 2.5rem;">🏦 Dashboard Risques Opérationnels DGEFRI-DOF</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Surveillance et Analyse en Temps Réel</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if base_df.empty:
        st.warning("Aucun incident dans la base de données")
        return
    
    # Conversion et nettoyage des données
    base_df['Date de survenance'] = pd.to_datetime(base_df['Date de survenance'], errors='coerce')
    base_df['Cout_total_estime(en 10 000 FCFA)'] = pd.to_numeric(base_df['Cout_total_estime(en 10 000 FCFA)'], errors='coerce')
    
    # 🎯 SECTION 1: 4 MÉTRIQUES PRINCIPALES (comme dans l'image)
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculs des métriques
    total_incidents = len(base_df)
    total_cout = base_df['Cout_total_estime(en 10 000 FCFA)'].sum()
    incidents_resolus = len(base_df[base_df['Gravité'].isin(['Très faible', 'Faible'])])
    taux_resolution = (incidents_resolus / total_incidents * 100) if total_incidents > 0 else 0
    cout_moyen = base_df['Cout_total_estime(en 10 000 FCFA)'].mean() if total_incidents > 0 else 0
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(45deg, #1e3c72 0%, #2a5298 100%);">
            <p class="metric-value">{total_incidents:,}</p>
            <p class="metric-label">Total Incidents Enregistrés</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(45deg, #c31432 0%, #240b36 100%);">
            <p class="metric-value">{total_cout:,.0f}</p>
            <p class="metric-label">Coût Total (en 10k FCFA)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(45deg, #11998e 0%, #38ef7d 100%);">
            <p class="metric-value">{taux_resolution:.1f}%</p>
            <p class="metric-label">Taux de Résolution</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(45deg, #fc4a1a 0%, #f7b733 100%);">
            <p class="metric-value">{cout_moyen:.0f}</p>
            <p class="metric-label">Coût Moyen par Incident</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 🎯 SECTION 2: 4 GAUGES CIRCULAIRES (comme dans l'image)
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculs pour les gauges
    incidents_critiques = len(base_df[base_df['Gravité'].isin(['Fort', 'Très fort'])])
    taux_critique = (incidents_critiques / total_incidents * 100) if total_incidents > 0 else 0
    
    incidents_swift = len(base_df[base_df['Entité'].str.contains('Confirmation', na=False)])
    taux_swift = (incidents_swift / total_incidents * 100) if total_incidents > 0 else 0
    
    incidents_rsi = len(base_df[base_df['Catégorie_Risque'] == 'RSI'])
    taux_rsi = (incidents_rsi / total_incidents * 100) if total_incidents > 0 else 0
    
    services_actifs = base_df['Entité'].nunique()
    taux_couverture = (services_actifs / 7 * 100)
    
    with col1:
        fig_gauge1 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = taux_critique,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Incidents Critiques"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#1f77b4"},
                'steps': [{'range': [0, 50], 'color': "lightgray"},
                         {'range': [50, 100], 'color': "gray"}],
                'threshold': {'line': {'color': "red", 'width': 4},
                            'thickness': 0.75, 'value': 80}}))
        fig_gauge1.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_gauge1, use_container_width=True)
    
    with col2:
        fig_gauge2 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = taux_swift,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Incidents SWIFT"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#ff7f0e"},
                'steps': [{'range': [0, 30], 'color': "lightgray"},
                         {'range': [30, 100], 'color': "gray"}],
                'threshold': {'line': {'color': "red", 'width': 4},
                            'thickness': 0.75, 'value': 60}}))
        fig_gauge2.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_gauge2, use_container_width=True)
    
    with col3:
        fig_gauge3 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = taux_rsi,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risques SI"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#d62728"},
                'steps': [{'range': [0, 40], 'color': "lightgray"},
                         {'range': [40, 100], 'color': "gray"}],
                'threshold': {'line': {'color': "red", 'width': 4},
                            'thickness': 0.75, 'value': 70}}))
        fig_gauge3.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_gauge3, use_container_width=True)
    
    with col4:
        fig_gauge4 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = taux_couverture,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Couverture Services"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#2ca02c"},
                'steps': [{'range': [0, 70], 'color': "lightgray"},
                         {'range': [70, 100], 'color': "gray"}],
                'threshold': {'line': {'color': "red", 'width': 4},
                            'thickness': 0.75, 'value': 85}}))
        fig_gauge4.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_gauge4, use_container_width=True)
    
    # 🎯 SECTION 3: GRAPHIQUES PRINCIPAUX (comme dans l'image)
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique linéaire d'évolution (comme "Net Working Capital vs Gross Working Capital")
        if not base_df['Date de survenance'].isna().all():
            base_df['Année-Mois'] = base_df['Date de survenance'].dt.to_period('M')
            
            # Évolution incidents vs coûts
            evolution_data = base_df.groupby('Année-Mois').agg({
                'Code': 'count',
                'Cout_total_estime(en 10 000 FCFA)': 'sum'
            }).reset_index()
            evolution_data['Année-Mois'] = evolution_data['Année-Mois'].astype(str)
            evolution_data = evolution_data.tail(12)  # 12 derniers mois
            
            fig_evolution = go.Figure()
            
            # Ligne des incidents (équivalent Net Working Capital)
            fig_evolution.add_trace(go.Scatter(
                x=evolution_data['Année-Mois'],
                y=evolution_data['Code'],
                mode='lines+markers',
                name='Nombre d\'Incidents',
                line=dict(color='#ffd700', width=3),
                marker=dict(size=8)
            ))
            
            # Ligne des coûts (équivalent Gross Working Capital)
            fig_evolution.add_trace(go.Scatter(
                x=evolution_data['Année-Mois'],
                y=evolution_data['Cout_total_estime(en 10 000 FCFA)']/10,  # Normalisé
                mode='lines+markers',
                name='Coûts (en 100k FCFA)',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8),
                yaxis='y2'
            ))
            
            fig_evolution.update_layout(
                title="Évolution Incidents vs Coûts Opérationnels",
                xaxis_title="Période",
                yaxis=dict(title="Nombre d'Incidents", side="left"),
                yaxis2=dict(title="Coûts (100k FCFA)", side="right", overlaying="y"),
                height=400,
                hovermode='x unified',
                legend=dict(x=0.7, y=1),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_evolution, use_container_width=True)
    
    with col2:
        # Graphique en barres empilées (comme "Profit and Loss summary")
        if not base_df['Date de survenance'].isna().all():
            # Données pour les barres empilées par catégorie de risque
            monthly_risks = base_df.groupby(['Année-Mois', 'Catégorie_Risque']).size().unstack(fill_value=0)
            monthly_risks = monthly_risks.tail(12)  # 12 derniers mois
            monthly_risks.index = monthly_risks.index.astype(str)
            
            fig_stacked = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for i, category in enumerate(monthly_risks.columns):
                fig_stacked.add_trace(go.Bar(
                    name=category,
                    x=monthly_risks.index,
                    y=monthly_risks[category],
                    marker_color=colors[i % len(colors)]
                ))
            
            fig_stacked.update_layout(
                title="Répartition Mensuelle par Catégorie de Risque",
                xaxis_title="Période",
                yaxis_title="Nombre d'Incidents",
                barmode='stack',
                height=400,
                legend=dict(x=0.7, y=1),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_stacked, use_container_width=True)
    
    # 🎯 SECTION 4: TABLEAU DÉTAILLÉ
    st.subheader("📋 Incidents Récents")
    
    # Mise en forme du tableau
    recent_incidents = base_df.sort_values('Date de survenance', ascending=False).head(10)
    display_cols = ['Code', 'Entité', 'Gravité', 'Catégorie_Risque', 'Cout_total_estime(en 10 000 FCFA)', 'Date de survenance']
    
    if not recent_incidents.empty:
        styled_df = recent_incidents[display_cols].copy()
        styled_df['Date de survenance'] = styled_df['Date de survenance'].dt.strftime('%Y-%m-%d')
        
        # Conversion des types pour éviter les erreurs Arrow
        for col in styled_df.columns:
            if styled_df[col].dtype == 'object':
                styled_df[col] = styled_df[col].fillna('').astype(str)
            elif pd.api.types.is_numeric_dtype(styled_df[col]):
                styled_df[col] = styled_df[col].fillna(0)
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=300
        )
    
    # Footer du dashboard
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: linear-gradient(90deg, #2d3436 0%, #636e72 100%); 
                padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
        <p style="margin: 0; font-size: 1rem;">
            📊 <strong>Dashboard DGEFRI-DOF</strong> • Système Expert de Gestion des Risques Opérationnels • 
            Mise à jour automatique en temps réel
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_incident_form():
    """Partie Formulaire de déclaration d'incident"""
    
    # Header du formulaire
    st.markdown("""
    <div style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); 
                padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1 style="margin: 0; font-size: 2.5rem;">🆕 Déclaration d'Incident</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Système Expert de Classification et d'Analyse</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Variables pour stocker les résultats de l'analyse
    neighbors = None
    similarities = None
    output_data = None
    
    with st.form("incident_form"):
        st.subheader("📝 Informations de l'incident")
        
        # Description de l'incident
        description = st.text_area(
            "Description de l'incident",
            placeholder="Décrivez l'incident en détail...",
            height=100
        )
        
        # Durée de résolution
        temps_h = st.number_input(
            "Durée de résolution (en heures)",
            min_value=0.0,
            value=1.0,
            step=0.5,
            format="%.1f"
        )
        
        # Bouton de soumission
        submitted = st.form_submit_button("🔍 Analyser et Enregistrer l'Incident")
        
        if submitted:
            if not description.strip():
                st.error("⚠️ Veuillez saisir une description de l'incident")
            else:
                # Chargement des données
                with st.spinner("🤖 Chargement du modèle et analyse..."):
                    incidents_df, recommendations_df = load_data()
                    if incidents_df is None or recommendations_df is None:
                        st.error("❌ Erreur lors du chargement des données")
                    else:
                        model = load_model()
                        if model is None:
                            st.error("❌ Erreur lors du chargement du modèle d'IA")
                            return
                        
                        # Analyse de l'incident
                        incident_id = generate_incident_id()
                        date_ouverture = datetime.now()
                        date_fin = date_ouverture + timedelta(hours=temps_h)
                        
                        # Recherche des voisins
                        neighbors, similarities = find_nearest_neighbors(description, incidents_df, model, k=5)
                        
                        # Détermination du risque et de la gravité
                        ligne_metier, gravite, categorie_risque = determine_risk_and_gravity(neighbors, similarities, description)
                        
                        # Recherche de recommandations - CORRIGÉ : deux recommandations
                        recommendation_1, recommendation_2 = get_recommendations(ligne_metier, gravite, categorie_risque, recommendations_df)
                        
                        # Calcul des coûts
                        incident_data = pd.DataFrame([{
                            'Code': incident_id,
                            'Entité': 'Inconnue',
                            'Gravité': gravite,
                            'Temps_H': temps_h,
                            'Ligne Metier': ligne_metier,
                            'Description': description,
                            'Categorie_Risque': categorie_risque
                        }])
                        
                        abc_estimator = EnhancedOpRiskABC()
                        resultats_enhanced = abc_estimator.estimate_enhanced_costs(incident_data)
                        cout_total = resultats_enhanced['Cout_total_enhanced'].iloc[0]
                        personnel = resultats_enhanced['Personnel_type'].iloc[0]
                        
                        # Préparation des données de sortie - CORRIGÉ : deux recommandations
                        output_data = {
                            'Identifiant': incident_id,
                            'Libellé': description,
                            'Ligne Metier': ligne_metier,
                            'Niveau de gravité': gravite,
                            'Ouverture': date_ouverture.strftime('%Y-%m-%d %H:%M:%S'),
                            'Fin': date_fin.strftime('%Y-%m-%d %H:%M:%S'),
                            'Recommandation_1': recommendation_1,
                            'Recommandation_2': recommendation_2,
                            'Coût total estimé (FCFA)': cout_total,
                            'Personnel_type': personnel,
                            'Categorie_Risque': map_risk_category_to_description(categorie_risque),
                            'Temps_H': temps_h,
                            'Categorie_Risque_Code': categorie_risque
                        }
                        
                        # Ajout à la base de données
                        updated_base = add_incident_to_base(output_data)
                
                # Affichage des résultats avec présentation améliorée
                st.success("✅ **INCIDENT ENREGISTRÉ AVEC SUCCÈS !**")
                
                # CSS pour les cartes de résultats
                st.markdown("""
                <style>
                .result-card {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 1.5rem;
                    border-radius: 15px;
                    color: white;
                    margin: 1rem 0;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                }
                .result-title {
                    font-size: 1.2rem;
                    font-weight: bold;
                    margin-bottom: 0.5rem;
                    text-align: center;
                }
                .result-value {
                    font-size: 1.5rem;
                    font-weight: bold;
                    text-align: center;
                    margin: 0.5rem 0;
                }
                .alert-high { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); }
                .alert-medium { background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%); }
                .alert-low { background: linear-gradient(135deg, #48dbfb 0%, #0abde3 100%); }
                .alert-success { background: linear-gradient(135deg, #1dd1a1 0%, #10ac84 100%); }
                </style>
                """, unsafe_allow_html=True)
                
                # En-tête avec l'ID de l'incident
                st.markdown(f"""
                <div class="result-card alert-success">
                    <div class="result-title">🆔 IDENTIFIANT DE L'INCIDENT</div>
                    <div class="result-value">{output_data['Identifiant']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Affichage en 2 colonnes principales
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 **CLASSIFICATION DE L'INCIDENT**")
                    
                    # Service détecté
                    service_short = output_data['Ligne Metier'].replace('DGEFRI-DOF : ', '')
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-title">🏢 SERVICE IDENTIFIÉ</div>
                        <div class="result-value">{service_short}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Niveau de gravité avec couleur appropriée
                    gravite_class = {
                        'Très faible': 'alert-success',
                        'Faible': 'alert-low', 
                        'Moyen': 'alert-medium',
                        'Fort': 'alert-high',
                        'Très fort': 'alert-high'
                    }
                    gravite_icon = {
                        'Très faible': '🟢',
                        'Faible': '🔵', 
                        'Moyen': '🟡',
                        'Fort': '🟠',
                        'Très fort': '🔴'
                    }
                    
                    st.markdown(f"""
                    <div class="result-card {gravite_class.get(output_data['Niveau de gravité'], 'alert-medium')}">
                        <div class="result-title">{gravite_icon.get(output_data['Niveau de gravité'], '⚠️')} NIVEAU DE GRAVITÉ</div>
                        <div class="result-value">{output_data['Niveau de gravité']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Catégorie de risque
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-title">🔍 CATÉGORIE DE RISQUE</div>
                        <div class="result-value">{output_data['Categorie_Risque']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 💰 **IMPACT FINANCIER & RESSOURCES**")
                    
                    # Coût estimé avec mise en forme
                    cout_formate = f"{output_data['Coût total estimé (FCFA)']:,.0f} FCFA"
                    cout_class = 'alert-high' if output_data['Coût total estimé (FCFA)'] > 50000 else 'alert-medium' if output_data['Coût total estimé (FCFA)'] > 20000 else 'alert-low'
                    
                    st.markdown(f"""
                    <div class="result-card {cout_class}">
                        <div class="result-title">💵 COÛT TOTAL ESTIMÉ</div>
                        <div class="result-value">{cout_formate}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Personnel assigné
                    personnel_mapping = {
                        'aes': '👨‍💼 Expert Senior (AES)',
                        'aem': '👨‍💻 Analyste Expert (AEM)', 
                        'accg': '👨‍🔧 Agent Comptable (ACCG)'
                    }
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-title">👥 PERSONNEL ASSIGNÉ</div>
                        <div class="result-value">{personnel_mapping.get(output_data['Personnel_type'], output_data['Personnel_type'])}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Durée de résolution
                    duree_text = f"{output_data['Temps_H']:.1f} heures"
                    if output_data['Temps_H'] >= 24:
                        duree_text += f" ({output_data['Temps_H']/24:.1f} jours)"
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-title">⏱️ DURÉE DE RÉSOLUTION</div>
                        <div class="result-value">{duree_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Section recommandations (pleine largeur) - DEUX RECOMMANDATIONS
                st.markdown("### 💡 **RECOMMANDATIONS D'ACTION**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); 
                               padding: 1.5rem; border-radius: 15px; color: white; 
                               border-left: 5px solid #fdcb6e; margin: 1rem 0;">
                        <h4 style="margin-top: 0; color: #fdcb6e;">📋 Recommandation Principale</h4>
                        <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 0;">
                            {output_data['Recommandation_1']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%); 
                               padding: 1.5rem; border-radius: 15px; color: white; 
                               border-left: 5px solid #00b894; margin: 1rem 0;">
                        <h4 style="margin-top: 0; color: #00b894;">🔄 Recommandation Alternative</h4>
                        <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 0;">
                            {output_data['Recommandation_2']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Timeline de l'incident
                st.markdown("### ⏰ **CHRONOLOGIE DE L'INCIDENT**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="result-card alert-low">
                        <div class="result-title">🚀 DÉBUT D'INTERVENTION</div>
                        <div class="result-value">{output_data['Ouverture'].split(' ')[1]}</div>
                        <div style="text-align: center; opacity: 0.8;">{output_data['Ouverture'].split(' ')[0]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="result-card alert-success">
                        <div class="result-title">✅ FIN PRÉVUE</div>
                        <div class="result-value">{output_data['Fin'].split(' ')[1]}</div>
                        <div style="text-align: center; opacity: 0.8;">{output_data['Fin'].split(' ')[0]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Section incidents similaires avec présentation améliorée
                st.markdown("### 🔍 **INCIDENTS SIMILAIRES ANALYSÉS**")
                st.markdown("*Les incidents suivants ont été utilisés pour la classification automatique :*")
                
                # Création du tableau avec style personnalisé
                similar_data = []
                for idx, row in neighbors.iterrows():
                    sim_value = similarities[neighbors.index.get_loc(idx)]
                    similarity_color = "🟢" if sim_value > 0.7 else "🟡" if sim_value > 0.5 else "🔴"
                    
                    similar_data.append({
                        '🆔 ID': row['Identifiant'],
                        '📝 Description': row['Libellé'][:80] + "..." if len(row['Libellé']) > 80 else row['Libellé'],
                        '🏢 Service': row['Ligne Metier'],
                        f'{similarity_color} Similarité': f"{sim_value:.1%}"
                    })
                
                # Affichage du tableau stylé
                similar_df = pd.DataFrame(similar_data)
                # Conversion des types pour éviter les erreurs Arrow
                for col in similar_df.columns:
                    if similar_df[col].dtype == 'object':
                        similar_df[col] = similar_df[col].astype(str)
                
                st.dataframe(
                    similar_df,
                    use_container_width=True,
                    height=200
                )
    
    # Bouton pour rafraîchir le dashboard (EN DEHORS du formulaire)
    if st.button("🔄 **Mettre à jour le Dashboard**", type="primary"):
        st.rerun()

def main():
    st.set_page_config(
        page_title="DGEFRI-DOF Risk Management System",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Titre principal avec style
    st.markdown("""
    <div style="background: linear-gradient(90deg, #2d3436 0%, #636e72 100%); 
                padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;">
        <h1 style="margin: 0; font-size: 2.2rem;">🏦 DGEFRI-DOF • Système Expert</h1>
        <p style="margin: 0.3rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Gestion Intelligente des Risques Opérationnels • IA & Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar pour navigation
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;">
        <h3 style="margin: 0;">🔧 Navigation</h3>
    </div>
    """, unsafe_allow_html=True)
    # Bouton retour accueil principal
    if st.sidebar.button("🏠 Retour Accueil Principal", use_container_width=True, type="secondary"):
        if 'redirect_to' in st.session_state:
            del st.session_state.redirect_to
        st.session_state.current_page = 'accueil'
        st.session_state.chatbot_subpage = None
        st.rerun()

    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Choisissez une section :",
        ["📊 Dashboard", "🆕 Nouvel Incident", "📚 Historique"],
        index=0
    )
    
    # Services DGEFRI-DOF dans la sidebar
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); 
                padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;">
        <h4 style="margin: 0 0 0.5rem 0;">🏢 Services DGEFRI-DOF</h4>
    </div>
    """, unsafe_allow_html=True)
    
    services = [
        "• Comptabilité",
        "• Confirmation, Règlements et Livraison",
        "• Conformité, Risques de crédit et Risques Opérationnels",
        "• Etudes et stratégies des marchés financiers",
        "• Risques de Marchés et Performance",
        "• Trésorerie et Position de Change",
        "• Trésorerie Pôle Clientèle"
    ]
    for service in services:
        st.sidebar.markdown(f"<small>{service}</small>", unsafe_allow_html=True)
    
    # Informations système dans la sidebar
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #00b894 0%, #00a085 100%); 
                padding: 0.8rem; border-radius: 8px; color: white; font-size: 0.8rem;">
        <strong>🤖 IA & Machine Learning</strong><br>
        • Classification automatique<br>
        • Analyse sémantique<br>
        • Recommandations intelligentes
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement de la base d'incidents
    base_df = load_or_create_base_incidents()
    
    if page == "📊 Dashboard":
        create_dashboard(base_df)
    elif page == "🆕 Nouvel Incident":
        create_incident_form()
    else:  # "📚 Historique"
        create_historique_page()

if __name__ == "__main__":
    # Configuration pour réduire les logs TensorFlow/PyTorch
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    
    try:
        main()
    except Exception as e:
        st.error(f"Erreur lors du démarrage de l'application : {e}")
        st.info("💡 Essayez de recharger la page ou contactez l'administrateur.")