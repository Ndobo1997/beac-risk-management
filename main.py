import streamlit as st
import os
from pathlib import Path
import subprocess
import sys

# Configuration de la page principale
st.set_page_config(
    page_title="BEAC - Système Expert Intelligent",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour l'interface
st.markdown("""
<style>
    /* Styles pour la sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Styles pour les titres */
    .main-title {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Styles pour les boutons de navigation */
    .nav-button {
        width: 100%;
        margin: 0.5rem 0;
        padding: 0.75rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Styles pour les sous-onglets */
    .sub-nav-button {
        width: 100%;
        margin: 0.25rem 0;
        margin-left: 1rem;
        padding: 0.5rem;
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .sub-nav-button:hover {
        background: linear-gradient(135deg, #0984e3 0%, #74b9ff 100%);
    }
    
    /* Styles pour les cartes d'information */
    .info-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
</style>
""", unsafe_allow_html=True)

# Initialisation des états de session
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'accueil'

if 'lda_subpage' not in st.session_state:
    st.session_state.lda_subpage = None

# Fonction pour afficher la page d'accueil
def show_home_page():
    """Affiche la page d'accueil avec l'image BEAC et la description"""
    
    # Titre principal
    st.markdown("""
    <div class="main-title">
        <h1>🏦 BEAC - Système Expert Intelligent</h1>
        <h3>Banque des États de l'Afrique Centrale</h3>
        <p>Plateforme Intégrée de Gestion des Risques Opérationnels et d'Assistance Réglementaire</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tentative de chargement de l'image BEAC
    image_path = r"images/photo_beac_1.jpg"  # Chemin relatif plus simple
    
    if os.path.exists(image_path):
        # Affichage de l'image avec description superposée
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            st.image(image_path, caption="Banque des États de l'Afrique Centrale", use_column_width=True)
    else:
        st.warning(f"⚠️ Image non trouvée au chemin : {image_path}")
        st.info("💡 Placez l'image dans le dossier 'images/photo_beac_1.jpg'")
    
    # Description de l'application
    st.markdown("""
    <div class="info-card">
        <h2>🎯 Bienvenue dans le Système Expert BEAC</h2>
        <p style="font-size: 1.1rem; line-height: 1.6;">
            Cette plateforme révolutionnaire combine intelligence artificielle et expertise réglementaire 
            pour vous accompagner dans la gestion des risques opérationnels et la conformité bancaire.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Guide d'utilisation
    st.markdown("## 📋 Guide d'Utilisation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔧 Gestion des Risques Opérationnels</h3>
            <p><strong>Accès :</strong> Chatbot → Gestion RO</p>
            <ul style="text-align: left; margin-top: 1rem;">
                <li>📊 Dashboard en temps réel</li>
                <li>🆕 Déclaration d'incidents</li>
                <li>📚 Historique détaillé</li>
                <li>🤖 Classification automatique par IA</li>
                <li>💰 Estimation des coûts ABC</li>
                <li>📈 Analyses et recommandations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>💬 Assistant Conversationnel DEMARIS</h3>
            <p><strong>Accès :</strong> Chatbot → Conversation</p>
            <ul style="text-align: left; margin-top: 1rem;">
                <li>🏛️ Expertise réglementaire Bâle III</li>
                <li>📖 Corpus documentaire enrichi</li>
                <li>🔍 Recherche intelligente RAG</li>
                <li>⚡ Réponses instantanées</li>
                <li>📊 Métriques de qualité</li>
                <li>🌐 Sources officielles BIS/EBA</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Instructions de navigation
    st.markdown("## 🧭 Navigation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>1️⃣ Menu Principal</h4>
            <p>Utilisez la barre latérale gauche pour naviguer entre les sections principales de l'application.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>2️⃣ Modules</h4>
            <p>Choisissez parmi les options disponibles comme 'Gestion RO', 'DEMARIS' ou 'LDA' pour accéder aux outils.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <h4>3️⃣ Retour Accueil</h4>
            <p>Cliquez sur '🏠 Accueil' à tout moment pour revenir à cette page principale.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>BEAC - Système Expert Intelligent</strong> | Version 1.0 | 
        Développé pour la Direction Générale des Études, de la Formation et de la Recherche Industrielle (DGEFRI)</p>
    </div>
    """, unsafe_allow_html=True)

# Fonction pour lancer un module externe
def launch_external_module(module_name, description):
    """Lance un module Python externe"""
    
    st.markdown(f"""
    <div class="main-title">
        <h1>🚀 Lancement de {description}</h1>
        <p>Redirection vers le module {module_name}...</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Vérifier si le fichier existe
    if not os.path.exists(f"{module_name}.py"):
        st.error(f"❌ Fichier {module_name}.py non trouvé dans le répertoire actuel")
        st.info("💡 Vérifiez que le fichier est présent et réessayez")
        return
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button(f"🔗 Ouvrir {description}", type="primary", use_container_width=True):
            # Option 1: Lancement direct dans le même contexte
            try:
                # Import dynamique du module
                if module_name == "S_expert":
                    st.info("🔄 Chargement du module Gestion RO...")
                    import S_expert
                    st.session_state.redirect_to = "S_expert"
                    st.rerun()
                    
                elif module_name == "chatbotdemaris":
                    st.info("🔄 Chargement du module DEMARIS...")
                    import chatbotdemaris
                    st.session_state.redirect_to = "chatbotdemaris"
                    st.rerun()
                    
                elif module_name == "Estimation_des_pertes":
                    st.info("🔄 Chargement du module Estimation des Pertes...")
                    import Estimation_des_pertes
                    st.session_state.redirect_to = "Estimation_des_pertes"
                    st.rerun()
                    
                elif module_name == "Stats_desc":
                    st.info("🔄 Chargement du module Stats Desc...")
                    import Stats_desc
                    st.session_state.redirect_to = "Stats_desc"
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement : {e}")
                
                # Option 2: Instructions pour lancement manuel
                st.warning("⚡ **Lancement Manuel Recommandé**")
                st.code(f"streamlit run {module_name}.py")
                
                st.info("""
                **Instructions :**
                1. Ouvrez un nouveau terminal/invite de commande
                2. Naviguez vers ce dossier
                3. Exécutez la commande ci-dessus
                4. Le module s'ouvrira dans un nouvel onglet
                """)
        
        if st.button("← Retour à l'Accueil", use_container_width=True):
            st.session_state.current_page = 'accueil'
            st.session_state.lda_subpage = None
            if 'chatbot_subpage' in st.session_state:
                del st.session_state.chatbot_subpage
            if 'redirect_to' in st.session_state:
                del st.session_state.redirect_to
            st.rerun()
    
    # Informations sur le module
    if module_name == "S_expert":
        st.markdown("""
        ### 📊 Fonctionnalités Gestion RO
        - **Dashboard temps réel** avec métriques avancées
        - **Classification automatique** par intelligence artificielle
        - **Calcul de coûts ABC** enrichi pour la BEAC
        - **Historique complet** avec filtres et exports
        - **Recommandations personnalisées** par incident
        """)
    
    elif module_name == "chatbotdemaris":
        st.markdown("""
        ### 💬 Fonctionnalités DEMARIS
        - **Expertise Bâle III** complète et à jour
        - **Sources documentaires** BIS, EBA officielles
        - **Recherche sémantique** RAG avancée
        - **Validation multi-sources** automatique
        - **Métriques de qualité** ROUGE, BLEU, similarité
        """)
    
    elif module_name == "Estimation_des_pertes":
        st.markdown("""
        ### 📈 Fonctionnalités Estimation des Pertes
        - **LDA Professionnel** avec modèle Weibull optimisé
        - **Distribution de Sévérité** et Fréquence
        - **Capital Économique** (VaR 95%, 99%, 99.9%)
        - **Simulation Monte Carlo** pour les pertes annuelles
        - **Résultats par Entité** avec formatage FCFA
        """)
    
    elif module_name == "Stats_desc":
        st.markdown("""
        ### 📊 Fonctionnalités Stats Desc
        - **Analyse Exploratoire** complète des incidents
        - **Statistiques par Entité**, Catégorie et Gravité
        - **Tendances Temporelles** annuelles et saisonnières
        - **Tests de Normalité** et QQ-Plots
        - **Percentiles de Risque** (VaR) en vraies valeurs FCFA
        """)

# Fonction de redirection après import
def handle_module_redirect():
    """Gère la redirection vers les modules importés"""
    if 'redirect_to' in st.session_state:
        if st.session_state.current_page != 'accueil':  # Ne redirige que si on n'est pas sur accueil
            if st.session_state.redirect_to == "S_expert":
                try:
                    import S_expert
                    S_expert.main()
                except Exception as e:
                    st.error(f"Erreur S_expert: {e}")
                    st.session_state.current_page = 'accueil'
                    del st.session_state.redirect_to
                    st.rerun()
                    
            elif st.session_state.redirect_to == "chatbotdemaris":
                try:
                    import chatbotdemaris
                    chatbotdemaris.main()
                except Exception as e:
                    st.error(f"Erreur DEMARIS: {e}")
                    st.session_state.current_page = 'accueil'
                    del st.session_state.redirect_to
                    st.rerun()
                    
            elif st.session_state.redirect_to == "Estimation_des_pertes":
                try:
                    import Estimation_des_pertes
                    Estimation_des_pertes.main()
                except Exception as e:
                    st.error(f"Erreur Estimation_des_pertes: {e}")
                    st.session_state.current_page = 'accueil'
                    del st.session_state.redirect_to
                    st.rerun()
                    
            elif st.session_state.redirect_to == "Stats_desc":
                try:
                    import Stats_desc
                    Stats_desc.main()
                except Exception as e:
                    st.error(f"Erreur Stats_desc: {e}")
                    st.session_state.current_page = 'accueil'
                    del st.session_state.redirect_to
                    st.rerun()
        else:
            del st.session_state.redirect_to  # Supprime redirect_to si on est sur accueil

# Fonction principale pour la navigation
def main():
    """Fonction principale de l'application"""
    
    # Vérifier s'il y a une redirection en cours
    handle_module_redirect()
    
    # Sidebar pour navigation
    st.sidebar.markdown("""
    <div style="background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%); 
                padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;">
        <h2>🏦 BEAC</h2>
        <p>Navigation Système Expert</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Bouton Accueil
    if st.sidebar.button("🏠 Accueil", use_container_width=True):
        st.session_state.current_page = 'accueil'
        st.session_state.lda_subpage = None
        if 'chatbot_subpage' in st.session_state:
            del st.session_state.chatbot_subpage
        if 'redirect_to' in st.session_state:
            del st.session_state.redirect_to
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Section Modules
    st.sidebar.markdown("### 🤖 Modules Disponibles")
    
    # Bouton principal LDA avec sous-boutons
    if st.sidebar.button("📊 LDA", use_container_width=True, help="Loss Distribution Approach pour l'analyse des pertes"):
        st.session_state.current_page = 'lda'
        st.session_state.lda_subpage = None
        st.rerun()
    
    if st.session_state.current_page == 'lda' and st.session_state.lda_subpage is None:
        with st.sidebar:
            if st.button("📈 Estimation des Pertes", key="estimation_pertes", help="Analyse LDA avec modèle Weibull", use_container_width=True):
                st.session_state.lda_subpage = 'estimation_pertes'
                st.rerun()
            if st.button("📊 Stats Desc", key="stats_desc", help="Analyse exploratoire des données", use_container_width=True):
                st.session_state.lda_subpage = 'stats_desc'
                st.rerun()
    
    # Boutons Chatbot existants
    if st.sidebar.button("🔧 Gestion RO", use_container_width=True, help="Module de gestion des risques opérationnels"):
        st.session_state.current_page = 'chatbot'
        st.session_state.chatbot_subpage = 'gestion_ro'
        st.rerun()
    
    if st.sidebar.button("💬 DEMARIS", use_container_width=True, help="Assistant conversationnel réglementaire"):
        st.session_state.current_page = 'chatbot'
        st.session_state.chatbot_subpage = 'conversation'
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Informations système
    st.sidebar.markdown("### ℹ️ Informations")
    st.sidebar.info(f"**Page actuelle :** {st.session_state.current_page}")
    
    if st.session_state.current_page == 'chatbot' and st.session_state.chatbot_subpage:
        module_name = "Gestion RO" if st.session_state.chatbot_subpage == 'gestion_ro' else "DEMARIS"
        st.sidebar.success(f"**Module :** {module_name}")
    elif st.session_state.current_page == 'lda' and st.session_state.lda_subpage:
        module_name = "Estimation des Pertes" if st.session_state.lda_subpage == 'estimation_pertes' else "Stats Desc"
        st.sidebar.success(f"**Module :** {module_name}")
    
    # Footer sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; font-size: 0.8rem; color: #666;">
        <p>BEAC Système Expert v1.0<br>
        DGEFRI-DOF<br>
        © 2024</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Affichage du contenu principal
    if st.session_state.current_page == 'accueil':
        show_home_page()
    
    elif st.session_state.current_page == 'chatbot':
        if st.session_state.chatbot_subpage == 'gestion_ro':
            launch_external_module("S_expert", "Gestion des Risques Opérationnels")
                
        elif st.session_state.chatbot_subpage == 'conversation':
            launch_external_module("chatbotdemaris", "Assistant DEMARIS")
        
        else:
            # Page de sélection des modules
            st.markdown("""
            <div class="main-title">
                <h1>🤖 Modules BEAC</h1>
                <p>Sélectionnez le module souhaité</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔧 Gestion des Risques Opérationnels", use_container_width=True, type="primary"):
                    st.session_state.chatbot_subpage = 'gestion_ro'
                    st.rerun()
            
            with col2:
                if st.button("💬 Assistant Conversationnel DEMARIS", use_container_width=True, type="primary"):
                    st.session_state.chatbot_subpage = 'conversation'
                    st.rerun()
    
    elif st.session_state.current_page == 'lda':
        if st.session_state.lda_subpage == 'estimation_pertes':
            launch_external_module("Estimation_des_pertes", "Estimation des Pertes")
        elif st.session_state.lda_subpage == 'stats_desc':
            launch_external_module("Stats_desc", "Stats Desc")
        else:
            # Page de sélection des sous-modules LDA
            st.markdown("""
            <div class="main-title">
                <h1>📊 Modules LDA</h1>
                <p>Sélectionnez l'outil d'analyse souhaité</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📈 Estimation des Pertes", use_container_width=True, type="primary"):
                    st.session_state.lda_subpage = 'estimation_pertes'
                    st.rerun()
            
            with col2:
                if st.button("📊 Stats Desc", use_container_width=True, type="primary"):
                    st.session_state.lda_subpage = 'stats_desc'
                    st.rerun()

if __name__ == "__main__":
    main()
