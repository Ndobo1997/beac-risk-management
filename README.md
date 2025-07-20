# beac-risk-management
Système Expert BEAC - Gestion des Risques Opérationnels et Assistant DEMARIS avec IA
 🏦 BEAC - Système Expert Intelligent

Plateforme intégrée de gestion des risques opérationnels développée pour la DGEFRI-DOF (Direction Générale des Études, de la Formation et de la Recherche Industrielle) de la Banque des États de l'Afrique Centrale.

## 🎯 Qu'est-ce que cette application fait ?

Cette application combine **Intelligence Artificielle** et **expertise réglementaire** pour :

- **🔧 Gestion des Risques Opérationnels** : Classification automatique des incidents, calcul de coûts ABC, recommandations intelligentes
- **💬 Assistant Réglementaire DEMARIS** : Chatbot expert en réglementation bancaire (Bâle III, COSO) avec technologie RAG
- **📈 Estimation des Pertes LDA** : Modélisation statistique avec distribution Weibull pour calcul du capital économique
- **📊 Analyse Statistique** : Analyse exploratoire complète des données d'incidents

## 🚀 Installation et Démarrage

### Prérequis
- Python 3.8+
- 4 GB RAM minimum

### Installation
```bash
# 1. Installer les dépendances
pip install streamlit pandas numpy plotly matplotlib seaborn scipy torch sentence-transformers scikit-learn langchain langchain-community langchain-huggingface faiss-cpu openpyxl rouge-score nltk

# 2. Lancer l'application
streamlit run main.py
L'application sera accessible sur http://localhost:8501
📁 Structure des Fichiers Requis
Créez cette structure dans votre dossier projet :
votre-projet/
├── main.py
├── S_expert.py
├── chatbotdemaris.py
├── Estimation_des_pertes.py
├── Stats_desc.py
├── data/
│   ├── Base_entrainement.xlsx
│   ├── base_incidents.xlsx
│   └── Recommandations_risques_supplementaires_4.csv
├── training_corpus/
│   ├── basel_ii_operational.pdf
│   ├── coso_framework.pdf
│   └── demaris_methodolo.pdf
└── images/
    └── photo_beac_1.jpg
📊 Format des Données Excel
Fichier data/base_incidents.xlsx (Feuille: "Incidents_DOF_augmente")
ColonneFormatExempleCodeTexte"Incident20241220143052"EntitéTexte"DGEFRI-DOF : Comptabilité"GravitéTexte"Fort", "Moyen", "Faible", "Très fort", "Très faible"Cout_total_estime(en 10 000 FCFA)Nombre25 (= 250,000 FCFA)Date de survenanceDate"15-03-2024" (format DD-MM-YYYY)Catégorie_RisqueTexte"RSI", "RF", "RH", "RC"Temps_HNombre2.5 (heures)
Services DGEFRI-DOF supportés :

Comptabilité
Confirmation, Règlements et Livraison
Conformité, Risques de crédit et Risques Opérationnels
Études et stratégies des marchés financiers
Risques de Marchés et Performance
Trésorerie et Position de Change
Trésorerie Pôle Clientèle

🔧 Comment Utiliser Chaque Module
1. 🔧 Gestion des Risques Opérationnels (S_expert.py)
Objectif : Déclarer et analyser des incidents avec IA
Utilisation :

Cliquez sur "🔧 Gestion RO" dans le menu
Allez à "🆕 Nouvel Incident"
Saisissez la description de l'incident
Spécifiez la durée de résolution
Le système classifie automatiquement et calcule les coûts

Exemple :
Description: "Module Summit FT inaccessible depuis ce matin"
Durée: 3.5 heures
→ Résultat: Service détecté, coût calculé, recommandations générées
2. 💬 Assistant DEMARIS (chatbotdemaris.py)
Objectif : Poser des questions sur la réglementation bancaire
Utilisation :

Cliquez sur "💬 DEMARIS" dans le menu
Posez votre question dans le chat
Obtenez une réponse basée sur les documents réglementaires

Questions types :

"Qu'est-ce que l'approche AMA pour le risque opérationnel ?"
"Expliquez les trois piliers de Bâle III"
"Comment fonctionne le framework COSO ?"

3. 📈 Estimation des Pertes (Estimation_des_pertes.py)
Objectif : Calculer le capital économique selon Bâle III
Utilisation :

Cliquez sur "📊 LDA" → "📈 Estimation des Pertes"
Le système analyse automatiquement vos données d'incidents
Obtenez le VaR 99.9% et le capital réglementaire

Résultat type :
Distribution: Weibull
VaR 99.9%: 2.3 milliards FCFA
Capital réglementaire requis: 2.3 milliards FCFA
4. 📊 Analyse Statistique (Stats_desc.py)
Objectif : Analyser les patterns et tendances des incidents
Utilisation :

Cliquez sur "📊 LDA" → "📊 Stats Desc"
Consultez les analyses automatiques :

Statistiques par entité
Évolution temporelle
Tests de normalité
Percentiles de risque



🛠️ Configuration Importante
Paramètres BEAC (dans S_expert.py)
python# Grille salariale BEAC
salaires_beac = {
    'aes': 800_000,    # Agent Expert Senior
    'aem': 500_000,    # Analyste Expert Moyen  
    'accg': 300_000    # Agent Comptable
}
Modèles IA utilisés

Classification des incidents : all-MiniLM-L6-v2
Assistant DEMARIS : qwen2.5:0.5b-instruct (Ollama)
Recherche sémantique : FAISS + embeddings

🔍 Dépannage
Problème : Erreur de chargement des modèles IA
❌ Erreur lors du chargement du modèle principal
Solution : Vérifiez votre connexion internet. Les modèles se téléchargent automatiquement au premier lancement.
Problème : Fichier Excel non trouvé
❌ Fichier non trouvé : data/base_incidents.xlsx
Solution :

Vérifiez que le fichier existe dans le dossier data/
Vérifiez le nom exact du fichier
Vérifiez que la feuille s'appelle "Incidents_DOF_augmente"

Problème : DEMARIS ne répond pas
⚠️ Documents non trouvés dans training_corpus
Solution : Placez vos documents PDF dans le dossier training_corpus/
Problème : Mémoire insuffisante
Solution :

Fermez les autres applications
Réduisez la taille des fichiers PDF (< 50 MB chacun)
Augmentez la mémoire virtuelle de votre système

📞 Support
DGEFRI-DOF - BEAC
Email : support-technique@beac.int
Développé pour la Direction Générale des Études, de la Formation et de la Recherche Industrielle

Version : 1.0
Dernière mise à jour : Décembre 2024
© 2024 BEAC - Usage interne uniquement
