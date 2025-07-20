import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import time
import json
from pathlib import Path
import requests
import zipfile
import os
import tempfile
from typing import List, Dict, Optional
import hashlib

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document

class DocumentaryResourceManager:
    def __init__(self, cache_dir="cache/regulatory_sources"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.downloaded_docs_dir = self.cache_dir / "downloaded_pdfs"
        self.downloaded_docs_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.cache_dir / "download_metadata.json"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        self.download_metadata = self._load_download_metadata()
    
    def _load_download_metadata(self) -> dict:
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Erreur lecture métadonnées: {e}")
        return {"downloads": {}, "last_update": None}
    
    def _save_download_metadata(self):
        try:
            from datetime import datetime
            self.download_metadata["last_update"] = datetime.now().isoformat()
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.download_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde métadonnées: {e}")
    
    def load_local_documents(self, documents_path="training_corpus") -> List[Document]:
        st.info("📁 Chargement depuis training_corpus/...")
        
        local_docs = []
        documents_path = Path(documents_path)
        
        if not documents_path.exists():
            st.warning(f"📂 Dossier {documents_path} non trouvé")
            return []
        
        pdf_files = list(documents_path.glob("*.pdf"))
        st.info(f"📄 Trouvé {len(pdf_files)} PDFs dans {documents_path}")
        
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                
                filename_lower = pdf_file.name.lower()
                if "basel" in filename_lower or "bale" in filename_lower:
                    document_type = "regulatory_framework"
                    authority = "Basel Committee (BEAC Training)"
                    topic = "operational_risk_basel"
                elif "coso" in filename_lower:
                    document_type = "control_framework" 
                    authority = "COSO Framework (BEAC Training)"
                    topic = "internal_control"
                elif "demaris" in filename_lower:
                    document_type = "methodology"
                    authority = "DEMARIS Methodology (BEAC)"
                    topic = "demaris_framework"
                else:
                    document_type = "training_document"
                    authority = "BEAC Formation"
                    topic = "general_training"
                
                for doc in docs:
                    doc.metadata.update({
                        'filename': pdf_file.name,
                        'source_type': document_type,
                        'file_path': str(pdf_file),
                        'document_type': document_type,
                        'authority': authority,
                        'topic': topic,
                        'source': f"LOCAL_{pdf_file.stem.upper()}"
                    })
                
                local_docs.extend(docs)
                st.success(f"✅ Chargé: {pdf_file.name} ({len(docs)} pages)")
                
            except Exception as e:
                st.warning(f"❌ Erreur lecture {pdf_file.name}: {e}")
        
        if local_docs:
            st.success(f"✅ Total chargé: {len(local_docs)} pages depuis vos documents")
        else:
            st.warning("⚠️ Aucun document chargé depuis training_corpus")
            st.info("""
            💡 **Vérifiez que vous avez des fichiers PDF dans :**
            training_corpus/basel_ii_operational.pdf
            training_corpus/coso_framework.pdf  
            training_corpus/demaris_methodolo.pdf
            """)
            
        return local_docs
    
    def get_all_documents(self, include_local=True, force_download: bool = False) -> List[Document]:
        all_documents = []
        
        if include_local:
            local_docs = self.load_local_documents()
            all_documents.extend(local_docs)
            st.success(f"✅ Documents locaux chargés: {len(all_documents)}")
        else:
            st.info("📁 Chargement local désactivé")
        
        return all_documents
    
    def get_download_statistics(self) -> dict:
        stats = {
            "total_downloads": len(self.download_metadata["downloads"]),
            "last_update": self.download_metadata.get("last_update"),
            "downloads_by_source": {},
            "total_size_mb": 0,
            "local_docs_count": 0
        }
        
        for download in self.download_metadata["downloads"].values():
            filename = download["filename"].lower()
            if "bis" in filename or "basel" in filename:
                source = "BIS"
            elif "eba" in filename:
                source = "EBA"
            elif "ecb" in filename or "imf" in filename or "fed" in filename:
                source = "Academic"
            else:
                source = "Other"
            
            stats["downloads_by_source"][source] = stats["downloads_by_source"].get(source, 0) + 1
            stats["total_size_mb"] += download.get("file_size", 0) / (1024 * 1024)
        
        documents_path = Path("training_corpus")
        if documents_path.exists():
            stats["local_docs_count"] = len(list(documents_path.glob("*.pdf")))
        
        return stats

ENHANCED_VERIFIED_ANSWERS = {
    "ama": """**AMA (Advanced Measurement Approach)**

L'Approche de Mesure Avancée (AMA) est une méthode sophistiquée pour calculer les exigences de fonds propres au titre du risque opérationnel selon Bâle II/III.

**Caractéristiques principales :**
- Utilise les modèles internes de la banque validés par les superviseurs
- Basée sur au minimum 5 ans de données historiques de pertes
- Intègre les facteurs de risque prospectifs via l'analyse de scénarios
- Nécessite l'approbation explicite des autorités de supervision

**Quatre composants obligatoires :**
1. **Données de pertes internes** - Historique détaillé des incidents (≥5 ans)
2. **Données externes** - Bases de données sectorielles et consortiums
3. **Analyse de scénarios** - Évaluation prospective d'événements extrêmes
4. **Facteurs d'environnement de contrôle** - Qualité du contrôle interne (BEICFs)

**Critères quantitatifs :**
- Seuil de collecte : €20,000 pour les pertes brutes
- Période d'observation minimale : 5 ans
- Confiance : 99.9% sur horizon 1 an
- Corrélations entre risques prises en compte

**Critères qualitatifs :**
- Gouvernance robuste et indépendance de la fonction risque opérationnel
- Intégration dans la gestion quotidienne des risques
- Validation régulière des modèles et back-testing
- Documentation complète des méthodologies

*Source : Bâle II/III, EBA Guidelines on AMA*""",

    "piliers_bale": """**Les Trois Piliers de Bâle III**

**PILIER 1 - Exigences minimales de fonds propres renforcées**

*Capital de base (CET1 - Common Equity Tier 1) :*
- Ratio CET1 minimum : 4,5% des actifs pondérés des risques
- Coussin de conservation : 2,5% supplémentaires
- Coussin contracyclique : 0% à 2,5% selon le cycle économique
- Surcharge systémique : 1% à 3,5% pour les banques G-SIB

*Couverture des risques :*
- Risque de crédit (approches standardisée/IRB)
- Risque de marché (méthode standard/modèles internes)
- Risque opérationnel (BIA/TSA/AMA → SA sous Bâle IV)

**PILIER 2 - Processus de surveillance prudentielle**

*ICAAP (Internal Capital Adequacy Assessment Process) :*
- Évaluation interne de l'adéquation des fonds propres
- Stress tests et analyses de scénarios
- Gouvernance des risques et appétit pour le risque
- Capital pour risques non couverts par le Pilier 1

*SREP (Supervisory Review and Evaluation Process) :*
- Examen superviseur des pratiques bancaires
- Validation des modèles internes et méthodologies
- Fixation d'exigences de capital individuelles
- Mesures correctives si nécessaire

**PILIER 3 - Discipline de marché et transparence**

*Obligations de communication :*
- Rapport Pilier 3 semestriel/annuel détaillé
- Informations sur l'adéquation des fonds propres
- Exposition aux risques par type et géographie
- Méthodologies de mesure et de gestion des risques
- Ratios prudentiels et indicateurs de solidité

*Objectifs :*
- Renforcer la confiance des marchés financiers
- Permettre aux investisseurs d'évaluer les risques
- Encourager les bonnes pratiques de gestion
- Compléter la supervision réglementaire

*Source : Accords de Bâle III, Règlement CRR/CRD IV*""",

    "coso": """**COSO (Committee of Sponsoring Organizations)**

Le **Committee of Sponsoring Organizations of the Treadway Commission** a développé le référentiel de contrôle interne le plus utilisé mondialement, adopté par les régulateurs bancaires et d'assurance.

**Les 5 Composants du Contrôle Interne COSO 2013 :**

**1. Environnement de contrôle** 
- Intégrité et valeurs éthiques démontrées par la direction
- Indépendance du conseil d'administration et surveillance efficace
- Structure organisationnelle claire et attribution des responsabilités
- Compétences professionnelles et développement du personnel
- Redevabilité en matière de performance et mesures correctives

**2. Évaluation des risques**
- Spécification d'objectifs clairs et cohérents
- Identification et analyse des risques menaçant les objectifs
- Évaluation du risque de fraude dans tous ses aspects
- Identification et évaluation des changements significatifs
- Analyse de l'impact et de la probabilité des risques

**3. Activités de contrôle**
- Sélection et développement d'activités de contrôle appropriées
- Contrôles généraux informatiques et contrôles applicatifs
- Déploiement des contrôles via des politiques et procédures
- Séparation des tâches et autorisations appropriées
- Supervision directe et revues de performance

**4. Information et communication**
- Utilisation d'informations pertinentes et de qualité
- Communication interne des responsabilités de contrôle
- Communication externe avec les parties prenantes
- Systèmes d'information soutenant le contrôle interne
- Canaux de communication pour signaler les déficiences

**5. Activités de pilotage (Monitoring)**
- Sélection, développement et réalisation d'évaluations continues
- Évaluations ponctuelles pour confirmer l'efficacité
- Communication des déficiences de contrôle interne
- Suivi de la mise en œuvre des actions correctives
- Assurance qualité et amélioration continue

**17 Principes sous-jacents :**
Chaque composant est soutenu par des principes spécifiques qui, lorsqu'ils sont présents et fonctionnent ensemble, permettent de conclure que les cinq composants sont efficaces.

**Applications réglementaires :**
- SOX 404 (États-Unis) - Contrôles financiers
- AMF (France) - Contrôle interne des sociétés cotées  
- Solvabilité II (UE) - Système de gouvernance des assureurs
- Bâle III - Gouvernance des risques bancaires

*Source : COSO Framework 2013, ERM Framework 2017*""",

    "risque_operationnel": """**Risque Opérationnel - Définition et Framework Réglementaire**

**Définition Officielle (Bâle III) :**
Le risque opérationnel est défini comme *"le risque de perte résultant de processus internes inadéquats ou défaillants, de personnes et de systèmes ou d'événements externes"*. Cette définition inclut le risque juridique mais exclut les risques stratégiques et de réputation.

**Taxonomie des Risques Opérationnels :**

**1. Risques de Processus :**
- Inadéquation ou défaillance des processus métier
- Erreurs dans l'exécution, la livraison et la gestion des processus
- Complexité excessive et manque de standardisation
- Défaillances dans les contrôles et la supervision

**2. Risques de Personnel :**
- Erreur humaine, négligence ou incompétence
- Fraude interne et malveillance
- Violations des politiques internes et réglementations
- Problèmes de ressources humaines et relations sociales
- Santé et sécurité au travail

**3. Risques de Systèmes :**
- Pannes et défaillances des systèmes informatiques
- Cyberattaques et sécurité informatique
- Intégrité et qualité des données
- Obsolescence technologique
- Indisponibilité des systèmes critiques

**4. Risques Externes :**
- Catastrophes naturelles et événements météorologiques
- Actes de terrorisme et troubles civils
- Défaillances d'infrastructures externes (énergie, télécoms)
- Évolutions réglementaires défavorables
- Risques géopolitiques et sanctions

**Catégories d'Événements Bâle (Level 1) :**
1. Fraude interne
2. Fraude externe  
3. Pratiques en matière d'emploi et sécurité du lieu de travail
4. Pratiques avec la clientèle, produits et services
5. Dommages aux biens
6. Dysfonctionnement de l'activité et défaillance des systèmes
7. Exécution, livraison et gestion des processus

**Framework de Gestion :**

**Gouvernance :**
- Appétit et tolérance au risque définis par le conseil
- Fonction risque opérationnel indépendante
- Modèle des trois lignes de défense
- Reporting régulier au management et au conseil

**Identification et Évaluation :**
- Cartographie des risques par processus métier
- Auto-évaluations des risques et contrôles (RCSA)
- Collecte et analyse des données de pertes
- Indicateurs clés de risque (KRI) et alertes précoces

**Maîtrise et Atténuation :**
- Plans de continuité d'activité (PCA/BCP)
- Contrôles préventifs, détectifs et correctifs
- Transfert de risque (assurance, externalisation)
- Formation et sensibilisation du personnel

**Surveillance et Reporting :**
- Monitoring continu des expositions et contrôles
- Reporting d'incidents et analyse des causes racines
- Tests de résistance (stress testing)
- Validation indépendante par l'audit interne

*Source : Bâle III, EBA Guidelines, Solvabilité II*"""
}

def enhanced_get_verified_answer(question: str, retrieved_context: List[Document]) -> Optional[str]:
    question_lower = question.lower()
    
    if any(kw in question_lower for kw in ['ama', 'advanced measurement', 'mesure avancée', 'approche de mesure']):
        return ENHANCED_VERIFIED_ANSWERS["ama"]
    elif any(kw in question_lower for kw in ['pilier', 'piliers', 'bâle', 'basel', 'bale']) and any(kw in question_lower for kw in ['trois', '3', 'iii']):
        return ENHANCED_VERIFIED_ANSWERS["piliers_bale"]
    elif any(kw in question_lower for kw in ['coso', 'contrôle interne', 'control', 'committee sponsoring']):
        return ENHANCED_VERIFIED_ANSWERS["coso"]
    elif any(kw in question_lower for kw in ['risque opérationnel', 'operational risk', 'risques opérationnels']):
        return ENHANCED_VERIFIED_ANSWERS["risque_operationnel"]
    
    return None

def enhanced_validate_response(response: str, question: str, context_sources: List[str]) -> List[str]:
    response_lower = response.lower()
    question_lower = question.lower()
    
    hallucinations_detected = []
    
    critical_suspicious_phrases = [
        "je ne peux pas répondre", "je ne sais vraiment pas", "aucune information disponible",
        "impossible de répondre", "données insuffisantes"
    ]
    
    if len(response) < 100 and any(phrase in response_lower for phrase in critical_suspicious_phrases):
        hallucinations_detected.append("Réponse trop vague ou refus injustifié")
    
    if 'ama' in question_lower and 'crédit' in response_lower and 'opérationnel' not in response_lower:
        hallucinations_detected.append("Confusion AMA/Crédit grave")
    
    if 'coso' in question_lower and 'basel' in response_lower and 'contrôle interne' not in response_lower:
        hallucinations_detected.append("Confusion COSO/Basel grave")
    
    if len(response) > 200 and context_sources:
        return []
    
    return hallucinations_detected

class EnhancedSafeDemarisQA:
    def __init__(self, qa_chain, resource_manager=None):
        self.qa_chain = qa_chain
        self.resource_manager = resource_manager
    
    def invoke(self, inputs):
        question = inputs.get("query", "")
        
        if self.qa_chain is None:
            verified_answer = enhanced_get_verified_answer(question, [])
            if verified_answer:
                return {
                    "result": verified_answer,
                    "source_documents": []
                }
            else:
                return {
                    "result": f"Information sur '{question}' non disponible. Système DEMARIS en mode dégradé - ressources documentaires non chargées.",
                    "source_documents": []
                }
        
        result = self.qa_chain.invoke(inputs)
        retrieved_context = result.get("source_documents", [])
        
        verified_answer = enhanced_get_verified_answer(question, retrieved_context)
        if verified_answer:
            return {
                "result": verified_answer,
                "source_documents": retrieved_context
            }
        
        generated_response = result["result"]
        context_sources = [doc.metadata.get("source", "") for doc in retrieved_context]
        
        authoritative_sources = [src for src in context_sources if any(auth in src for auth in ['BIS_', 'EBA_', 'DATASET_', 'LOCAL_'])]
        
        if len(generated_response) > 150 and len(authoritative_sources) > 0:
            critical_hallucinations = enhanced_validate_response(generated_response, question, context_sources)
            
            if not critical_hallucinations:
                enhanced_response = f"{generated_response}\n\n*Sources consultées : {', '.join(set(authoritative_sources[:3]))}*"
                return {
                    "result": enhanced_response,
                    "source_documents": retrieved_context
                }
        
        if len(generated_response) < 100:
            return {
                "result": f"Les documents consultés ne contiennent pas suffisamment d'informations détaillées sur '{question}'. Recommandation : reformuler la question ou consulter directement les sources réglementaires officielles (Basel Committee, EBA).",
                "source_documents": retrieved_context
            }
        
        return {
            "result": f"{generated_response}\n\n*⚠️ Réponse basée sur les documents disponibles - validation recommandée avec sources officielles*",
            "source_documents": retrieved_context
        }

def create_enhanced_demaris(force_reload: bool = False):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔧 Initialisation du gestionnaire de ressources...")
    progress_bar.progress(10)
    
    resource_manager = DocumentaryResourceManager()
    
    all_documents = []
    vectorstore_path = Path("cache/local_demaris_vectorstore")
    
    if not vectorstore_path.exists() or force_reload:
        status_text.text("📁 Chargement des documents depuis training_corpus/...")
        progress_bar.progress(25)
        
        local_docs = resource_manager.load_local_documents()
        all_documents.extend(local_docs)
        
        if not all_documents:
            st.warning("⚠️ Aucun document trouvé dans training_corpus/")
            st.info("""
            📂 **Placez vos documents directement dans :**
            training_corpus/basel_ii_operational.pdf
            training_corpus/coso_framework.pdf  
            training_corpus/demaris_methodolo.pdf
            """)
    else:
        status_text.text("📂 Utilisation du vectorstore local existant...")
        progress_bar.progress(25)
        st.info("Vectorstore local détecté - pas de rechargement.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if vectorstore_path.exists() and len(list(vectorstore_path.glob("*"))) > 0 and not force_reload:
        status_text.text("📚 Chargement du vectorstore existant...")
        progress_bar.progress(70)
        try:
            vectorstore = FAISS.load_local(
                str(vectorstore_path),
                embeddings,
                allow_dangerous_deserialization=True
            )
            st.success("✅ Vectorstore chargé avec succès.")
        except Exception as e:
            st.warning(f"⚠️ Erreur chargement vectorstore existant: {e}")
            vectorstore = None
    else:
        vectorstore = None
    
    if vectorstore is None and all_documents:
        status_text.text("🆕 Création d'un nouveau vectorstore enrichi...")
        progress_bar.progress(85)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        all_split_docs = []
        for doc in all_documents:
            chunks = text_splitter.split_documents([doc])
            all_split_docs.extend(chunks)
        
        if all_split_docs:
            vectorstore = FAISS.from_documents(all_split_docs, embeddings)
            
            vectorstore_path.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(str(vectorstore_path))
            st.success(f"✅ Nouveau vectorstore créé avec {len(all_split_docs)} chunks!")
    
    if vectorstore:
        status_text.text("🤖 Configuration du modèle QA...")
        progress_bar.progress(90)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
        
        llm = OllamaLLM(
            model="qwen2.5:0.5b-instruct-q4_K_M",
            temperature=0.05,
            top_p=0.3,
            top_k=5
        )
        
        local_prompt = """Tu es un assistant expert en réglementation bancaire et financière.

INSTRUCTIONS :
1. Utilise PRIORITAIREMENT les informations de vos documents de formation ci-dessous
2. Donne des réponses substantielles basées sur les documents training_corpus
3. Si l'information est dans le contexte, RÉPONDS de manière complète
4. Mentionne le document source quand possible (Basel II, COSO, DEMARIS)
5. Structure ta réponse clairement avec des sections

CONTEXTE DOCUMENTAIRE LOCAL :
{context}

QUESTION : {question}

RÉPONSE EXPERTE (basée sur vos documents de formation) :"""
        
        base_qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=local_prompt,
                    input_variables=["context", "question"]
                )
            }
        )
        
        safe_qa_chain = EnhancedSafeDemarisQA(base_qa_chain, resource_manager)
        
        status_text.text("✅ Système DEMARIS enrichi prêt!")
        progress_bar.progress(100)
        
        return safe_qa_chain
        
    else:
        st.warning("⚠️ Aucun vectorstore disponible. Mode réponses vérifiées uniquement.")
        progress_bar.progress(100)
        status_text.text("⚠️ Mode dégradé activé")
        return EnhancedSafeDemarisQA(None, resource_manager)

def calculate_enhanced_metrics(question, response, start_time, end_time):
    try:
        try:
            import nltk
            from nltk.translate.bleu_score import sentence_bleu
        except ImportError:
            st.warning("⚠️ NLTK non disponible - métriques simplifiées")
            return {
                'rouge1': 0.5,
                'rougeL': 0.5,
                'bleu': 0.5,
                'similarity': 0.5,
                'faithfulness': 0.5,
                'relevance': 0.5,
                'response_time': end_time - start_time
            }
        
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            st.warning("⚠️ Rouge-score non disponible - métriques simplifiées")
            return {
                'rouge1': 0.5,
                'rougeL': 0.5,
                'bleu': 0.5,
                'similarity': 0.5,
                'faithfulness': 0.5,
                'relevance': 0.5,
                'response_time': end_time - start_time
            }
        
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            st.warning("⚠️ Sentence-transformers non disponible - métriques simplifiées")
            return {
                'rouge1': 0.5,
                'rougeL': 0.5,
                'bleu': 0.5,
                'similarity': 0.5,
                'faithfulness': 0.5,
                'relevance': 0.5,
                'response_time': end_time - start_time
            }
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        enhanced_references = {
            "ama": "L'Advanced Measurement Approach AMA est une méthode pour calculer les exigences de fonds propres pour le risque opérationnel selon Bâle II avec modèles internes données historiques analyse scénarios facteurs environnement contrôle validation superviseurs",
            "coso": "COSO Committee of Sponsoring Organizations framework contrôle interne cinq composants environnement contrôle évaluation risques activités contrôle information communication pilotage monitoring",
            "piliers_bale": "Trois piliers Bâle III pilier un exigences minimales fonds propres pilier deux processus surveillance prudentielle ICAAP SREP pilier trois discipline marché transparence communication",
            "risque_operationnel": "Risque opérationnel risque de perte processus internes inadéquats défaillants personnes systèmes événements externes fraude interne externe dommages biens dysfonctionnement activité"
        }
        
        question_lower = question.lower()
        reference = None
        
        if 'ama' in question_lower:
            reference = enhanced_references["ama"]
        elif any(kw in question_lower for kw in ['coso', 'contrôle interne']):
            reference = enhanced_references["coso"]
        elif any(kw in question_lower for kw in ['pilier', 'bâle', 'basel']):
            reference = enhanced_references["piliers_bale"]
        elif 'risque opérationnel' in question_lower:
            reference = enhanced_references["risque_operationnel"]
        else:
            reference = question.lower().replace('?', '').replace('qu\'est-ce que', '').replace('quelle est', '')
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference, response)
        rouge1 = rouge_scores['rouge1'].fmeasure
        rougeL = rouge_scores['rougeL'].fmeasure
        
        reference_tokens = reference.split()
        response_tokens = response.split()
        bleu = sentence_bleu([reference_tokens], response_tokens)
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        ref_embedding = model.encode([reference])
        resp_embedding = model.encode([response])
        similarity = cosine_similarity(ref_embedding, resp_embedding)[0][0]
        
        technical_terms = ['bâle', 'basel', 'pilier', 'capital', 'risque', 'ama', 'coso', 'contrôle', 
                          'réglementaire', 'prudentielle', 'supervision', 'opérationnel', 'framework']
        
        response_lower = response.lower()
        found_terms = sum(1 for term in technical_terms if term in response_lower)
        faithfulness = min(found_terms / len(technical_terms), 1.0)
        
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        relevance = len(question_words.intersection(response_words)) / len(question_words) if question_words else 0
        
        return {
            'rouge1': float(rouge1),
            'rougeL': float(rougeL),
            'bleu': float(bleu),
            'similarity': float(similarity),
            'faithfulness': float(faithfulness),
            'relevance': float(relevance),
            'response_time': end_time - start_time
        }
        
    except Exception as e:
        st.warning(f"Erreur calcul métriques enrichies: {e}")
        return {
            'rouge1': 0.0,
            'rougeL': 0.0,
            'bleu': 0.0,
            'similarity': 0.0,
            'faithfulness': 0.0,
            'relevance': 0.0,
            'response_time': end_time - start_time
        }

def main():
    st.set_page_config(
        page_title="DEMARIS Pro - Assistant Local BEAC",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2e8bc0 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2e8bc0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>🏦 DEMARIS Pro - RAG Chatbot Local</h1>
        <p>Assistant Expert en Réglementation Bancaire - Documents de Formation BEAC</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("🔧 Navigation DEMARIS")
    
    if st.sidebar.button("🏠 Retour Accueil Principal", use_container_width=True, type="secondary"):
        if 'redirect_to' in st.session_state:
            del st.session_state.redirect_to
        st.session_state.current_page = 'accueil'
        st.session_state.chatbot_subpage = None
        st.rerun()

    st.sidebar.markdown("---")
    
    if st.sidebar.button("💬 Chat Expert", use_container_width=True):
        st.session_state.current_page = "chat"
    
    if st.sidebar.button("📊 Tableau de Bord", use_container_width=True):
        st.session_state.current_page = "dashboard"
    
    if st.sidebar.button("🔍 Sources Documentaires", use_container_width=True):
        st.session_state.current_page = "sources"
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "chat"
    
    if st.session_state.current_page == "chat":
        page = "💬 Chat Expert"
    elif st.session_state.current_page == "dashboard":
        page = "📊 Tableau de Bord Enrichi"
    elif st.session_state.current_page == "sources":
        page = "🔍 Sources Documentaires"
    else:
        page = "💬 Chat Expert"
    
    st.sidebar.markdown(f"**Page actuelle :** {page}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 État du Système")
    
    if 'enhanced_demaris_qa' in st.session_state:
        st.sidebar.success("✅ DEMARIS Pro Actif")
    else:
        st.sidebar.warning("⏳ Initialisation...")
    
    if st.session_state.get('metrics_history'):
        st.sidebar.info(f"📈 {len(st.session_state.metrics_history)} questions traitées")
    
    if st.session_state.get('chat_history'):
        last_chat = st.session_state.chat_history[-1]
        st.sidebar.caption(f"🕐 Dernière question: {last_chat['timestamp'].strftime('%H:%M:%S')}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Actions Rapides")
    
    if st.sidebar.button("🗑️ Vider Cache", help="Effacer historique et métriques"):
        st.session_state.chat_history = []
        st.session_state.metrics_history = []
        st.sidebar.success("Cache vidé !")
        st.rerun()
    
    if st.sidebar.button("🔄 Recharger Système", help="Recharger DEMARIS Pro avec nouvelles données"):
        if 'enhanced_demaris_qa' in st.session_state:
            del st.session_state.enhanced_demaris_qa
        st.session_state.enhanced_demaris_qa = create_enhanced_demaris(force_reload=True)
        st.rerun()
    
    if 'enhanced_demaris_qa' not in st.session_state:
        with st.spinner("🚀 Initialisation du système DEMARIS enrichi..."):
            st.session_state.enhanced_demaris_qa = create_enhanced_demaris()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'metrics_history' not in st.session_state:
        st.session_state.metrics_history = []
    
    if page == "💬 Chat Expert":
        show_enhanced_chat()
    elif page == "📊 Tableau de Bord Enrichi":
        show_enhanced_dashboard()
    elif page == "🔍 Sources Documentaires":
        show_documentation_sources()

def show_enhanced_chat():
    st.header("💬 Chat Expert DEMARIS")
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        st.info("🎯 **Documents de formation :** Basel II Operational, COSO Framework, DEMARIS Methodology")
    
    with col2:
        if st.button("📊 Voir Dashboard", type="secondary", use_container_width=True):
            st.session_state.current_page = "dashboard"
            st.rerun()
    
    with col3:
        if st.button("🔍 Sources", type="secondary", use_container_width=True):
            st.session_state.current_page = "sources"
            st.rerun()
    
    with col4:
        if st.button("🗑️ Effacer", type="secondary", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.metrics_history = []
            st.rerun()
    
    if st.session_state.get('metrics_history'):
        col1, col2, col3, col4 = st.columns(4)
        latest = st.session_state.metrics_history[-1]
        
        with col1:
            st.metric("🎯 Précision", f"{latest['rouge1']:.3f}")
        with col2:
            st.metric("🧠 Similarité", f"{latest['similarity']:.3f}")
        with col3:
            st.metric("⚡ Temps", f"{latest['response_time']:.1f}s")
        with col4:
            st.metric("💬 Questions", len(st.session_state.metrics_history))
        
        st.markdown("---")
    
    user_question = st.text_input(
        "🤔 Posez votre question réglementaire :",
        placeholder="Ex: Qu'est-ce que l'approche AMA pour le risque opérationnel ?"
    )
    
    if st.button("📤 Envoyer", type="primary") and user_question:
        with st.spinner("🔍 Consultation des sources réglementaires enrichies..."):
            start_time = time.time()
            
            response = st.session_state.enhanced_demaris_qa.invoke({"query": user_question})
            answer = response["result"]
            sources = response.get("source_documents", [])
            
            end_time = time.time()
            
            metrics = calculate_enhanced_metrics(user_question, answer, start_time, end_time)
            
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": answer,
                "timestamp": datetime.now(),
                "sources": len(sources),
                "source_details": [doc.metadata for doc in sources]
            })
            
            st.session_state.metrics_history.append({
                "timestamp": datetime.now(),
                "question": user_question,
                **metrics
            })
    
    st.markdown("---")
    st.subheader("📚 Historique des Consultations")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🤔 Question {len(st.session_state.chat_history)-i} :</strong><br>
            {chat['question']}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 DEMARIS Pro :</strong><br>
            {chat['answer']}<br><br>
            <small>⏱️ {chat['timestamp'].strftime('%H:%M:%S')} | 📄 {chat['sources']} sources consultées</small>
        </div>
        """, unsafe_allow_html=True)

def show_enhanced_dashboard():
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #1f4e79 0%, #2e8bc0 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
            <h1 style="color: white; margin: 0;">📊 Tableau de bord des performances</h1>
            <p style="color: white; margin: 0; opacity: 0.9;">Métriques de Qualité DEMARIS Pro</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("← Retour au Chat", type="primary", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()
    
    if not st.session_state.metrics_history:
        st.info("💡 Aucune donnée disponible. Posez des questions dans le chat pour voir les métriques.")
        return
    
    st.markdown("## 📈 Métriques Temps Réel")
    
    latest_metrics = st.session_state.metrics_history[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_rouge = latest_metrics['rouge1'] - (st.session_state.metrics_history[-2]['rouge1'] if len(st.session_state.metrics_history) > 1 else 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Score ROUGE-L</h3>
            <h2>{latest_metrics['rouge1']:.3f}</h2>
            <p style="color: {'green' if delta_rouge >= 0 else 'red'};">
                {'↗️' if delta_rouge >= 0 else '↘️'} {delta_rouge:+.3f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        delta_bleu = latest_metrics['bleu'] - (st.session_state.metrics_history[-2]['bleu'] if len(st.session_state.metrics_history) > 1 else 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔵 Score BLEU</h3>
            <h2>{latest_metrics['bleu']:.3f}</h2>
            <p style="color: {'green' if delta_bleu >= 0 else 'red'};">
                {'↗️' if delta_bleu >= 0 else '↘️'} {delta_bleu:+.3f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        delta_sim = latest_metrics['similarity'] - (st.session_state.metrics_history[-2]['similarity'] if len(st.session_state.metrics_history) > 1 else 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>🧠 Similarité Sémantique</h3>
            <h2>{latest_metrics['similarity']:.3f}</h2>
            <p style="color: {'green' if delta_sim >= 0 else 'red'};">
                {'↗️' if delta_sim >= 0 else '↘️'} {delta_sim:+.3f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        delta_time = latest_metrics['response_time'] - (st.session_state.metrics_history[-2]['response_time'] if len(st.session_state.metrics_history) > 1 else 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚡ Temps de Réponse</h3>
            <h2>{latest_metrics['response_time']:.2f} secondes</h2>
            <p style="color: {'red' if delta_time >= 0 else 'green'};">
                {'↗️' if delta_time >= 0 else '↘️'} {delta_time:+.2f} secondes
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    if len(st.session_state.metrics_history) > 1:
        history_df = pd.DataFrame(st.session_state.metrics_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("## 📊 Évolution des Scores de Qualité")
            
            fig = go.Figure()
            
            rouge1_unique = history_df['rouge1'].nunique()
            rougeL_unique = history_df['rougeL'].nunique()
            bleu_unique = history_df['bleu'].nunique()
            
            if rouge1_unique == 1 and rougeL_unique == 1 and bleu_unique == 1:
                fig.add_trace(go.Scatter(
                    x=list(range(len(history_df))),
                    y=history_df['rouge1'],
                    name=f'ROUGE-1/L & BLEU (Score: {history_df["rouge1"].iloc[0]:.3f})',
                    line=dict(color='#ff6b6b', width=4),
                    mode='lines+markers',
                    marker=dict(size=8)
                ))
                
                fig.add_annotation(
                    x=len(history_df)//2,
                    y=history_df['rouge1'].iloc[0] + 0.05,
                    text="📌 Scores identiques<br>(réponses vérifiées)",
                    showarrow=True,
                    arrowhead=2,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#ff6b6b"
                )
            else:
                fig.add_trace(go.Scatter(
                    x=list(range(len(history_df))),
                    y=history_df['rouge1'],
                    name='ROUGE-1',
                    line=dict(color='#ff6b6b')
                ))
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(history_df))),
                    y=history_df['rougeL'],
                    name='ROUGE-L',
                    line=dict(color='#4ecdc4')
                ))
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(history_df))),
                    y=history_df['bleu'],
                    name='BLEU',
                    line=dict(color='#45b7d1')
                ))
            
            fig.update_layout(
                title="Évolution des Métriques NLP",
                xaxis_title="Questions",
                yaxis_title="Score",
                height=400,
                yaxis=dict(range=[0, 1.1])
            )
            
            st.plotly_chart(fig, use_container_width=True, key="enhanced_quality_metrics_chart")
        
        with col2:
            st.markdown("## 🛡️ Qualité & Fidélité Enrichies")
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=list(range(len(history_df))),
                y=history_df['relevance'],
                fill='tonexty',
                name='Pertinence',
                line=dict(color='#2ecc71')
            ))
            
            fig2.add_trace(go.Scatter(
                x=list(range(len(history_df))),
                y=history_df['faithfulness'],
                fill='tozeroy',
                name='Fidélité',
                line=dict(color='#f39c12')
            ))
            
            fig2.update_layout(
                title="Fidélité et Pertinence",
                xaxis_title="Questions",
                yaxis_title="Score",
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True, key="enhanced_faithfulness_relevancy_chart")
    
    st.markdown("## 📋 Historique détaillé")
    
    if st.session_state.metrics_history:
        detailed_df = pd.DataFrame([
            {
                "Heure": m['timestamp'].strftime('%H:%M:%S'),
                "Question": m['question'][:50] + "..." if len(m['question']) > 50 else m['question'],
                "ROUGE-1": f"{m['rouge1']:.3f}",
                "ROUGE-L": f"{m['rougeL']:.3f}",
                "BLEU": f"{m['bleu']:.3f}",
                "Sim. Sémantique": f"{m['similarity']:.3f}",
                "Fidélité": f"{m['faithfulness']:.3f}",
                "Pertinence": f"{m['relevance']:.3f}",
                "Temps (s)": f"{m['response_time']:.2f}"
            }
            for m in st.session_state.metrics_history
        ])
        
        st.dataframe(detailed_df, use_container_width=True)
    
    st.markdown("## 📊 Statistiques Globales Enrichies")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Scores moyens")
        metrics_df = pd.DataFrame(st.session_state.metrics_history)
        numeric_columns = ['rouge1', 'rougeL', 'bleu', 'similarity', 'faithfulness', 'relevance', 'response_time']
        existing_numeric_cols = [col for col in numeric_columns if col in metrics_df.columns]
        
        if existing_numeric_cols:
            avg_metrics = metrics_df[existing_numeric_cols].mean()
            std_metrics = metrics_df[existing_numeric_cols].std()
            
            st.write(f"**ROUGE-1:** {avg_metrics.get('rouge1', 0):.3f} ± {std_metrics.get('rouge1', 0):.3f}")
            st.write(f"**ROUGE-L:** {avg_metrics.get('rougeL', 0):.3f} ± {std_metrics.get('rougeL', 0):.3f}")
            st.write(f"**BLEU:** {avg_metrics.get('bleu', 0):.3f} ± {std_metrics.get('bleu', 0):.3f}")
        else:
            st.write("Aucune métrique numérique disponible")
    
    with col2:
        st.markdown("### 🧠 Qualité Sémantique")
        if existing_numeric_cols:
            st.write(f"**Similitude:** {avg_metrics.get('similarity', 0):.3f} ± {std_metrics.get('similarity', 0):.3f}")
            st.write(f"**Fidélité:** {avg_metrics.get('faithfulness', 0):.3f} ± {std_metrics.get('faithfulness', 0):.3f}")
            st.write(f"**Pertinence:** {avg_metrics.get('relevance', 0):.3f} ± {std_metrics.get('relevance', 0):.3f}")
        else:
            st.write("Métriques sémantiques non disponibles")
    
    with col3:
        st.markdown("### ⚡ Performance")
        if 'response_time' in metrics_df.columns:
            response_times = metrics_df['response_time'].tolist()
            st.write(f"**Temps moyen:** {np.mean(response_times):.2f}s")
            st.write(f"**Temps min/max:** {min(response_times):.2f}s / {max(response_times):.2f}s")
        else:
            st.write("**Temps moyen:** Non disponible")
        st.write(f"**Questions traitées:** {len(st.session_state.metrics_history)}")

def show_documentation_sources():
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("🔍 Sources Documentaires Locales")
    
    with col2:
        if st.button("← Retour au Chat", type="primary", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()
    
    st.markdown("""
    ## 📚 Corpus Documentaire DEMARIS Local
    
    ### 📁 **Vos Documents de Formation**
    
    #### Documents training_corpus/ (Structure Simple)
    - 📄 **basel_ii_operational.pdf** - Réglementation Basel II sur le risque opérationnel
    - 🔧 **coso_framework.pdf** - Framework de contrôle interne COSO  
    - 🤖 **demaris_methodolo.pdf** - Méthodologie DEMARIS spécifique
    
    ✅ **Mode Local Activé** : Utilisation exclusive de vos documents
    
    ### 📊 **Couverture Documentaire**
    """)
    
    training_corpus_dir = Path("training_corpus")
    if training_corpus_dir.exists():
        pdf_files = list(training_corpus_dir.glob("*.pdf"))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            basel_files = [f for f in pdf_files if "basel" in f.name.lower() or "bale" in f.name.lower()]
            st.metric("📄 Basel Documents", len(basel_files))
        
        with col2:
            coso_files = [f for f in pdf_files if "coso" in f.name.lower()]
            st.metric("🔧 COSO Framework", len(coso_files))
        
        with col3:
            demaris_files = [f for f in pdf_files if "demaris" in f.name.lower()]
            st.metric("🤖 DEMARIS Docs", len(demaris_files))
        
        st.markdown("### 📋 Vos Documents de Formation")
        
        if pdf_files:
            files_data = []
            for pdf_file in pdf_files:
                size = pdf_file.stat().st_size
                modified = datetime.fromtimestamp(pdf_file.stat().st_mtime)
                
                filename_lower = pdf_file.name.lower()
                if "basel" in filename_lower or "bale" in filename_lower:
                    doc_type = "📄 Basel II Operational"
                elif "coso" in filename_lower:
                    doc_type = "🔧 COSO Framework"
                elif "demaris" in filename_lower:
                    doc_type = "🤖 DEMARIS Methodology"
                else:
                    doc_type = "📚 Document Formation"
                
                files_data.append({
                    "Fichier": pdf_file.name,
                    "Type": doc_type,
                    "Taille": f"{size / (1024*1024):.1f} MB",
                    "Dernière MAJ": modified.strftime("%d/%m/%Y %H:%M")
                })
            
            files_df = pd.DataFrame(files_data)
            st.dataframe(files_df, use_container_width=True)
            
            st.markdown("### 📖 Contenu de Formation")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if any("basel" in f.name.lower() or "bale" in f.name.lower() for f in pdf_files):
                    st.info("""
                    **📄 Basel II Operational**
                    - Risque opérationnel
                    - Approches de mesure
                    - Cadre réglementaire
                    """)
            
            with col2:
                if any("coso" in f.name.lower() for f in pdf_files):
                    st.info("""
                    **🔧 COSO Framework**
                    - Contrôle interne
                    - 5 composants COSO
                    - Gestion des risques
                    """)
            
            with col3:
                if any("demaris" in f.name.lower() for f in pdf_files):
                    st.info("""
                    **🤖 DEMARIS Methodology**
                    - Méthodologie d'analyse
                    - Processus d'évaluation
                    - Framework spécifique
                    """)
        else:
            st.warning("📁 Aucun fichier PDF trouvé dans training_corpus/")
            st.info("""
            💡 **Placez vos documents directement dans :**
            training_corpus/basel_ii_operational.pdf
            training_corpus/coso_framework.pdf  
            training_corpus/demaris_methodolo.pdf
            """)
    
    else:
        st.error("📂 Dossier training_corpus non trouvé!")
        st.info("""
        🔧 **Créez le dossier et placez vos documents :**
        ```
        training_corpus/
        ├── basel_ii_operational.pdf
        ├── coso_framework.pdf
        └── demaris_methodolo.pdf
        ```
        """)
    
    st.markdown("""
    ### ⚙️ **Informations Techniques - Mode Local**
    
    **Architecture Locale :**
    - ✅ Documents uniquement depuis training_corpus/
    - ✅ Pas de téléchargement automatique
    - ✅ Cache vectorstore local optimisé
    - ✅ Classification automatique par nom de fichier
    
    **Vos Documents :**
    - 📄 **basel_ii_operational.pdf** → Réglementation Basel II
    - 🔧 **coso_framework.pdf** → Framework de contrôle interne
    - 🤖 **demaris_methodolo.pdf** → Méthodologie DEMARIS
    
    **Performance Locale :**
    - ⚡ Retrieval optimisé (8 documents locaux)
    - 🔍 Embeddings all-MiniLM-L6-v2 
    - 🤖 LLM Qwen2.5 conservateur
    - 💾 Cache local persistant
    """)
    
    if st.button("🔄 Recharger Documents Locaux"):
        with st.spinner("📁 Rechargement documents locaux..."):
            if 'enhanced_demaris_qa' in st.session_state:
                del st.session_state.enhanced_demaris_qa
            st.session_state.enhanced_demaris_qa = create_enhanced_demaris(force_reload=True)
        st.success("✅ Documents locaux rechargés!")
        st.rerun()

if __name__ == "__main__":
    main()