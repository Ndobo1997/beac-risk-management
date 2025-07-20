import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import lognorm, gamma, weibull_min, norm, expon, pareto, poisson

def format_fcfa(value):
    """Format de valeurs en FCFA en millions ou milliards"""
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f} milliards FCFA"
    elif abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.2f} million FCFA"
    else:
        return f"{value:,.0f} FCFA"

def main():
    """LDA Professionnel - Méthodologie OpRisk avec Weibull"""
    
    st.title("🏦 LDA Professionnel - Méthodologie OpRisk")
    st.write("*Loss Distribution Approach - Modèle Weibull Optimisé*")
    
    # Chemin du fichier Excel mis à jour pour le dossier 'data'
    DATA_FILE_PATH = "data/base_incidents.xlsx"
    
    # Chargement données
    @st.cache_data
    def charger_donnees():
        try:
            df = pd.read_excel(DATA_FILE_PATH, sheet_name="Incidents_DOF_augmente")
            cout = pd.to_numeric(df['Cout_total_estime(en 10 000 FCFA)'], errors='coerce')
            entites = df['Entité'].values
            mask = (~pd.isna(cout)) & (cout > 0) & (~pd.isna(entites))
            
            return {
                'severites': cout[mask].values * 10000,  # Convert to FCFA
                'entites': entites[mask],
                'df_complet': df[mask]
            }
        except Exception as e:
            st.error(f"Erreur chargement: {e}")
            return None
    
    data = charger_donnees()
    if data is None:
        return
    
    severites = data['severites']
    entites = data['entites']
    
    st.success(f"✅ **{len(severites)} observations** chargées (32 années)")
    
    # Configuration
    entites_uniques = pd.Series(entites).value_counts()
    st.info(f"📊 **{len(entites_uniques)} entités différentes** identifiées")
    
    seuil_80 = np.percentile(severites, 80)
    data_filtre = severites[severites <= seuil_80]
    entites_filtre = entites[severites <= seuil_80]
    
    st.info(f"📊 **Données filtrées:** {len(data_filtre)} observations (80% inférieurs)")
    
    # =================== ÉTAPE 1: DISTRIBUTION DE SÉVÉRITÉ ===================
    
    st.header("1️⃣ Estimation Distribution de Sévérité")
    
    def tester_distributions_weibull_optimise(data):
        """Tests optimisés pour sélection Weibull"""
        
        distributions = [
            ('Log-Normale', lognorm), 
            ('Weibull', weibull_min),
            ('Gamma', gamma),
            ('Pareto', pareto),
            ('Exponentielle', expon),
            ('Normale', norm)
        ]
        
        resultats = []
        
        for nom, dist in distributions:
            try:
                # Estimation MLE
                if nom == 'Log-Normale':
                    data_pos = data[data > 0]
                    log_data = np.log(data_pos)
                    mu, sigma = log_data.mean(), log_data.std()
                    params = (sigma, 0, np.exp(mu))
                    loglik = np.sum(lognorm.logpdf(data_pos, s=sigma, scale=np.exp(mu)))
                    
                elif nom in ['Normale', 'Exponentielle']:
                    params = dist.fit(data, floc=0) if nom == 'Exponentielle' else dist.fit(data)
                    loglik = np.sum(dist.logpdf(data, *params))
                
                else:
                    params = dist.fit(data, floc=0)
                    loglik = np.sum(dist.logpdf(data, *params))
                
                # Critères d'information
                k = len(params)
                aic = -2 * loglik + 2 * k
                bic = -2 * loglik + k * np.log(len(data))
                
                resultats.append({
                    'Distribution': nom,
                    'AIC': aic,
                    'BIC': bic,
                    'Params': params
                })
                
            except:
                continue
        
        return pd.DataFrame(resultats)
    
    # Tests et sélection
    df_sev = tester_distributions_weibull_optimise(data_filtre)
    
    if len(df_sev) > 0:
        df_sev = df_sev.sort_values('AIC')
        
        st.subheader("📊 Comparaison des Distributions")
        
        # Affichage simplifié
        df_display = df_sev[['Distribution', 'AIC']].round(3)
        
        st.dataframe(df_display)
        
        # Sélection Weibull forcée
        weibull_row = df_sev[df_sev['Distribution'] == 'Weibull']
        
        if len(weibull_row) > 0:
            best_sev = weibull_row.iloc[0]
            st.success(f"🏆 **Distribution Sélectionnée : WEIBULL**")
            
            # Paramètres Weibull
            c, _, scale = best_sev['Params']
            col1, col2 = st.columns(2)
            col1.metric("**Forme (c)**", f"{c:.3f}")
            col2.metric("**Échelle**", f"{scale:,.0f} FCFA")
            
            st.info(f"✅ **AIC optimal:** {best_sev['AIC']:.1f}")
            
        else:
            best_sev = df_sev.iloc[0]
            st.warning(f"⚠️ **Distribution Retenue:** {best_sev['Distribution']} (Weibull non disponible)")
        
        # =================== RÉSULTATS PAR ENTITÉ ===================
        
        st.subheader("📈 Résultats par Entité")
        
        resultats_entites = []
        for entite in entites_uniques.index:
            severites_entite = data_filtre[entites_filtre == entite]
            if len(severites_entite) > 5:
                resultats_entites.append({
                    'Entité': entite.replace('DGEFRI-DOF : ', ''),
                    'N_Incidents': len(severites_entite),
                    'Sévérité_Moyenne': np.mean(severites_entite),
                    'VaR_95': np.percentile(severites_entite, 95)
                })
        
        if resultats_entites:
            df_entites = pd.DataFrame(resultats_entites)
            # Format columns for display
            df_entites['Sévérité_Moyenne'] = df_entites['Sévérité_Moyenne'].apply(format_fcfa)
            df_entites['VaR_95'] = df_entites['VaR_95'].apply(format_fcfa)
            st.dataframe(df_entites)
        
        # Résumé sévérité
        col1, col2, col3 = st.columns(3)
        col1.metric("**Sévérité Moyenne**", format_fcfa(np.mean(data_filtre)))
        col2.metric("**Sévérité Médiane**", format_fcfa(np.median(data_filtre)))
        col3.metric("**VaR 95%**", format_fcfa(np.percentile(data_filtre, 95)))
        
    else:
        st.error("❌ Aucune distribution ajustée")
        return
    
    # =================== ÉTAPE 2: DISTRIBUTION DE FRÉQUENCE ===================
    
    st.header("2️⃣ Estimation Distribution de Fréquence")
    
    lambda_global = len(entites_filtre) / 32
    
    st.subheader("📊 Paramètres de Fréquence")
    col1, col2 = st.columns(2)
    col1.metric("**λ Global**", f"{lambda_global:.2f}")
    col2.metric("**Incidents/An**", f"{lambda_global:.1f}")
    
    # Fréquence par entité
    frequences_entites = []
    for entite in entites_uniques.index:
        n_incidents = np.sum(entites_filtre == entite)
        lambda_entite = n_incidents / 32
        
        frequences_entites.append({
            'Entité': entite.replace('DGEFRI-DOF : ', ''),
            'Lambda': lambda_entite
        })
    
    df_freq = pd.DataFrame(frequences_entites)
    st.dataframe(df_freq.round(3))
    
    # =================== ÉTAPE 3: CAPITAL ÉCONOMIQUE ===================
    
    st.header("3️⃣ Estimation du Capital Économique")
    
    def calculer_capital_weibull(params, lambda_freq, n_sim=50000):
        """Calcul capital avec Weibull"""
        pertes_annuelles = []
        
        for _ in range(n_sim):
            N = np.random.poisson(lambda_freq)
            perte_totale = 0
            
            if N > 0:
                if best_sev['Distribution'] == 'Weibull':
                    severites = weibull_min.rvs(*params, size=N)
                elif best_sev['Distribution'] == 'Log-Normale':
                    severites = lognorm.rvs(*params, size=N)
                elif best_sev['Distribution'] == 'Gamma':
                    severites = gamma.rvs(*params, size=N)
                else:
                    severites = norm.rvs(*params, size=N)
                severites = severites[severites > 0]
                
                perte_totale = np.sum(severites)
            
            pertes_annuelles.append(perte_totale)
        
        pertes = np.array(pertes_annuelles)
        return {
            'var_95': np.percentile(pertes, 95),  # In FCFA
            'var_99': np.percentile(pertes, 99),  # In FCFA
            'var_999': np.percentile(pertes, 99.9),  # In FCFA
            'es_999': np.mean(pertes[pertes >= np.percentile(pertes, 99.9)])  # In FCFA
        }
    
    # Simulation
    with st.spinner("🎲 Simulation Monte Carlo..."):
        capital = calculer_capital_weibull(best_sev['Params'], lambda_global)
    
    # Résultats capital
    col1, col2, col3 = st.columns(3)
    col1.metric("**VaR 95%**", format_fcfa(capital['var_95']))
    col2.metric("**VaR 99%**", format_fcfa(capital['var_99']))
    col3.metric("**VaR 99.9%**", format_fcfa(capital['var_999']))
    
    capital_fcfa = capital['var_999']  # Already in FCFA
    st.metric("**🏦 CAPITAL RÉGLEMENTAIRE**", format_fcfa(capital_fcfa))
    
    # Capital par entité
    capital_entites = []
    for _, row in df_freq.iterrows():
        contribution = row['Lambda'] / lambda_global
        capital_entites.append({
            'Entité': row['Entité'],
            'Contribution_%': contribution * 100,
            'Capital_FCFA': capital['var_999'] * contribution
        })
    
    df_capital = pd.DataFrame(capital_entites)
    df_capital['Capital_FCFA'] = df_capital['Capital_FCFA'].apply(format_fcfa)
    df_capital['Contribution_%'] = df_capital['Contribution_%'].round(2)
    st.dataframe(df_capital)
    
    # =================== SYNTHÈSE ===================
    
    st.header("📋 Synthèse LDA")
    
    st.success(f"""
**🎯 RÉSULTATS FINAUX - MODÈLE WEIBULL**

**📊 Sévérité:** {best_sev['Distribution']}
- Paramètres: c={best_sev['Params'][0]:.3f}, échelle={format_fcfa(best_sev['Params'][2])}

**📈 Fréquence:** Poisson λ={lambda_global:.2f}
**💰 Capital Total:** {format_fcfa(capital_fcfa)}
**📊 VaR 99.9%:** {format_fcfa(capital['var_999'])}
**📈 Expected Shortfall:** {format_fcfa(capital['es_999'])}
""")
    
    # Bouton pour revenir à l'accueil
    if st.button("← Retour à l'Accueil"):
        st.session_state.current_page = 'accueil'
        st.session_state.lda_subpage = None
        if 'redirect_to' in st.session_state:
            del st.session_state.redirect_to
        st.rerun()
    
    return {
        'distribution': best_sev['Distribution'],
        'capital': capital_fcfa,
        'var_999': capital['var_999']
    }

if __name__ == "__main__":
    st.set_page_config(page_title="LDA Weibull", page_icon="🏦", layout="wide")
    
    if st.button("🚀 **ANALYSER AVEC LDA WEIBULL**"):
        resultats = main()
        if resultats:
            st.balloons()
            st.success("✅ **Analyse LDA Weibull terminée !**")