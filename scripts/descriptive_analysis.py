import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import geopandas as gpd
# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
DOSSIER_DONNEES = os.path.join(script_dir, "..", "data", "output","global_data" )
DOSSIER_OUTPUT = os.path.join(script_dir, "..", "data", "output","description" )

PATH_SHP = r"D:\GeoDataScience\Statistiques\Notre Projet\donnees-mobilite_INSEE\ADMIN-EXPRESS-COG_3-0__SHP_LAMB93_FXX_2021-05-19\ADMIN-EXPRESS-COG_3-0__SHP_LAMB93_FXX_2021-05-19\ADMIN-EXPRESS-COG\1_DONNEES_LIVRAISON_2021-05-19\ADECOG_3-0_SHP_LAMB93_FXX\COMMUNE.shp"
gdf_communes = gpd.read_file(PATH_SHP)
if 'INSEE_COM' in gdf_communes.columns:
    gdf_communes = gdf_communes.rename(columns={'INSEE_COM': 'code_insee'})

# Noms des modes pour l'affichage
MODE_LABELS = {
    'TRANS_MP': 'Marche',
    'TRANS_TC': 'Transports en Commun',
    'TRANS_VT': 'Voiture',
    'TRANS_DRM': 'Moto-Scooter',
    'TRANS_VL': 'Vélo'
}

# Couleurs pour les graphiques
COLORS = {
    'TRANS_MP': '#1f77b4',   # Bleu
    'TRANS_TC': '#ff7f0e',   # Orange
    'TRANS_VT': '#d62728',   # Rouge
    'TRANS_DRM': '#7f7f7f',  # Gris
    'TRANS_VL': '#2ca02c'    # Vert
}

# Colonnes attendues dans le fichier analytique
COLS_ANALYTIC = [
    'COM_ORG', 'COM_DEST', 
    'DELTA_PART_MP', 'DELTA_PART_TC', 'DELTA_PART_VT', 'DELTA_PART_DRM', 'DELTA_PART_VL', 
    'DELTA_DUREE'
]

# ==============================================================================
# 2. CHARGEMENT DES DONNÉES
# ==============================================================================

def load_annual_data():
    """Charge les fichiers 2017_data_with_duration.csv à 2022..."""
    all_dfs = []
    print("--- Chargement des fichiers annuels ---")
    
    for year in range(2017, 2023): 
        filename = f"{year}_data_with_duration.csv"
        path = os.path.join(DOSSIER_DONNEES, filename)
        
        if os.path.exists(path):
            print(f"Lecture de {filename}...")
            df = pd.read_csv(path, sep=',') 
            df = df[df['COM_ORG'] != df['COM_DEST']]
            df['TRAJET'] = df['COM_ORG'].astype(str) + " - " + df['COM_DEST'].astype(str)
            df['ANNEE'] = year
            
            cols_modes = list(MODE_LABELS.keys())
            cols_to_keep = ['TRAJET', 'ANNEE', 'TOTAL_FLUX', 'DISTANCE', 'DUREE'] + cols_modes
            
            existing_cols = [c for c in cols_to_keep if c in df.columns]
            all_dfs.append(df[existing_cols])
        else:
            print(f"ATTENTION : Fichier manquant {path}")

    if not all_dfs:
        raise ValueError("Aucun fichier annuel trouvé !")
        
    return pd.concat(all_dfs, ignore_index=True)

def load_analytic_data():
    """Charge analytic_data.csv et assigne les années basées sur l'ordre des lignes"""
    print("\n--- Chargement des données analytiques (Deltas) ---")
    path = os.path.join(DOSSIER_DONNEES, "analytic_data.csv")
    
    
    try:
        df = pd.read_csv(path, sep=',')
    except:
        df = pd.read_csv(path, sep=',', header=None, names=COLS_ANALYTIC)

    print("Assignation des années aux deltas...")
    
    df['seq_id'] = df.groupby(['COM_ORG', 'COM_DEST']).cumcount()
    
    base_year = 2018 
    df['ANNEE_CIBLE'] = base_year + df['seq_id']
    
    df = df[df['ANNEE_CIBLE'] <= 2022]
    
    return df

# ==============================================================================
# 3. ANALYSE ET DASHBOARD
# ==============================================================================

def generate_dashboard(df_vol, df_delta):
    print("\n>>> Génération des graphiques individuels...")
    
    # S'assurer que le dossier de sortie existe
    os.makedirs(DOSSIER_OUTPUT, exist_ok=True)
    
    modes_cols = list(MODE_LABELS.keys())
    
    # --- GRAPH 1 : ÉVOLUTION DU VOLUME (Lignes non-cumulatives) ---
    plt.figure(figsize=(12, 7))
    vol_per_year = df_vol.groupby('ANNEE')[modes_cols].sum()
    
    # On boucle sur chaque mode pour tracer sa propre ligne
    for mode in modes_cols:
        plt.plot(vol_per_year.index, 
                 vol_per_year[mode], 
                 marker='o',                # Ajout de points sur les années
                 linewidth=2.5, 
                 label=MODE_LABELS[mode], 
                 color=COLORS[mode])

    plt.title("1. Évolution du Volume de Flux par Mode (Non-cumulé)", fontweight='bold', fontsize=14)
    plt.ylabel("Flux total")
    plt.xlabel("Année")
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER_OUTPUT, "1_evolution_volume_lignes.png"), dpi=300)
    plt.close()

    # --- GRAPH 2 : PARTS DE MARCHÉ (100% Bar) ---
    plt.figure(figsize=(12, 7))
    pct_per_year = vol_per_year.div(vol_per_year.sum(axis=1), axis=0) * 100
    clean_labels = [MODE_LABELS[m] for m in modes_cols]
    palette = [COLORS[m] for m in modes_cols]
    
    pct_per_year.columns = clean_labels
    ax2 = pct_per_year.plot(kind='bar', stacked=True, color=palette, width=0.8, ax=plt.gca())
    plt.title("2. Évolution des Parts Modales (%)", fontweight='bold', fontsize=14)
    plt.ylabel("%")
    plt.xlabel("Année")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for c in ax2.containers:
        labels = [f'{v.get_height():.1f}%' if v.get_height() > 3 else '' for v in c]
        ax2.bar_label(c, labels=labels, label_type='center', color='white', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER_OUTPUT, "2_parts_modales.png"), dpi=300)
    plt.close()

    # --- GRAPH 3 : TOP FLUX (Pareto) ---
    plt.figure(figsize=(12, 8))
    top_flux = df_vol.groupby('TRAJET')['TOTAL_FLUX'].sum().sort_values(ascending=False).head(15)
    sns.barplot(x=top_flux.values, y=top_flux.index, palette='viridis')
    plt.title("3. Top 15 des Couples de Villes (Flux Cumulés)", fontweight='bold', fontsize=14)
    plt.xlabel("Volume de flux")
    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER_OUTPUT, "3_top_flux.png"), dpi=300)
    plt.close()


def export_statistics(df_delta):
    """Calcule et exporte les stats des deltas"""
    print("\n>>> Calcul des statistiques descriptives...")
    
    cols_to_stat = [c for c in df_delta.columns if 'DELTA_' in c]
    stats_list = []
    
    for col in cols_to_stat:
        data = df_delta[col].dropna()
        stats = {
            'Variable': col,
            'Moyenne': data.mean(),
            'Mediane': data.median(),
            'Ecart_Type': data.std(),
            'Min': data.min(),
            'Max': data.max(),
            '%_Hausse': (data > 0).mean() * 100,
            '%_Baisse': (data < 0).mean() * 100
        }
        stats_list.append(stats)
        
    df_stats = pd.DataFrame(stats_list)
    output_csv = os.path.join(DOSSIER_OUTPUT, "stats_descriptives_deltas.csv")
    df_stats.to_csv(output_csv, index=False, sep=';', decimal=',')
    print(f"✓ Stats exportées : {output_csv}")
def mobility_map(mode, titre, df_flux):
    
    # 1. Identifier toutes les communes présentes dans le CSV (origines et destinations)
    codes_etude = set(df_flux['COM_ORG'].astype(str).unique()) | set(df_flux['COM_DEST'].astype(str).unique())

    # 2. Calcul des Sortants et Entrants
    sortants = df_flux[df_flux['COM_ORG'] != df_flux['COM_DEST']].groupby('COM_ORG')[mode].sum()
    entrants = df_flux[df_flux['COM_ORG'] != df_flux['COM_DEST']].groupby('COM_DEST')[mode].sum()

    # 3. Création du bilan
    stats = pd.DataFrame({'entrants': entrants, 'sortants': sortants}).fillna(0)
    stats['solde'] = stats['entrants'] - stats['sortants']
    stats.index = stats.index.astype(str)

    # 4. Filtrer le fond de carte : On ne garde QUE les communes présentes dans le CSV
    map_data = gdf_communes[gdf_communes['code_insee'].isin(codes_etude)].copy()

    if map_data.empty:
        print("Attention : Aucune commune du CSV n'a été trouvée dans le Shapefile. Vérifiez les codes INSEE.")
        return

    # 5. Fusionner les données de solde
    map_data = map_data.merge(stats, left_on='code_insee', right_index=True, how='left')
    map_data['solde'] = map_data['solde'].fillna(0)

    # 6. Création de la carte
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    
    # Échelle de couleur (on centre sur 0)
    vlimit = max(abs(map_data['solde'].min()), abs(map_data['solde'].max())) * 0.7
    
    map_data.plot(column='solde', 
                  ax=ax, 
                  cmap='RdYlGn', 
                  legend=True,
                  vmin=-vlimit, vmax=vlimit,
                  edgecolor='black',
                  linewidth=0.1,
                  legend_kwds={'label': "Solde (Attractivité Emploi)"})

    ax.set_title(f"Attractivité des Communes an 2022 par ({titre})", fontsize=15)
    ax.axis('off') 
    
    plt.savefig(os.path.join(DOSSIER_OUTPUT, f"carte_attractivite_{titre}.png"), dpi=300, bbox_inches='tight')
    plt.close()
def generate_mobility_map(df_flux):
    for mode in MODE_LABELS:
        mobility_map(mode, MODE_LABELS[mode], df_flux)
    mobility_map('TOTAL_FLUX', 'tous les modes', df_flux)


if __name__ == "__main__":
    try:
        df_annual = load_annual_data()
        df_analytic = load_analytic_data()
        
        generate_dashboard(df_annual, df_analytic)
        export_statistics(df_analytic)
        file_path = DOSSIER_DONNEES+ '/2022_data_with_duration.csv'
        df_flux = pd.read_csv(file_path, sep=',') 
        generate_mobility_map(df_flux)
        print("\nTERMINE !")
        
    except Exception as e:
        print(f"\nERREUR CRITIQUE : {e}")