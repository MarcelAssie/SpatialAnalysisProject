import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import geopandas as gpd
from typing import Dict
import logging

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

class GlobalAnalysis:
    def __init__(self, analytic_data: pd.DataFrame,
                 descriptive_data: pd.DataFrame,
                 ign_communes: gpd.GeoDataFrame,
                 logger: logging.Logger,
                 output_folder: str = "output/analysis"):

        self.analytic_data = analytic_data
        self.descriptive_data = descriptive_data
        self.ign_communes = ign_communes
        self.output_folder = output_folder
        self.logger = logger

        os.makedirs(output_folder, exist_ok=True)

        self.delta_part_cols = [c for c in analytic_data.columns if c.startswith('DELTA_PART_')]
        self.trans_cols = ['TRANS_MP', 'TRANS_TC', 'TRANS_VT', 'TRANS_DRM', 'TRANS_VL']

        self.mode_names = {
            'MP': 'Marche', 'TC': 'Transports Commun', 'VT': 'Voiture',
            'DRM': '2 Roues Mot.', 'VL': 'Vélo'
        }
        self.mode_colors = {
            'MP': '#3498db', 'TC': '#2ecc71', 'VT': '#e74c3c',
            'DRM': '#f39c12', 'VL': '#9b59b6'
        }

    def plot_global_trends(self):
        self.logger.info("Génération de la tendance globale annuelle...")

        yearly_sum = self.descriptive_data.groupby('ANNEE')[self.trans_cols].sum()
        yearly_share = yearly_sum.div(yearly_sum.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(12, 7))

        for col in self.trans_cols:
            mode_code = col.replace('TRANS_', '')
            ax.plot(yearly_share.index, yearly_share[col],
                    marker='o', linewidth=3,
                    label=self.mode_names.get(mode_code),
                    color=self.mode_colors.get(mode_code))

        ax.set_title("Évolution globale des parts de marché", fontsize=16, fontweight='bold')
        ax.set_ylabel("Part de marché (%)")
        ax.set_xlabel("Année")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"{self.output_folder}/1_tendance_globale.png", dpi=300)
        plt.close()

    def descriptive_statistics(self) -> Dict:
        self.logger.info("Calcul des statistiques descriptives...")
        stats_dict = {}

        stats_dict['delta_duree'] = self.analytic_data['DELTA_DUREE'].describe().to_dict()

        stats_dict['delta_parts'] = {}
        for col in self.delta_part_cols:
            mode = col.replace('DELTA_PART_', '')
            stats_dict['delta_parts'][mode] = self.analytic_data[col].describe().to_dict()

        with open(f"{self.output_folder}/2_statistiques_descriptives.txt", "w") as f:
            f.write("=== STATISTIQUES DESCRIPTIVES ===\n\n")
            f.write(f"VARIATION DUREE :\n{pd.Series(stats_dict['delta_duree']).to_string()}\n\n")
            for m, s in stats_dict['delta_parts'].items():
                f.write(f"VARIATION {m}:\n{pd.Series(s).to_string()}\n\n")

        return stats_dict

    def analyze_correlations(self):
        self.logger.info("Génération des régressions et statistiques...")

        target_modes = [col.replace('DELTA_PART_', '') for col in self.delta_part_cols]

        def add_stats_box(ax, x_data, y_data, color_code):
            slope, intercept, r_value, p_value, _ = stats.linregress(x_data, y_data)
            r_squared = r_value ** 2

            txt = f"y = {slope:.3f}x + {intercept:.3f}\n"
            txt += f"$R^2$ = {r_squared:.3f}\n"
            txt += f"Pearson r = {r_value:.2f}\n"
            txt += "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"

            edge_color = color_code if p_value < 0.05 else 'gray'
            ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor=edge_color, alpha=0.9))

        n_modes = len(target_modes)
        fig1, axes1 = plt.subplots(1, n_modes, figsize=(5 * n_modes, 5), sharey=True)
        fig1.suptitle("Impact de la Variation de la DURÉE", fontsize=16, fontweight='bold')

        for idx, mode in enumerate(target_modes):
            col_y = f"DELTA_PART_{mode}"
            data_clean = self.analytic_data[['DELTA_DUREE', col_y]].dropna()
            ax = axes1[idx] if n_modes > 1 else axes1

            if len(data_clean) > 2:
                sns.regplot(x='DELTA_DUREE', y=col_y, data=data_clean, ax=ax,
                            scatter_kws={'alpha': 0.3, 's': 20, 'color': 'gray'},
                            line_kws={'color': '#e74c3c', 'linewidth': 2})
                add_stats_box(ax, data_clean['DELTA_DUREE'], data_clean[col_y], '#e74c3c')
                ax.set_title(self.mode_names.get(mode, mode), fontsize=12, fontweight='bold')
                ax.set_xlabel("Variation Durée (min)")
                ax.set_ylabel("Variation Part Modale (%)" if idx == 0 else "")
                ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{self.output_folder}/3_regression_duree_stats.png", dpi=300)
        plt.close()

        modes_vs_tc = [m for m in target_modes if m != 'TC']
        n_modes_tc = len(modes_vs_tc)
        fig2, axes2 = plt.subplots(1, n_modes_tc, figsize=(5 * n_modes_tc, 5), sharey=True)
        fig2.suptitle("Compétition face aux Transports en Commun (TC)", fontsize=16, fontweight='bold')

        for idx, mode in enumerate(modes_vs_tc):
            col_x, col_y = "DELTA_PART_TC", f"DELTA_PART_{mode}"
            data_clean = self.analytic_data[[col_x, col_y]].dropna()
            ax = axes2[idx] if n_modes_tc > 1 else axes2

            if len(data_clean) > 2:
                sns.regplot(x=col_x, y=col_y, data=data_clean, ax=ax,
                            scatter_kws={'alpha': 0.3, 's': 20, 'color': 'gray'},
                            line_kws={'color': '#3498db', 'linewidth': 2})
                add_stats_box(ax, data_clean[col_x], data_clean[col_y], '#3498db')
                ax.set_title(f"TC vs {self.mode_names.get(mode, mode)}", fontsize=12, fontweight='bold')
                ax.set_xlabel("Variation Part TC (%)")
                ax.set_ylabel(f"Variation Part {mode} (%)" if idx == 0 else "")
                ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{self.output_folder}/3_regression_competition_tc_stats.png", dpi=300)
        plt.close()

    def analyze_distance_impact(self):
        self.logger.info("Analyse Distance vs Parts Modales...")

        last_year = self.analytic_data.index.get_level_values('ANNEE').max()
        df_latest = self.analytic_data[self.analytic_data.index.get_level_values('ANNEE') == last_year].copy()

        modes = ['PART_MP', 'PART_VL', 'PART_TC', 'PART_VT']
        titles = ['Marche', 'Vélo', 'Transport Commun', 'Voiture']
        colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c']

        fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
        fig.suptitle(f"Structure du choix modal selon la Distance ({last_year})", fontsize=20, fontweight='bold')

        for idx, (col, title, color) in enumerate(zip(modes, titles, colors)):
            ax = axes[idx]
            sns.regplot(x='DISTANCE', y=col, data=df_latest, ax=ax,
                        scatter_kws={'alpha': 0.1, 's': 10, 'color': 'gray'},
                        line_kws={'color': color, 'linewidth': 3},
                        order=2, ci=95)

            ax.set_title(title, fontsize=16, fontweight='bold', color=color)
            ax.set_xlabel("Distance (km)")
            ax.set_ylabel("Part de Marché (%)" if idx == 0 else "")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle='--', alpha=0.5)

            avg_dist = (df_latest['DISTANCE'] * df_latest[col]).sum() / df_latest[col].sum()
            ax.text(0.5, 0.9, f"Dist. Moyenne: {avg_dist:.1f} km",
                    transform=ax.transAxes, ha='center',
                    bbox=dict(facecolor='white', edgecolor=color, boxstyle='round'))

        plt.tight_layout()
        plt.savefig(f"{self.output_folder}/5_structure_distance_mode.png", dpi=300)
        plt.close()

    def analyze_spatial_patterns(self):
        self.logger.info("Génération des cartes de variations spatiales...")

        if self.ign_communes.empty:
            self.logger.warning("Fond de carte IGN manquant. Annulation.")
            return

        vars_to_map = ['DELTA_DUREE', 'DELTA_PART_VT', 'DELTA_PART_TC']
        titles = {
            'DELTA_DUREE': 'Variation Durée (min)',
            'DELTA_PART_VT': 'Variation part Voiture',
            'DELTA_PART_TC': 'Variation part TC'
        }

        for scale_col, scale_name in [('COM_ORG', 'Origine'), ('COM_DEST', 'Destination')]:
            agg_stats = self.analytic_data.reset_index().groupby(scale_col)[vars_to_map].mean().reset_index()
            gdf = self.ign_communes.merge(agg_stats, left_on='INSEE_COM', right_on=scale_col, how='inner')

            if gdf.empty: continue

            fig, axes = plt.subplots(1, 3, figsize=(20, 8))
            fig.suptitle(f"Variations moyennes par Commune ({scale_name})", fontsize=20, fontweight='bold')

            for idx, var in enumerate(vars_to_map):
                ax = axes[idx]
                abs_max = max(abs(gdf[var].min()), abs(gdf[var].max()))

                gdf.plot(column=var, ax=ax, cmap='RdBu_r', legend=True,
                         vmin=-abs_max, vmax=abs_max,
                         legend_kwds={'orientation': "horizontal", 'shrink': 0.6})
                ax.set_title(titles[var], fontsize=14)
                ax.set_axis_off()

            plt.tight_layout()
            plt.savefig(f"{self.output_folder}/4_cartes_variations_{scale_name.lower()}.png", dpi=300)
            plt.close()

    def run_full_analysis(self):
        self.logger.info("Démarrage du pipeline d'analyse...")
        try:
            self.plot_global_trends()
            self.descriptive_statistics()
            self.analyze_correlations()
            self.analyze_spatial_patterns()
            self.analyze_distance_impact()
            self.logger.info(f"Analyse terminée : {self.output_folder}")
        except Exception as e:
            self.logger.error(f"Erreur d'analyse : {e}")
            import traceback
            traceback.print_exc()