import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import geopandas as gpd
import logging

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

class GlobalAnalysis:
    def __init__(self, analytic_data: pd.DataFrame,
                 descriptive_data: pd.DataFrame,
                 ign_communes: gpd.GeoDataFrame,
                 logger: logging.Logger,
                 output_folder: str = "output/analysis",
                 sample_size: int = 500):

        self.analytic_data = analytic_data
        self.descriptive_data = descriptive_data
        self.ign_communes = ign_communes
        self.output_folder = output_folder
        self.logger = logger
        self.sample_size = sample_size

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

    def _subsample_data(self, df):
        """Sous-échantillonne les données pour améliorer la lisibilité des graphiques"""
        if len(df) <= self.sample_size:
            return df
        return df.sample(n=self.sample_size, random_state=42)

    def plot_global_trends(self):
        self.logger.info("Génération de la tendance globale annuelle...")

        yearly_sum = self.descriptive_data.groupby('ANNEE')[self.trans_cols].sum()
        yearly_share = yearly_sum.div(yearly_sum.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(12, 7))

        for col in self.trans_cols:
            mode_code = col.replace('TRANS_', '')
            ax.plot(yearly_share.index, yearly_share[col],
                    label=self.mode_names.get(mode_code, mode_code),
                    color=self.mode_colors.get(mode_code, 'gray'),
                    linewidth=3, marker='o', markersize=6)

        ax.set_title("Évolution des Parts Modales Globales", fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel("Année", fontsize=14)
        ax.set_ylabel("Part de Marché (%)", fontsize=14)
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(f"{self.output_folder}/1_tendencies_globales.png", dpi=300, bbox_inches='tight')
        plt.close()

    def descriptive_statistics(self):
        self.logger.info("Calcul des statistiques descriptives...")

        stats_summary = {}

        for col in self.trans_cols:
            mode_code = col.replace('TRANS_', '')
            data = self.descriptive_data[col]

            stats_summary[f'{mode_code}_mean'] = data.mean()
            stats_summary[f'{mode_code}_std'] = data.std()
            stats_summary[f'{mode_code}_min'] = data.min()
            stats_summary[f'{mode_code}_max'] = data.max()
            stats_summary[f'{mode_code}_median'] = data.median()

        summary_df = pd.DataFrame(stats_summary, index=[0])

        summary_df.to_csv(f"{self.output_folder}/2_statistiques_descriptives.csv", index=False)
        self.logger.info("Statistiques descriptives sauvegardées")

    def analyze_correlations(self):
        self.logger.info("Génération des régressions et statistiques...")

        def add_stats_box(ax, x_data, y_data, color_code):
            if len(x_data) < 2 or len(y_data) < 2:
                return
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

        target_modes = ['MP', 'VL', 'TC', 'VT']
        n_modes = len(target_modes)
        fig1, axes1 = plt.subplots(1, n_modes, figsize=(5 * n_modes, 5), sharey=True)
        fig1.suptitle("Impact de la Variation de la DURÉE", fontsize=16, fontweight='bold')

        for idx, mode in enumerate(target_modes):
            col_y = f"DELTA_PART_{mode}"
            data_clean = self.analytic_data[['DELTA_DUREE', col_y]].dropna()
            ax = axes1[idx] if n_modes > 1 else axes1

            if len(data_clean) > 2:
                # Sous-échantillonnage pour améliorer la lisibilité
                data_plot = self._subsample_data(data_clean)
                
                sns.regplot(x='DELTA_DUREE', y=col_y, data=data_plot, ax=ax,
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
                # Sous-échantillonnage pour améliorer la lisibilité
                data_plot = self._subsample_data(data_clean)
                
                sns.regplot(x=col_x, y=col_y, data=data_plot, ax=ax,
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

            # Sous-échantillonnage pour améliorer la lisibilité
            data_plot = self._subsample_data(df_latest)

            sns.regplot(x='DISTANCE', y=col, data=data_plot, ax=ax,
                        scatter_kws={'alpha': 0.1, 's': 10, 'color': 'gray'},
                        line_kws={'color': color, 'linewidth': 3},
                        logistic=True,
                        ci=None)  #

            ax.set_title(title, fontsize=16, fontweight='bold', color=color)
            ax.set_xlabel("Distance (km)")
            ax.set_ylabel("Part de Marché (%)" if idx == 0 else "")

            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle='--', alpha=0.5)

            sum_col = df_latest[col].sum()
            if sum_col > 0:
                avg_dist = (df_latest['DISTANCE'] * df_latest[col]).sum() / sum_col
                ax.text(0.5, 0.9, f"Dist. Moyenne: {avg_dist:.1f} km",
                        transform=ax.transAxes, ha='center',
                        bbox=dict(facecolor='white', edgecolor=color, boxstyle='round', alpha=0.8))
        plt.tight_layout()
        plt.savefig(f"{self.output_folder}/5_structure_distance_mode.png", dpi=300)
        plt.close()

    def run_full_analysis(self):
        self.logger.info("DÉMARRAGE DU PIPELINE D'ANALYSE...")
        try:
            self.plot_global_trends()
            self.descriptive_statistics()
            self.analyze_correlations()
            self.analyze_distance_impact()
            self.logger.info(f"Analyse terminée : {Path(self.output_folder)}")
        except Exception as e:
            self.logger.error(f"Erreur d'analyse : {e}")
            import traceback
            traceback.print_exc()
