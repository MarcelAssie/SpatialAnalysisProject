import pandas as pd
import numpy as np
import os
import traceback
import matplotlib.pyplot as plt
import logging
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
from typing import Dict
import warnings

warnings.filterwarnings('ignore')

# Configuration des styles
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


class GlobalAnalysis:
    """Classe pour l'analyse spatio-statistique des données de déplacement"""

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

        self.mode_names = {
            'MP': 'Marche à pied',
            'TC': 'Transports en commun',
            'VT': 'Voiture/Camion',
            'DRM': 'Deux-roues motorisé',
            'VL': 'Vélo'
        }

        self.mode_colors = {
            'MP': '#1f77b4',
            'TC': '#2ca02c',
            'VT': '#d62728',
            'DRM': '#ff7f0e',
            'VL': '#9467bd'
        }

    def _global_viz(self, sample_trajet: str = "92040 - 92062", sample_year: int = 2022):
        trans_cols = ['TRANS_MP', 'TRANS_TC', 'TRANS_VT', 'TRANS_DRM', 'TRANS_VL']
        colors = list(self.mode_colors.values())
        labels = list(self.mode_names.values())

        if sample_trajet in self.descriptive_data.index.get_level_values('TRAJET'):
            pass
        else:
            self.logger.warning(f"Le trajet '{sample_trajet}' n'existe pas dans les données")
            return

        # 1. Premier graphique (camembert simple)
        # Vérification que le trajet existe
        try:
            if (sample_trajet, sample_year) in self.descriptive_data.index:
                self.logger.info(
                    f"Tracé du camembert simple pour le trajet '{sample_trajet}' et l'année '{sample_year}'")
                sizes = self.descriptive_data.loc[(sample_trajet, sample_year), trans_cols].values.astype(float)

                fig, ax = plt.subplots(figsize=(10, 8))
                wedges, texts, autotexts = ax.pie(sizes, colors=colors, autopct='%1.1f%%',
                                                  startangle=90, pctdistance=0.85)
                ax.legend(wedges, labels, title="Modes de transport",
                          loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                ax.set_title(f"Répartition des modes de transport\nTrajet: {sample_trajet} - Année: {sample_year}",
                             fontsize=14, fontweight='bold')
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')

                plt.tight_layout()
                plt.savefig(f"{self.output_folder}/pie_chart.png", dpi=300, bbox_inches='tight')
                plt.show()
            else:
                self.logger.warning(f"L'année {sample_year} n'existe pas pour le trajet '{sample_trajet}'")

            # 2. Deuxième graphique (camembert concentrique) amélioré
            self.logger.info(f"Tracé du camembert concentrique pour le trajet '{sample_trajet}'")
            # Préparation des données
            valeurs_annuelles = []
            annees = list(range(2017, 2023))

            for annee in annees:
                if (sample_trajet, annee) in self.descriptive_data.index:
                    vals = self.descriptive_data.loc[(sample_trajet, annee), trans_cols].values.astype(float)
                    valeurs_annuelles.append(vals)
                else:
                    self.logger.warning(f"Données manquantes pour {sample_trajet} en {annee}")
                    valeurs_annuelles.append(np.zeros(len(trans_cols)))

            # Conversion en array numpy
            vals_array = np.array(valeurs_annuelles)

            # Création du graphique
            fig, ax = plt.subplots(figsize=(14, 10))
            size = 0.35

            # Couleurs améliorées
            tab20c = plt.cm.tab20c.colors
            outer_colors = [tab20c[i] for i in [0, 4, 8, 12, 16, 1, 5, 9, 13]]
            inner_colors = colors * len(annees)  # Répéter les couleurs pour chaque année

            # Anneau extérieur : répartition par année
            total_annuel = vals_array.sum(axis=1)
            wedges_outer, texts_outer, autotexts_outer = ax.pie(
                total_annuel,
                radius=1,
                colors=outer_colors[:len(annees)],
                wedgeprops=dict(width=size, edgecolor='white', linewidth=2),
                autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
                pctdistance=0.8,
                startangle=90
            )

            # Anneau intérieur : répartition par mode de transport pour chaque année
            vals_flat = vals_array.flatten()
            ax.pie(
                vals_flat,
                radius=1 - size,
                colors=inner_colors,
                wedgeprops=dict(width=size, edgecolor='white', linewidth=1),
                autopct='',
                startangle=90
            )

            # Légende et titres
            ax.legend(wedges_outer[:len(annees)], [f"Année {a}" for a in annees],
                      title="Années", loc="upper left", bbox_to_anchor=(1.05, 1))

            # Ajout d'une légende pour les modes de transport
            legend_elements = [Patch(facecolor=colors[i], label=labels[i])
                               for i in range(len(labels))]
            ax.legend(handles=legend_elements, title="Modes de transport",
                      loc="lower left", bbox_to_anchor=(1.05, 0))

            # Titre principal
            ax.set_title(f"Évolution de la répartition des modes de transport\nTrajet: {sample_trajet}",
                         fontsize=16, fontweight='bold', pad=20)

            plt.figtext(0.5, 0.01,
                        "Anneau extérieur: répartition annuelle | Anneau intérieur: modes de transport par année",
                        ha="center", fontsize=10, style='italic')

            # Ajustement de l'aspect
            ax.set(aspect="equal")

            plt.tight_layout()
            plt.savefig(f"{self.output_folder}/nested_pie_chart.png", dpi=300, bbox_inches='tight')
            plt.show()

            # 3. Graphique supplémentaire : évolution temporelle
            self.logger.info(f"Tracé de l'évolution temporelle pour le trajet '{sample_trajet}'")
            fig2, ax2 = plt.subplots(figsize=(12, 7))

            evolution_data = []
            for i, col in enumerate(trans_cols):
                mode_data = []
                for annee in annees:
                    if (sample_trajet, annee) in self.descriptive_data.index:
                        mode_data.append(self.descriptive_data.loc[(sample_trajet, annee), col])
                    else:
                        mode_data.append(0)
                evolution_data.append(mode_data)

            # Tracé des courbes d'évolution
            for i, (data, color, label) in enumerate(zip(evolution_data, colors, labels)):
                ax2.plot(annees, data, marker='o', color=color, linewidth=2.5,
                         markersize=8, label=label)

            ax2.set_xlabel("Année", fontsize=12, fontweight='bold')
            ax2.set_ylabel("Nombre de trajets", fontsize=12, fontweight='bold')
            ax2.set_title(f"Évolution des modes de transport\nTrajet: {sample_trajet}",
                          fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(title="Modes de transport", title_fontsize=12)
            ax2.set_xticks(annees)

            plt.tight_layout()
            plt.savefig(f"{self.output_folder}/evolution_chart.png", dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            self.logger.error(e)

    def descriptive_statistics(self) -> Dict:
        """
        Calcule les statistiques descriptives des variations

        Returns
        -------
        Dict
            Dictionnaire contenant toutes les statistiques
        """
        self.logger.info("=" * 60)
        self.logger.info("STATISTIQUES DESCRIPTIVES DES VARIATIONS")
        self.logger.info("=" * 60)

        stats_dict = {}

        # 1. Statistiques pour DELTA_DUREE
        delta_duree = self.analytic_data['DELTA_DUREE'].dropna()
        stats_dict['delta_duree'] = {
            'mean': delta_duree.mean(),
            'median': delta_duree.median(),
            'std': delta_duree.std(),
            'min': delta_duree.min(),
            'max': delta_duree.max(),
            'q1': delta_duree.quantile(0.25),
            'q3': delta_duree.quantile(0.75),
            'count': len(delta_duree)
        }

        self.logger.info(f"\nVariations de durée (DELTA_DUREE):")
        self.logger.info(f"  Moyenne: {stats_dict['delta_duree']['mean']:.2f} min")
        self.logger.info(f"  Médiane: {stats_dict['delta_duree']['median']:.2f} min")
        self.logger.info(f"  Écart-type: {stats_dict['delta_duree']['std']:.2f} min")
        self.logger.info(f"  Min/Max: {stats_dict['delta_duree']['min']:.2f} / {stats_dict['delta_duree']['max']:.2f} min")
        self.logger.info(f"  25%/75%: {stats_dict['delta_duree']['q1']:.2f} / {stats_dict['delta_duree']['q3']:.2f} min")

        # 2. Statistiques pour chaque variation de part modale
        self.logger.info("=" * 60)
        self.logger.info("VARIATIONS DES PARTS MODALES")
        self.logger.info("=" * 60)

        stats_dict['delta_parts'] = {}
        for col in self.delta_part_cols:
            mode = col.replace('DELTA_PART_', '')
            data = self.analytic_data[col].dropna()

            stats_dict['delta_parts'][mode] = {
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'q1': data.quantile(0.25),
                'q3': data.quantile(0.75),
                'count': len(data)
            }

            mode_name = self.mode_names.get(mode, mode)
            self.logger.info(f"\n{mode_name}:")
            self.logger.info(f"  Moyenne: {stats_dict['delta_parts'][mode]['mean']:.4f} (Δ part)")
            self.logger.info(f"  Médiane: {stats_dict['delta_parts'][mode]['median']:.4f} (Δ part)")
            self.logger.info(f"  Écart-type: {stats_dict['delta_parts'][mode]['std']:.4f}")
            self.logger.info(
                f"  Min/Max: {stats_dict['delta_parts'][mode]['min']:.4f} / {stats_dict['delta_parts'][mode]['max']:.4f}")

        return stats_dict

    def plot_distributions(self):
        """
        Visualise les distributions des variations
        """
        self.logger.info("=" * 60)
        self.logger.info("VISUALISATION DES DISTRIBUTIONS")
        self.logger.info("=" * 60)

        # Liste de toutes les variables à visualiser (durée + parts modales)
        all_vars = ['DELTA_DUREE'] + self.delta_part_cols
        n_vars = len(all_vars)


        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=['Variations de durée'] +
                           [
                               f"Δ Part {self.mode_names.get(var.replace('DELTA_PART_', ''), var.replace('DELTA_PART_', ''))}"
                               for var in self.delta_part_cols],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Ajouter chaque histogramme
        for idx, var in enumerate(all_vars):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1

            if var == 'DELTA_DUREE':
                color = 'lightblue'
                name = 'Δ Durée'
            else:
                mode = var.replace('DELTA_PART_', '')
                color = self.mode_colors.get(mode, 'gray')
                name = f'Δ {mode}'

            fig.add_trace(
                go.Histogram(
                    x=self.analytic_data[var].dropna(),
                    name=name,
                    marker_color=color,
                    opacity=0.7,
                    showlegend=False
                ),
                row=row, col=col
            )

            # Ajouter des titres d'axe
            if var == 'DELTA_DUREE':
                fig.update_xaxes(title_text="Δ Durée (min)", row=row, col=col)
            else:
                fig.update_xaxes(title_text="Δ Part", row=row, col=col)

            # Ajouter le nombre d'observations comme annotation
            n_obs = len(self.analytic_data[var].dropna())
            fig.add_annotation(
                xref=f"x{idx + 1}",
                yref=f"y{idx + 1}",
                x=0.95, y=0.95,
                xanchor='right',
                yanchor='top',
                text=f"N={n_obs:,}",
                showarrow=False,
                font=dict(size=10),
                row=row, col=col
            )

        fig.update_layout(
            height=300 * n_rows,  # Ajuster la hauteur selon le nombre de lignes
            showlegend=False,
            title_text="Distributions des variations annuelles",
            title_x=0.5
        )

        # Sauvegarder
        fig.write_html(f"{self.output_folder}/distributions_variations.html")
        self.logger.info("✓ Graphique des distributions sauvegardé")

        return fig
    def analyze_correlations(self) -> pd.DataFrame:
        """
        Analyse les corrélations entre variations de durée et variations de parts modales

        Returns
        -------
        pd.DataFrame
            DataFrame avec les coefficients de corrélation
        """
        self.logger.info("=" * 60)
        self.logger.info("ANALYSE DES CORRÉLATIONS")
        self.logger.info("=" * 60)

        # Calculer les corrélations
        corr_results = []

        for col in self.delta_part_cols:
            mode = col.replace('DELTA_PART_', '')

            # Filtrer les valeurs non nulles
            valid_data = self.analytic_data[['DELTA_DUREE', col]].dropna()

            if len(valid_data) > 10:  # Minimum d'observations
                corr, p_value = stats.pearsonr(valid_data['DELTA_DUREE'], valid_data[col])

                corr_results.append({
                    'Mode': mode,
                    'Nom_Mode': self.mode_names.get(mode, mode),
                    'Correlation': corr,
                    'P_Value': p_value,
                    'Significatif': p_value < 0.05,
                    'N_Observations': len(valid_data)
                })

                # Afficher les résultats
                signif = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                self.logger.info(f"{self.mode_names.get(mode, mode):<25}: r = {corr:7.4f} {signif} (p={p_value:.6f})")

        # Créer le DataFrame de résultats
        corr_df = pd.DataFrame(corr_results)

        # Visualiser les corrélations
        self._plot_correlation_matrix(corr_df)

        return corr_df

    def _plot_correlation_matrix(self, corr_df: pd.DataFrame):
        """Crée une visualisation de la matrice de corrélation"""

        n_modes = len(corr_df)

        # Calculer la disposition optimale
        n_cols = min(3, n_modes)
        n_rows = (n_modes + n_cols - 1) // n_cols

        # Tracer les nuages de points avec droites de régression
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[self.mode_names.get(row['Mode'], row['Mode'])
                            for _, row in corr_df.iterrows()],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        for idx, (_, row) in enumerate(corr_df.iterrows()):
            mode = row['Mode']
            col = f'DELTA_PART_{mode}'

            # Calculer la position dans la grille
            row_pos = (idx // n_cols) + 1
            col_pos = (idx % n_cols) + 1

            # Données pour ce mode
            plot_data = self.analytic_data[['DELTA_DUREE', col]].dropna()

            # Échantillonner si trop de données pour la performance
            if len(plot_data) > 1000:
                plot_data = plot_data.sample(1000, random_state=42)

            # Nuage de points
            fig.add_trace(
                go.Scatter(
                    x=plot_data['DELTA_DUREE'],
                    y=plot_data[col],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=self.mode_colors.get(mode, 'gray'),
                        opacity=0.3
                    ),
                    name=self.mode_names.get(mode, mode),
                    showlegend=False
                ),
                row=row_pos, col=col_pos
            )

            # Droite de régression si assez de points
            if len(plot_data) > 2:
                z = np.polyfit(plot_data['DELTA_DUREE'], plot_data[col], 1)
                p = np.poly1d(z)
                x_range = np.linspace(plot_data['DELTA_DUREE'].min(), plot_data['DELTA_DUREE'].max(), 100)

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=p(x_range),
                        mode='lines',
                        line=dict(color='black', width=2, dash='dash'),
                        showlegend=False
                    ),
                    row=row_pos, col=col_pos
                )

            # Mise en forme des axes
            fig.update_xaxes(title_text="Δ Durée (min)", row=row_pos, col=col_pos)
            fig.update_yaxes(title_text=f"Δ Part {mode}", row=row_pos, col=col_pos)

            # Ajouter le coefficient de corrélation et p-value
            fig.add_annotation(
                xref=f"x{idx + 1}",
                yref=f"y{idx + 1}",
                x=0.05, y=0.95,
                xanchor='left',
                yanchor='top',
                text=f"r = {row['Correlation']:.3f}<br>p = {row['P_Value']:.4f}",
                showarrow=False,
                font=dict(size=10, color='red'),
                row=row_pos, col=col_pos,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )

        fig.update_layout(
            height=400 * n_rows,
            title_text="Corrélations entre Δ Durée et Δ Parts modales",
            title_x=0.5
        )

        # Sauvegarder
        fig.write_html(f"{self.output_folder}/correlations_duree_parts.html")
        self.logger.info("✓ Graphique des corrélations sauvegardé")

    def analyze_spatial_patterns(self):
        """
        Analyse les patterns spatiaux des variations
        """
        self.logger.info("=" * 60)
        self.logger.info("ANALYSE DES PATTERNS SPATIAUX")
        self.logger.info("=" * 60)

        # Préparer les données spatiales
        if not self.ign_communes.empty:
            # Agrégation par commune d'origine
            self._analyze_by_commune('COM_ORG', 'origine')

            # Agrégation par commune de destination
            self._analyze_by_commune('COM_DEST', 'destination')
        else:
            self.logger.info("Données IGN non disponibles pour l'analyse spatiale détaillée")

    def _analyze_by_commune(self, commune_col: str, label: str):
        """
        Analyse les variations agrégées par commune

        Parameters
        ----------
        commune_col : str
            Colonne de la commune (COM_ORG ou COM_DEST)
        label : str
            Label pour les titres (origine/destination)
        """
        # Agrégation des variations par commune
        aggregated = self.analytic_data.reset_index()

        # Moyenne des variations par commune
        commune_stats = aggregated.groupby(commune_col).agg({
            'DELTA_DUREE': 'mean',
            **{col: 'mean' for col in self.delta_part_cols}
        }).reset_index()

        self.logger.info(f"\nVariations moyennes par commune d'{label}:")
        self.logger.info(f"  Nombre de communes: {len(commune_stats)}")
        self.logger.info(f"  Δ Durée moyen: {commune_stats['DELTA_DUREE'].mean():.2f} min")

        # Jointure avec les géométries
        commune_stats = commune_stats.merge(
            self.ign_communes[['INSEE_COM', 'geometry']],
            left_on=commune_col,
            right_on='INSEE_COM',
            how='left'
        )

        # Convertir en GeoDataFrame
        if not commune_stats.empty and 'geometry' in commune_stats.columns:
            gdf = gpd.GeoDataFrame(commune_stats, geometry='geometry', crs=self.ign_communes.crs)

            # Créer une carte pour chaque variable
            self._create_spatial_maps(gdf, label)
        else:
            self.logger.info(f"Impossible de créer les cartes pour les communes d'{label}")

    def _create_spatial_maps(self, gdf: gpd.GeoDataFrame, label: str):
        """
        Crée des cartes spatiales des variations

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            Données géospatiales agrégées
        label : str
            Type d'agrégation (origine/destination)
        """
        # Variables à cartographier
        variables = ['DELTA_DUREE'] + self.delta_part_cols[:3]  # Limité à 3 modes pour lisibilité

        # Créer une figure avec subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for idx, var in enumerate(variables):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # Déterminer le titre
            if var == 'DELTA_DUREE':
                title = f"Δ Durée moyenne (min)\npar commune d'{label}"
                cmap = 'RdBu_r'  # Rouge-Blue inversé pour avoir rouge=augmentation
            else:
                mode = var.replace('DELTA_PART_', '')
                title = f"Δ Part {self.mode_names.get(mode, mode)}\npar commune d'{label}"
                cmap = 'RdYlBu_r'  # Divergente

            # Tracer la carte
            gdf.plot(
                column=var,
                ax=ax,
                legend=True,
                legend_kwds={
                    'label': f"Valeur de {var.split('_')[-1]}",
                    'orientation': "horizontal",
                    'shrink': 0.5
                },
                cmap=cmap,
                missing_kwds={
                    'color': 'lightgrey',
                    'label': 'Données manquantes'
                },
                edgecolor='black',
                linewidth=0.1
            )

            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_axis_off()

        # Ajuster la disposition
        plt.tight_layout()
        plt.savefig(f"{self.output_folder}/carte_variations_{label}.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Cartes des variations par commune d'{label} sauvegardées")

    def analyze_by_distance_segments(self):
        """
        Analyse les variations par segments de distance
        """
        self.logger.info("=" * 60)
        self.logger.info("ANALYSE PAR SEGMENTS DE DISTANCE")
        self.logger.info("=" * 60)

        # Ajouter la distance aux données analytiques
        analytic_with_dist = self.analytic_data.reset_index().merge(
            self.descriptive_data.reset_index()[['TRAJET', 'ANNEE', 'DISTANCE']],
            on=['TRAJET', 'ANNEE'],
            how='left'
        )

        # Créer des segments de distance
        analytic_with_dist['SEGMENT_DISTANCE'] = pd.cut(
            analytic_with_dist['DISTANCE'],
            bins=[0, 5, 10, 20, 50, 100, float('inf')],
            labels=['<5km', '5-10km', '10-20km', '20-50km', '50-100km', '>100km']
        )

        # Statistiques par segment
        segment_stats = analytic_with_dist.groupby('SEGMENT_DISTANCE').agg({
            'DELTA_DUREE': ['mean', 'std', 'count'],
            **{col: 'mean' for col in self.delta_part_cols}
        })

        # Aplatir les colonnes multi-index
        segment_stats.columns = ['_'.join(col).strip() for col in segment_stats.columns.values]
        segment_stats = segment_stats.reset_index()

        self.logger.info("\nVariations moyennes par segment de distance:")
        self.logger.info(segment_stats[['SEGMENT_DISTANCE', 'DELTA_DUREE_mean',
                             'DELTA_PART_TC_mean', 'DELTA_PART_VT_mean']].to_string())

        # Visualiser
        self._plot_distance_segments(segment_stats)

        return segment_stats

    def _plot_distance_segments(self, segment_stats: pd.DataFrame):
        """Visualise les variations par segment de distance"""

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[
                "Δ Durée moyenne par segment de distance",
                "Δ Parts modales par segment de distance"
            ],
            vertical_spacing=0.15
        )

        # Graphique 1: Δ Durée
        fig.add_trace(
            go.Bar(
                x=segment_stats['SEGMENT_DISTANCE'],
                y=segment_stats['DELTA_DUREE_mean'],
                error_y=dict(
                    type='data',
                    array=segment_stats['DELTA_DUREE_std'],
                    visible=True
                ),
                marker_color='lightblue',
                name='Δ Durée (min)',
                text=[f"{v:.2f}" for v in segment_stats['DELTA_DUREE_mean']],
                textposition='auto'
            ),
            row=1, col=1
        )

        # Graphique 2: Δ Parts modales
        for mode in ['TC', 'VT', 'VL']:  # Sélection de modes
            col = f'DELTA_PART_{mode}_mean'
            if col in segment_stats.columns:
                fig.add_trace(
                    go.Scatter(
                        x=segment_stats['SEGMENT_DISTANCE'],
                        y=segment_stats[col],
                        mode='lines+markers',
                        name=self.mode_names.get(mode, mode),
                        line=dict(color=self.mode_colors.get(mode, 'gray'), width=3),
                        marker=dict(size=10)
                    ),
                    row=2, col=1
                )

        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Analyse par segments de distance",
            title_x=0.5
        )

        fig.update_xaxes(title_text="Segment de distance", row=1, col=1)
        fig.update_xaxes(title_text="Segment de distance", row=2, col=1)
        fig.update_yaxes(title_text="Δ Durée moyenne (min)", row=1, col=1)
        fig.update_yaxes(title_text="Δ Part modale", row=2, col=1)

        # Sauvegarder
        fig.write_html(f"{self.output_folder}/analyse_segments_distance.html")
        self.logger.info("✓ Analyse par segments de distance sauvegardée")

    def generate_summary_report(self):
        """
        Génère un rapport complet d'analyse
        """
        self.logger.info("=" * 60)
        self.logger.info("GÉNÉRATION DU RAPPORT D'ANALYSE")
        self.logger.info("=" * 60)

        # Exécuter toutes les analyses
        stats_dict = self.descriptive_statistics()
        corr_df = self.analyze_correlations()
        self.plot_distributions()
        self.analyze_spatial_patterns()
        segment_stats = self.analyze_by_distance_segments()

        # Créer un rapport textuel
        report_lines = [
            "=" * 60,
            "RAPPORT D'ANALYSE SPATIO-STATISTIQUE",
            "=" * 60,
            f"\nDate: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Nombre d'observations: {len(self.analytic_data)}",
            f"Période: {self.analytic_data.reset_index()['ANNEE'].min()}-{self.analytic_data.reset_index()['ANNEE'].max()}",

            "\n" + "=" * 60,
            "1. RÉSUMÉ DES VARIATIONS",
            "=" * 60,
        ]

        # Résumé Δ Durée
        duree_stats = stats_dict['delta_duree']
        report_lines.append(f"\nVariations de durée (Δ_DUREE):")
        report_lines.append(f"  • Moyenne: {duree_stats['mean']:.2f} min")
        report_lines.append(f"  • Écart-type: {duree_stats['std']:.2f} min")
        report_lines.append(f"  • Intervalle: [{duree_stats['min']:.2f}, {duree_stats['max']:.2f}] min")

        # Résumé Δ Parts modales
        report_lines.append("\nVariations des parts modales:")
        for mode, stats in stats_dict['delta_parts'].items():
            mode_name = self.mode_names.get(mode, mode)
            trend = "↗" if stats['mean'] > 0 else "↘"
            report_lines.append(f"  • {mode_name:<20}: {stats['mean']:7.4f} {trend} (moyenne)")

        # Corrélations significatives
        report_lines.append("\n" + "=" * 60)
        report_lines.append("2. CORRÉLATIONS SIGNIFICATIVES")
        report_lines.append("=" * 60)

        sig_corr = corr_df[corr_df['Significatif']]
        if len(sig_corr) > 0:
            for _, row in sig_corr.iterrows():
                direction = "positive" if row['Correlation'] > 0 else "negative"
                report_lines.append(
                    f"  • {row['Nom_Mode']:<20}: r = {row['Correlation']:.3f} ({direction})"
                )
        else:
            report_lines.append("  Aucune corrélation significative au seuil de 5%")

        # Insights par distance
        report_lines.append("\n" + "=" * 60)
        report_lines.append("3. INSIGHTS PAR DISTANCE")
        report_lines.append("=" * 60)

        if segment_stats is not None:
            for _, row in segment_stats.iterrows():
                report_lines.append(f"\n  Segment {row['SEGMENT_DISTANCE']}:")
                report_lines.append(f"    • Δ Durée: {row['DELTA_DUREE_mean']:.2f} min")
                if 'DELTA_PART_TC_mean' in row:
                    report_lines.append(f"    • Δ TC: {row['DELTA_PART_TC_mean']:.4f}")
                if 'DELTA_PART_VT_mean' in row:
                    report_lines.append(f"    • Δ Voiture: {row['DELTA_PART_VT_mean']:.4f}")

        # Recommandations
        report_lines.append("\n" + "=" * 60)
        report_lines.append("4. RECOMMANDATIONS POUR L'ANALYSE SUIVANTE")
        report_lines.append("=" * 60)
        report_lines.append("""
  1. Vérifier l'autocorrélation spatiale des résidus
  2. Intégrer les variables socio-économiques comme contrôles
  3. Tester des modèles à effets fixes par trajet
  4. Analyser les différences entre trajets intra/inter-départementaux
  5. Examiner les outliers (trajets avec variations extrêmes)
        """)

        report_lines.append("\n" + "=" * 60)
        report_lines.append("FIN DU RAPPORT")
        report_lines.append("=" * 60)

        # Sauvegarder le rapport
        report_text = "\n".join(report_lines)

        with open(f"{self.output_folder}/rapport_analyse.txt", 'w', encoding='utf-8') as f:
            f.write(report_text)

        self.logger.info("✓ Rapport d'analyse généré et sauvegardé")
        self.logger.info("\n" + report_text)

        return report_text

    def run_full_analysis(self):
        """
        Exécute l'analyse complète
        """
        self.logger.info("=" * 60)
        self.logger.info("DÉMARRAGE DE L'ANALYSE SPATIO-STATISTIQUE COMPLÈTE")
        self.logger.info("=" * 60)

        try:
            # 1. Génération des diagrammes
            self._global_viz()

            # 2. Statistiques descriptives
            self.descriptive_statistics()

            # 3. Analyse des distributions
            self.plot_distributions()

            # 4. Analyse des corrélations
            self.analyze_correlations()

            # 5. Analyse spatiale
            self.analyze_spatial_patterns()

            # 6. Analyse par distance
            self.analyze_by_distance_segments()

            # 7. Rapport final
            self.generate_summary_report()

            self.logger.info("=" * 60)
            self.logger.info("ANALYSE TERMINÉE AVEC SUCCÈS")
            self.logger.info("=" * 60)
            self.logger.info(f"Tous les résultats ont été sauvegardés dans: {self.output_folder}")

        except Exception as e:
            self.logger.error(e)
            traceback.print_exc()

