import pandas as pd
import geopandas as gpd
from pyproj import Geod
import numpy as np
import os
import logging

from utils import GlobalAnalysis




class SpatialStatisticalAnalysis:
    def __init__(self, input_folder:str, output_folder:str, base_cols:list[str]):
        self.logger = self._setup_logger()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.base_cols = base_cols
        self.ign_communes = gpd.GeoDataFrame()
        self.insee_communes  = None
        self.ign_com_geoms = None
        self.data_2017 = pd.DataFrame()
        self.dfs = {}
        self.merged_data = pd.DataFrame()
        self.descriptive_data = pd.DataFrame()
        self.analytic_data = pd.DataFrame()



    @staticmethod
    def _setup_logger():
        logger = logging.getLogger("Spatial Analysis")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            line_format = "%(asctime)s | %(levelname)s | %(module)s - %(lineno)d (%(funcName)s) | %(message)s"
            formatter = logging.Formatter(fmt=line_format, datefmt='%y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger



    def load_data(self, code_geo_reg:str ='INSEE_REG', code_geo_com:str ='INSEE_COM', delimiter:str=';'):
        self.logger.info(f"Chargement des données")
        file_paths = [os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder)]
        ign_communes = None
        if file_paths:
            csv_files = [f for f in file_paths if os.path.splitext(f)[1].lower() == '.csv']
            vec_files = [f for f in file_paths if os.path.splitext(f)[1].lower() in ['.shp', '.gpkg']]
            self.logger.info(f"{len(csv_files) + len(vec_files)} données trouvées dont {len(csv_files)} "
                             f"CSV et {len(vec_files)} Vecteurs")
            for f_path in file_paths:
                _, file_ext = os.path.splitext(f_path)
                try:
                    if file_ext.lower() == '.csv' and not "coordinates" in f_path:
                        df = pd.read_csv(f_path, delimiter=delimiter)
                        parts = os.path.basename(f_path).split("_")
                        if len(parts) > 2 and parts[2].isdigit():
                            year = int(parts[2])
                            self.dfs[year] = df
                            if year == 2022:
                                self.insee_communes = df.copy()
                    else:
                        if file_ext.lower() == '.shp':
                            ign_communes = gpd.read_file(f_path)
                        elif file_ext.lower() == '.gpkg':
                            if "COMMUNE_COMMUNS" in f_path:
                                self.ign_com_geoms = gpd.read_file(f_path)
                        else:
                            continue
                except Exception as e:
                    self.logger.error(f"Erreur lors du chargement de {f_path}: {e}")

            if not self.dfs[2017].empty:
                self.data_2017 = self.dfs[2017].copy()

            if ign_communes is not None:
                self.ign_communes = ign_communes
                if code_geo_reg in self.ign_communes.columns:
                    self.ign_communes = self.ign_communes[self.ign_communes[code_geo_reg].astype(str) == "11"]
                if code_geo_com in self.ign_communes.columns:
                    self.ign_communes[code_geo_com] = self.ign_communes[code_geo_com].astype(str)
        else:
            self.logger.warning(f"Aucunes données trouvées dans le dossier -> {self.input_folder}")
            raise ValueError("Aucunes données trouvées")

    def _transform_data(self, data):
        self.logger.info("Transformation des données récupérées")
        
        mapping = {
            "COMMUNE": "COM_ORG", 
            "DCLT": "COM_DEST", 
            "Deux-roues motorisé": "TRANS_DRM",
            "Marche à pied": "TRANS_MP", 
            "Pas de transport (travail domicile)": "TRANS_M",
            "Transports en commun": "TRANS_TC", 
            "Voiture, camion, fourgonnette": "TRANS_VT",
            "Vélo": "TRANS_VL"
        }
        
        data.rename(columns=mapping, inplace=True)

        expected_trans = ["TRANS_DRM", "TRANS_MP", "TRANS_M", "TRANS_TC", "TRANS_VT", "TRANS_VL"]
        for col in expected_trans:
            if col not in data.columns:
                self.logger.warning(f"Colonne {col} manquante, création avec des zéros")
                data[col] = 0.0

        if "TOTAL_FLUX" not in data.columns:
            data["TOTAL_FLUX"] = data[expected_trans].sum(axis=1)

        data['COM_ORG'] = data['COM_ORG'].astype(str)
        data['COM_DEST'] = data['COM_DEST'].astype(str)
        data['COM_ORG'] = data['COM_ORG'].str.zfill(5)
        data['COM_DEST'] = data['COM_DEST'].str.zfill(5)
        return data

    def join_communes_geoms(self, data):
        self.logger.info("Ajout des centroids origine / destination et filtrage des paires valides")

        communes_geoms = self.ign_com_geoms.copy()

        valid_geoms_mask = (communes_geoms['geometry'].notna()) & (communes_geoms['geom_dclt'].notna())
        communes_clean = communes_geoms[valid_geoms_mask].copy()

        self.logger.info(f"Paires de communes avec géométries valides : {len(communes_clean)} / {len(communes_geoms)}")

        if communes_clean.crs is None:
            communes_clean = communes_clean.set_crs(epsg=2154)
        else:
            communes_clean = communes_clean.to_crs(epsg=2154)

        # Centroids
        communes_clean["centroid_org_geom"] = communes_clean.geometry.centroid
        gs_dest = gpd.GeoSeries.from_wkt(communes_clean["geom_dclt"])
        gs_dest = gs_dest.set_crs(communes_clean.crs)
        communes_clean["centroid_dest_geom"] = gs_dest.centroid

        # Coordonnées WGS84
        gdf_org_pts = gpd.GeoDataFrame(geometry=communes_clean["centroid_org_geom"], crs=2154).to_crs(epsg=4326)
        communes_clean["lon_org"] = gdf_org_pts.geometry.x
        communes_clean["lat_org"] = gdf_org_pts.geometry.y

        gdf_dest_pts = gpd.GeoDataFrame(geometry=communes_clean["centroid_dest_geom"], crs=2154).to_crs(epsg=4326)
        communes_clean["lon_dest"] = gdf_dest_pts.geometry.x
        communes_clean["lat_dest"] = gdf_dest_pts.geometry.y

        ref_geo = communes_clean[["COMMUNE", "DCLT", "lon_org", "lat_org", "lon_dest", "lat_dest"]]

        data_merged = data.merge(
            ref_geo,
            left_on=["COM_ORG", "COM_DEST"],
            right_on=["COMMUNE", "DCLT"],
            how="inner"
        )

        data_merged.drop(columns=["COMMUNE", "DCLT"], inplace=True)

        # Calcul de la distance
        g = Geod(ellps='WGS84')
        _, _, dist_meters = g.inv(
            data_merged['lon_org'].values,
            data_merged['lat_org'].values,
            data_merged['lon_dest'].values,
            data_merged['lat_dest'].values
        )
        data_merged['DISTANCE'] = dist_meters / 1000

        self.logger.info(f"Données finales après jointure géométrique : {len(data_merged)} lignes")

        return data_merged

    def compute_a_year_duration(self):
        self.logger.info("Ajout de la 'durée' pour l'année 2017")
        try:
            self.data_2017 = self.data_2017[self.base_cols].copy()
            vitesse_tc = 25
            duree_base = self.data_2017['DISTANCE'] / vitesse_tc
            variation = np.random.normal(0, 0.05, len(self.data_2017))
            self.data_2017['DUREE'] = duree_base * 60 * (1 + variation)
            self._save_data(self.data_2017, "2017_data_with_duration")
        except Exception as e:
            self.logger.error(f"Erreur dans compute_a_year_duration: {e}")

    def generate_all_years_duration(self):
        self.logger.info("Calculs préliminaires sur l'année 2017")
        self.data_2017 = self._transform_data(self.data_2017)
        self.data_2017 = self.join_communes_geoms(self.data_2017)
        # raise
        self.compute_a_year_duration()
        self.dfs[2017] = self.data_2017
        self.logger.info("Calculs sur les années 2018 à 2022")
        for year in range(2018, 2023):
            self.logger.info(f"Année {year}")
            df_prev = self.dfs[year - 1].copy()

            # Garder les colonnes de base
            df = self.dfs[year].copy()
            df = self._transform_data(df)
            df = self.join_communes_geoms(df)
            df = df[self.base_cols].copy()

            # Calculer la durée pour l'année en cours
            variation_duree = np.random.normal(0, 0.05, len(df))

            # Récupérer la durée de l'année précédente
            prev_duree = df_prev['DUREE'].copy() if year == 2017 else df_prev['DUREE'].copy()
            df['DUREE'] = prev_duree * (1 + variation_duree)
            # df['DUREE'] = df_prev.loc[df.index, 'DUREE'] * (1 + variation_duree)

            # Calculer delta
            # delta = (df['DUREE'] - prev_duree) / prev_duree

            # # Appliquer les sensibilités
            # sens_mp = -0.4
            # sens_tc = -0.6
            # sens_vt = 0.4
            # sens_drm = 0.15
            # sens_vl = 0.05
            #
            # df['TRANS_MP'] = df_prev['TRANS_MP'] * (1 + sens_mp * delta)
            # df['TRANS_TC'] = df_prev['TRANS_TC'] * (1 + sens_tc * delta)
            # df['TRANS_VT'] = df_prev['TRANS_VT'] * (1 + sens_vt * delta)
            # df['TRANS_DRM'] = df_prev['TRANS_DRM'] * (1 + sens_drm * delta)
            # df['TRANS_VL'] = df_prev['TRANS_VL'] * (1 + sens_vl * delta)
            #
            # # Clip et normalisation
            # trans_cols = ['TRANS_MP', 'TRANS_TC', 'TRANS_VT', 'TRANS_DRM', 'TRANS_VL']
            # # Clip et normalisation
            # for col in trans_cols:
            #     df[col] = df[col].clip(lower=0)
            #
            # sum_modes = df[trans_cols].sum(axis=1)
            # factor = df['TOTAL_FLUX'] / sum_modes
            #
            # for col in trans_cols:
            #     df[col] = df[col] * factor

            self.dfs[year] = df
            self._save_data(df, f"{year}_data_with_duration")
        return self.dfs

    def _get_descriptive_data(self):
        dfs = self.generate_all_years_duration()
        list_dfs = []
        if dfs:
            for year, df in dfs.items():
                temp = df.copy()
                temp['ANNEE'] = year
                temp['TRAJET'] = temp['COM_ORG'] + " - " + temp['COM_DEST']

                list_dfs.append(temp)

            # Combiner tout
            final_df = pd.concat(list_dfs)

            # Mettre le trajet et l'année en index
            final_df = final_df.set_index(['TRAJET', 'ANNEE'])
            # final_df = final_df.drop("DISTANCE", axis='columns')
            final_df = final_df.drop(final_df[final_df['COM_ORG'] == final_df['COM_DEST']].index)
            # Trier l'index pour un accès plus rapide
            self.descriptive_data = final_df.sort_index()
            self._save_data(self.descriptive_data, "descriptive_data")
        else:
            self.logger.warning("Aucune données générées trouvées")

    def _get_analytic_data(self):
        df = self.descriptive_data.reset_index().copy()

        trans_cols = ['TRANS_MP', 'TRANS_TC', 'TRANS_VT', 'TRANS_DRM', 'TRANS_VL']

        for col in trans_cols:
            part_col = col.replace("TRANS_", "PART_")
            df[part_col] = df[col] / df['TOTAL_FLUX']

        df = df.sort_values(by=['TRAJET', 'ANNEE'])

        df['DELTA_DUREE'] = df.groupby('TRAJET')['DUREE'].diff()

        for col in trans_cols:
            part_col = col.replace("TRANS_", "PART_")
            delta_col = part_col.replace("PART_", "DELTA_PART_")
            df[delta_col] = df.groupby('TRAJET')[part_col].diff()

        df_analytic = df.copy()

        df_analytic = df_analytic[
            df_analytic['ANNEE'] > 2017
            ].copy()

        self.analytic_data = df_analytic[
            ['TRAJET', 'ANNEE', 'COM_ORG', 'COM_DEST',
             'DELTA_PART_MP', 'DELTA_PART_TC', 'DELTA_PART_VT',
             'DELTA_PART_DRM', 'DELTA_PART_VL', 'DELTA_DUREE']
        ].set_index(['TRAJET', 'ANNEE']).sort_index()
        self._save_data(self.analytic_data, "analytic_data")

    def _save_data(self, data, data_output_name):
        path =f"{self.output_folder}/global_data"
        os.makedirs(path, exist_ok=True)
        try:
            if not data.empty:
                data.to_csv(
                    f"{path}/{data_output_name}.csv",
                    index=False
                )

            self.logger.info(f"Sauvegarde de {data_output_name} terminée")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde : {e}")

    def add_socioeconomic_data(self, path_to_socio_data):
        pass

    def compute_real_durations(self, api_client=None):
        pass

    def run_spatial_analysis(self):

        self.logger.info("Démarrage de l'analyse spatio-statistique")

        # S'assurer que les données sont disponibles
        if self.descriptive_data.empty:
            self.logger.warning("Génération...")
            self._get_descriptive_data()


        if self.analytic_data.empty:
            self.logger.warning("Génération...")
            self._get_analytic_data()


        global_analysis = GlobalAnalysis(
            analytic_data=self.analytic_data,
            descriptive_data=self.descriptive_data,
            ign_communes=self.ign_communes,
            logger=self.logger,
            output_folder=f"{self.output_folder}/spatial_analysis"
        )

        global_analysis.run_full_analysis()

        self.logger.info("Analyse spatio-statistique terminée")

    def process(self):
        self.load_data()
        self.run_spatial_analysis()