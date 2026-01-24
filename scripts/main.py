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
        self.data_2016 = pd.DataFrame()
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



    def load_data(self, code_geo_reg:str ='INSEE_REG', code_geo_com:str ='INSEE_COM', layer:str="COMMUNE", delimiter:str=';'):
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
                        # Extract year from filename (e.g., FD_MOBPRO_2016_result.csv)
                        parts = os.path.basename(f_path).split("_")
                        if len(parts) > 2 and parts[2].isdigit():
                            year = int(parts[2])
                            self.dfs[year] = df
                            if year == 2021:
                                self.insee_communes = df.copy()
                    else:
                        if file_ext.lower() == '.shp':
                            ign_communes = gpd.read_file(f_path)
                        elif file_ext.lower() == '.gpkg':
                            if "COMMUNE_COMMUNS" in f_path:
                                self.ign_com_geoms = gpd.read_file(f_path)
                                ign_communes = self.ign_com_geoms
                            else:
                                ign_communes = gpd.read_file(f_path)
                        else:
                            continue
                except Exception as e:
                    self.logger.error(f"Erreur lors du chargement de {f_path}: {e}")

            if not self.dfs[2016].empty:
                self.data_2016 = self.dfs[2016].copy()

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
        self.logger.info("Transformation des données récupérée")
        data.rename(columns={
            "COMMUNE": "COM_ORG", "DCLT": "COM_DEST", "Deux-roues motorisé": "TRANS_DRM",
            "Marche à pied": "TRANS_MP", "Pas de transport (travail domicile)": "TRANS_M",
            "Transports en commun": "TRANS_TC", "Voiture, camion, fourgonnette": "TRANS_VT",
            "Vélo": "TRANS_VL"
        }, inplace=True)


        #-------------------------------------------------------------------------
        # # TODO : Demander à Imane de garder 'TOTAL_FLUX' dans les données de base
        # cols_flux = ['TRANS_DRM', 'TRANS_MP', 'TRANS_TC', 'TRANS_VT', 'TRANS_VL']
        # df["TOTAL_FLUX"] = df[cols_flux].sum(axis=1)
        #-------------------------------------------------------------------------
        data['COM_ORG'] = data['COM_ORG'].astype(str)
        data['COM_DEST'] = data['COM_DEST'].astype(str)
        data['COM_ORG'] = data['COM_ORG'].str.zfill(5)
        data['COM_DEST'] = data['COM_DEST'].str.zfill(5)
        return data

    def join_communes_geoms(self, data, code_geo_com: str = "COMMUNE"):
        self.logger.info("Ajout des centroids origine / destination")

        communes = self.ign_communes.set_geometry("geometry")

        if communes.crs is None:
            communes = communes.set_crs(epsg=2154)

        communes_l93 = communes.to_crs(epsg=2154)

        # Calcul des centroids
        communes_l93["centroid_org"] = communes_l93.geometry.centroid
        communes_l93["centroid_dest"] = communes_l93.geom_dclt.centroid

        # Reprojection en WGS84
        communes_wgs84 = communes_l93.to_crs(epsg=4326)

        # Extraction coordonnées
        communes_wgs84["lon_org"] = communes_wgs84.centroid_org.x
        communes_wgs84["lat_org"] = communes_wgs84.centroid_org.y
        communes_wgs84["lon_dest"] = communes_wgs84.centroid_dest.x
        communes_wgs84["lat_dest"] = communes_wgs84.centroid_dest.y

        # Colonnes utiles uniquement
        ref = communes_wgs84[
            ["COMMUNE", "DCLT", "lon_org", "lat_org", "lon_dest", "lat_dest"]
        ].copy()

        # Jointure OD
        data_merged = data.merge(
            ref,
            left_on=["COM_ORG", "COM_DEST"],
            right_on=["COMMUNE", "DCLT"],
            how="left"
        )

        # Nettoyage
        data_merged.drop(columns=["COMMUNE", "DCLT"], inplace=True)
        self._save_data(data_merged, "data_merged")

        return data_merged

    def compute_a_year_duration(self):
        self.logger.info("Ajout de la 'durée' entre chaque paire de communes sur de 2016 à 2021")
        data_merged = self.data_2016.copy()
        try:
            g = Geod(ellps='WGS84')
            mask_valid = data_merged['lon_org'].notna() & data_merged['lon_dest'].notna()

            data_merged['DISTANCE'] = np.nan

            if mask_valid.any():
                _, _, dist_meters = g.inv(
                    data_merged.loc[mask_valid, 'lon_org'].values,
                    data_merged.loc[mask_valid, 'lat_org'].values,
                    data_merged.loc[mask_valid, 'lon_dest'].values,
                    data_merged.loc[mask_valid, 'lat_dest'].values
                )
                data_merged.loc[mask_valid, 'DISTANCE'] = dist_meters / 1000

            data_merged = data_merged[~data_merged['DISTANCE'].isna()]

            self.data_2016 = data_merged[self.base_cols].copy()
            vitesse_tc = 25
            duree_base = self.data_2016['DISTANCE'] / vitesse_tc
            variation = np.random.normal(0, 0.05, len(self.data_2016))
            self.data_2016['DUREE'] = duree_base * 60 * (1 + variation)
            self._save_data(self.data_2016, "2016_data_with_duration")
        except Exception as e:
            self.logger.error(e)

    def generate_all_years_duration(self):
        self.logger.info("Calculs préliminaires sur l'année 2016")
        self.data_2016 = self._transform_data(self.data_2016)
        self.data_2016 = self.join_communes_geoms(self.data_2016)
        raise
        self.compute_a_year_duration()
        self.dfs[2016] = self.data_2016
        self.logger.info("Calculs sur les années 2017 à 2021")
        for year in range(2017, 2022):
            df_prev = self.dfs[year - 1].copy()

            # Garder les colonnes de base
            df = self.dfs[year].copy()
            df = self._transform_data(df)
            df = self.join_communes_geoms(df)
            df = df[self.base_cols].copy()

            # Calculer la durée pour l'année en cours
            variation_duree = np.random.normal(0, 0.05, len(df))

            # Récupérer la durée de l'année précédente
            prev_duree = df_prev['DUREE'].copy() if year == 2016 else df_prev['DUREE'].copy()
            df['DUREE'] = prev_duree * (1 + variation_duree)

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
            df_analytic['ANNEE'] > 2016
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

            self.logger.info("Sauvegarde des données terminée")
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
            self.logger.warning("Données descriptives non disponibles, génération...")
            self._get_descriptive_data()


        if self.analytic_data.empty:
            self.logger.warning("Données analytiques non disponibles, génération...")
            self._get_analytic_data()


        # global_analysis = GlobalAnalysis(
        #     analytic_data=self.analytic_data,
        #     descriptive_data=self.descriptive_data,
        #     ign_communes=self.ign_communes,
        #     logger=self.logger,
        #     output_folder=f"{self.output_folder}/spatial_analysis"
        # )

        # global_analysis.run_full_analysis()

        self.logger.info("Analyse spatio-statistique terminée")

    def process(self):
        self.load_data()
        self.run_spatial_analysis()




