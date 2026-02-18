import pandas as pd
import geopandas as gpd
from pyproj import Geod
import numpy as np
import os
import logging
from utils import GlobalAnalysis


class SpatialStatisticalAnalysis:
    def __init__(self, input_folder: str, output_folder: str, base_cols: list[str]):
        self.logger = self._setup_logger()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.base_cols = base_cols

        self.ign_communes = gpd.GeoDataFrame()
        self.ign_com_geoms = None
        self.dfs = {}
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

    def load_data(self, delimiter: str = ';'):
        self.logger.info("Chargement des données brutes...")

        for f in os.listdir(self.input_folder):
            path = os.path.join(self.input_folder, f)
            ext = os.path.splitext(f)[1].lower()

            try:
                if ext == '.csv' and "coordinates" not in f:
                    parts = f.split("_")
                    if len(parts) > 2 and parts[2].isdigit():
                        year = int(parts[2])
                        self.dfs[year] = pd.read_csv(path, delimiter=delimiter)

                elif ext == '.shp':
                    self.ign_communes = gpd.read_file(path)
                    if 'INSEE_REG' in self.ign_communes.columns:
                        self.ign_communes = self.ign_communes[self.ign_communes['INSEE_REG'].astype(str) == "11"]

                elif ext == '.gpkg' and "COMMUNE_COMMUNS" in f:
                    self.ign_com_geoms = gpd.read_file(path)

            except Exception as e:
                self.logger.error(f"Erreur chargement {f}: {e}")

        if not self.dfs:
            raise ValueError("Aucun fichier CSV de données trouvé.")

    @staticmethod
    def _transform_data(data):
        mapping = {
            "COMMUNE": "COM_ORG", "DCLT": "COM_DEST",
            "Marche à pied": "TRANS_MP", "Transports en commun": "TRANS_TC",
            "Voiture, camion, fourgonnette": "TRANS_VT", "Deux-roues motorisé": "TRANS_DRM", "Vélo": "TRANS_VL"
        }
        data.rename(columns=mapping, inplace=True)

        cols = ["TRANS_MP", "TRANS_TC", "TRANS_VT", "TRANS_DRM", "TRANS_VL"]
        for c in cols:
            if c not in data.columns: data[c] = 0.0

        data["TOTAL_FLUX"] = data[cols].sum(axis=1)

        for c in ['COM_ORG', 'COM_DEST']:
            data[c] = data[c].astype(str).str.zfill(5)

        return data

    def join_communes_geoms(self, data):
        self.logger.info("Jointure géométrique et calcul des distances...")

        communes_geoms = self.ign_com_geoms.copy()
        valid_geoms_mask = (communes_geoms['geometry'].notna()) & (communes_geoms['geom_dclt'].notna())
        communes_clean = communes_geoms[valid_geoms_mask].copy()

        if communes_clean.crs is None:
            communes_clean = communes_clean.set_crs(epsg=2154)
        else:
            communes_clean = communes_clean.to_crs(epsg=2154)

        communes_clean["centroid_org_geom"] = communes_clean.geometry.centroid
        gs_dest = gpd.GeoSeries.from_wkt(communes_clean["geom_dclt"]).set_crs(communes_clean.crs)
        communes_clean["centroid_dest_geom"] = gs_dest.centroid

        gdf_org_pts = gpd.GeoDataFrame(geometry=communes_clean["centroid_org_geom"], crs=2154).to_crs(epsg=4326)
        communes_clean["lon_org"], communes_clean["lat_org"] = gdf_org_pts.geometry.x, gdf_org_pts.geometry.y

        gdf_dest_pts = gpd.GeoDataFrame(geometry=communes_clean["centroid_dest_geom"], crs=2154).to_crs(epsg=4326)
        communes_clean["lon_dest"], communes_clean["lat_dest"] = gdf_dest_pts.geometry.x, gdf_dest_pts.geometry.y

        ref_geo = communes_clean[["COMMUNE", "DCLT", "lon_org", "lat_org", "lon_dest", "lat_dest"]]
        data_merged = data.merge(ref_geo, left_on=["COM_ORG", "COM_DEST"], right_on=["COMMUNE", "DCLT"], how="inner")
        data_merged.drop(columns=["COMMUNE", "DCLT"], inplace=True)

        g = Geod(ellps='WGS84')
        _, _, dist_meters = g.inv(data_merged['lon_org'].values, data_merged['lat_org'].values,
                                  data_merged['lon_dest'].values, data_merged['lat_dest'].values)
        data_merged['DISTANCE'] = dist_meters / 1000

        return data_merged

    def generate_all_years_duration(self):
        self.logger.info("Simulation de la durée et de la congestion...")
        processed_dfs = {}
        years = sorted(self.dfs.keys())
        base_year = min(years)
        annual_congestion_rate = 0.015

        for year in years:
            df = self.dfs[year].copy()
            df = self._transform_data(df)
            df = self.join_communes_geoms(df)

            theoretical_duration = (df['DISTANCE'] / 30) * 60
            congestion_factor = 1 + ((year - base_year) * annual_congestion_rate)
            random_noise = np.random.normal(1, 0.05, len(df))

            df['DUREE'] = theoretical_duration * congestion_factor * random_noise
            processed_dfs[year] = df

        return processed_dfs

    def _prepare_descriptive_and_analytic(self):
        dfs = self.generate_all_years_duration()
        all_data = []

        for y, df in dfs.items():
            df['ANNEE'] = y
            df['TRAJET'] = df['COM_ORG'] + " - " + df['COM_DEST']
            all_data.append(df)

        self.descriptive_data = pd.concat(all_data)
        self.descriptive_data = self.descriptive_data[
            self.descriptive_data['COM_ORG'] != self.descriptive_data['COM_DEST']]
        self.descriptive_data.set_index(['TRAJET', 'ANNEE'], inplace=True)
        self.descriptive_data.sort_index(inplace=True)

        df_an = self.descriptive_data.copy()
        trans_cols = [c for c in df_an.columns if 'TRANS_' in c]

        for col in trans_cols:
            df_an[col.replace('TRANS_', 'PART_')] = df_an[col] / df_an['TOTAL_FLUX']

        df_an = df_an.sort_values(['TRAJET', 'ANNEE'])
        df_an['DELTA_DUREE'] = df_an.groupby('TRAJET')['DUREE'].diff()

        for col in trans_cols:
            part_col = col.replace('TRANS_', 'PART_')
            df_an[f"DELTA_{part_col}"] = df_an.groupby('TRAJET')[part_col].diff()

        self.analytic_data = df_an.dropna(subset=['DELTA_DUREE'])

        path = f"{self.output_folder}/global_data"
        os.makedirs(path, exist_ok=True)
        self.descriptive_data.to_csv(f"{path}/descriptive_data.csv")
        self.analytic_data.to_csv(f"{path}/analytic_data.csv")

    def process(self):
        self.load_data()
        self._prepare_descriptive_and_analytic()

        analysis = GlobalAnalysis(
            analytic_data=self.analytic_data,
            descriptive_data=self.descriptive_data,
            ign_communes=self.ign_communes,
            logger=self.logger,
            output_folder=f"{self.output_folder}/final_results"
        )
        analysis.run_full_analysis()