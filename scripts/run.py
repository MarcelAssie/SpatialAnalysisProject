from main import *

if __name__ == "__main__":
    input_folder = "../data/input"
    if not os.path.exists(input_folder):
        raise ValueError("Le dossier d'input n'existe pas")
    output_folder = "../data/output/"
    base_cols = ['COM_ORG', 'COM_DEST', 'TOTAL_FLUX', 'TRANS_MP', 'TRANS_TC', 'TRANS_VT', 'TRANS_DRM', 'TRANS_VL', 'DISTANCE']
    spatial_analyzer = SpatialStatisticalAnalysis(input_folder, output_folder, base_cols)
    spatial_analyzer.process()
