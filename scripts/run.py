from main import *

if __name__ == "__main__":
    input_folder = "../data/input"
    output_folder = "../data/output/"
    if not os.path.exists(input_folder) or not os.path.exists(output_folder):
        raise ValueError(f"Le dossier {input_folder or output_folder} n'existe pas")
    base_cols = ['COM_ORG', 'COM_DEST', 'TOTAL_FLUX', 'TRANS_MP', 'TRANS_TC', 'TRANS_VT', 'TRANS_DRM', 'TRANS_VL', 'DISTANCE']
    spatial_analyzer = SpatialStatisticalAnalysis(input_folder, output_folder, base_cols)
    spatial_analyzer.process()
