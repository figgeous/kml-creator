import os

REPO_FOLDER = "mussel_data"
# Change working directory to mussel_data if not already there
cur_dir = os.getcwd()
if REPO_FOLDER not in cur_dir:
    os.chdir(REPO_FOLDER)
    print("Changed working directory to " + REPO_FOLDER)
    print(os.getcwd())

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
CSV_PATH = "./csv_files/"
SHP_PATH = "./shapefiles/"
TIF_PATH = "./tif_files/"
KML_PATH = "kml_files/"
