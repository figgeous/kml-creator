from pathlib import Path

# Folder paths
SRC_FOLDER = Path("src")
TEMP_FILES = Path("temp_files")
CSV_PATH = TEMP_FILES / "csv_files"
SHP_PATH = TEMP_FILES / "shp_files"
BINNED_SHP_PATH = TEMP_FILES / "binned_shp_files"
POLYGON_SHP_PATH = TEMP_FILES / "polygon_shp_files"
TIF_PATH = TEMP_FILES / "tif_files"
KML_PATH = TEMP_FILES / "kml_files"
SAVED_STATE = TEMP_FILES / "saved_state.json"

# Test path and file names
TEST_PATH = Path("tests")
TEST_CONTEXT_PATH = TEST_PATH / "context"
TEST_CSV_NAME = "test"
TEST_CSV_DIMENSIONS = (430, 2)
TEST_KML_NAME = "expected"
TEST_SHP_NAME = "expected_bin"
TEST_TIF_NAME = "expected"
TEST_POLYGON_SHP_NAME = "expected_polygons"

# GUI Spinner speed
SPINNER_SPEED = 100

# Assorted backend constants
COLUMNS_TO_DROP = ["index", "lat", "lon", "long", "latitude", "longitude"]
ELLIPSOID = "WGS-84"
PROJECTION = "EPSG:4326"
SHAPEFILE_DRIVER = "ESRI Shapefile"
INTERPOLATION_OUTPUT_FORMAT = "GTiff"
INTERPOLATION_OUTPUT_TYPE = "Byte"
INTERPOLATION_ALGORITHM = "average"

# When True, logging is enabled
DEVELOPMENT = True
