import os
from pathlib import Path

CSV_PATH = Path("./csv_files")
SHP_PATH = Path("./shapefiles")
TIF_PATH = Path("./tif_files")
KML_PATH = Path("./kml_files")


a = CSV_PATH / "test" / ".csv"
print(a)
