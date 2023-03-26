import logging
import os
from pathlib import Path

CSV_PATH = Path("./csv_files")
SHP_PATH = Path("./shapefiles")
TIF_PATH = Path("./tif_files")
KML_PATH = Path("./kml_files")

DEBUG = True

if DEBUG:
    logging.basicConfig(
        filename="app.log",
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
