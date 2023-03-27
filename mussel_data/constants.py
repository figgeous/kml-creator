import logging
import os
from pathlib import Path

mussel_data_path = Path("mussel_data")
CSV_PATH = mussel_data_path / "csv_files"
SHP_PATH = mussel_data_path / "shp_files"
TIF_PATH = mussel_data_path / "tif_files"
KML_PATH = mussel_data_path / "kml_files"

DEBUG = True

if DEBUG:
    logging.basicConfig(
        filename="app.log",
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
