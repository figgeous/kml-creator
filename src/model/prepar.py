import logging
import os
from pathlib import Path
from typing import Callable

import geopandas as gp
import pandas as pd

from .bin import Bin
from config import BINNED_SHP_PATH
from config import COLUMNS_TO_DROP
from config import CSV_PATH
from config import ELLIPSOID
from config import KML_PATH
from config import PROJECTION
from config import SHAPEFILE_DRIVER
from config import SHP_PATH

logger = logging.getLogger(__name__)


class Preper:
    """
    Prepares the data for the Runner. This includes:
    - Creating a shapefile from the csv if one does not exist
    - Creating seperate shapefiles for each bin
    - Getting the width and height of the dataset in meters
    """

    name: str
    csv_file_name: str
    csv_file_path: Path
    geo_df: gp.GeoDataFrame
    dataset_height: float
    dataset_width: float
    bins: list[Bin]
    _print_to_gui: Callable[[str], None]

    def __init__(
        self,
        name,
        bins: list[Bin],
        save_directory: Path = None,
        csv_file_name: str = None,
        csv_file_path: Path = None,
        print_to_view: Callable[[str], None] = None,
    ):
        assert (
            csv_file_name or csv_file_path
        ), "Must provide either a csv_file_name or a csv_file_path"
        self.name = name
        self.save_directory = save_directory or KML_PATH
        self._print_to_view = print_to_view
        self.csv_file_name = csv_file_name
        self.csv_file_path = csv_file_path
        self.geo_df = self._csv_to_shp()
        self._get_geo_df_dimensions()
        self.bins = bins

    def print_to_view(self, text: str) -> None:
        """
        Prints text to the view's console if self._print_to_gui is not None.
        """
        logger.info(text)
        if print_to_view := self._print_to_view:
            print_to_view(text)

    def _get_geo_df_dimensions(self) -> None:
        """
        Gets the width and height of the dataset in meters and saves them to
        self.dataset_width and self.dataset_height.
        """
        import geopy.distance

        min_lon, min_lat, max_lon, max_lat = (
            self.geo_df.total_bounds[0],
            self.geo_df.total_bounds[1],
            self.geo_df.total_bounds[2],
            self.geo_df.total_bounds[3],
        )

        lower_left = geopy.point.Point(longitude=min_lon, latitude=min_lat)
        upper_left = geopy.point.Point(longitude=min_lon, latitude=max_lat)
        lower_right = geopy.point.Point(longitude=max_lon, latitude=min_lat)
        self.dataset_width = geopy.distance.geodesic(
            lower_left, lower_right, ellipsoid=ELLIPSOID
        ).m
        self.dataset_height = geopy.distance.geodesic(
            lower_left, upper_left, ellipsoid=ELLIPSOID
        ).m

        self.print_to_view(
            f"\nDataset width: {round(self.dataset_width)} m, height: "
            f"{round(self.dataset_height)} m"
        )

    def _csv_to_shp(self) -> gp.GeoDataFrame:
        """
        Creates a shapefile from a csv file with the name csv_file_name. If the
        it is loaded as a geopandas.GeoDataFrame and returned. If the shapefile does
        and returned. If an error occurs when trying to open an existing SHP file, a
        shapefile is done by converting the csv to a pandas.DataFrame, adding a
        objects, and converting the pandas.DataFrame to a geopandas.GeoDataFrame. The
        saved as a shapefile.
        """
        from shapely.geometry import Point

        logger.info("Creating shapefile from csv")
        self.print_to_view("Creating shapefile from csv...")

        if self.csv_file_name:
            self.csv_file_path = CSV_PATH / (self.csv_file_name + ".csv")
        else:
            # Select filename from Path (e.g. "/path/to/file.csv" -> "file")
            self.csv_file_name = self.csv_file_path.stem

        # Give shp_file_name the same name as the csv_file_name
        self.shp_file_path = SHP_PATH / (self.csv_file_name + ".shp.zip")

        # Check if the shapefile already exists. If it does, load it and return it.
        if self.shp_file_path.exists():
            logger.info("Shapefile already exists. Opening...")
            self.print_to_view("Shapefile already exists. Opening...")
            try:
                geo_df = gp.read_file(str(self.shp_file_path))
                logger.info(f"Shapefile opened. Shape: {geo_df.shape}")
                return geo_df
            except Exception as e:
                logger.error(
                    "Could not open geopandas.GeoDataFrame from "
                    + str(self.shp_file_path)
                )
                logger.error(e)
                self.print_to_view("Problem opening shapefile. Creating new one...")

        # Create a new shapefile
        # Create a pandas.DataFrame from the csv
        try:
            input_df = pd.read_csv(self.csv_file_path, sep=",", index_col=0)
            logger.info(
                "Opened csv from %s. Shape: %s", self.csv_file_path, input_df.shape
            )
        except pd.errors.ParserError:
            # If the csv is not separated by commas, try semicolons
            input_df = pd.read_csv(self.csv_file_path, sep=";", index_col=0)
            logger.info(
                "Opened csv from %s. Shape: %s", self.csv_file_path, input_df.shape
            )
        except Exception as e:
            logger.error("Could not open csv from %s. Error: %s", self.csv_file_path, e)
            raise e

        # Create a geometry column with shapely.Point objects
        input_df["geometry"] = input_df.apply(
            lambda x: Point((float(x.lon), float(x.lat))), axis=1
        )
        geo_df = gp.GeoDataFrame(input_df, geometry="geometry", crs="EPSG:4326")
        geo_df = geo_df.reset_index()
        cols_to_drop = [col for col in COLUMNS_TO_DROP if col in geo_df.columns]
        geo_df = geo_df.drop(columns=cols_to_drop)

        # Save the geopandas.GeoDataFrame as a shapefile
        try:
            geo_df.to_file(
                self.shp_file_path,
                driver=SHAPEFILE_DRIVER,
                projection=PROJECTION,
            )
            self.print_to_view("\nShapefile created.")
        except Exception as e:
            logger.error(
                "Could not save geopandas.GeoDataFrame to " + str(self.shp_file_path)
            )
            logger.error(e)
            self.print_to_view("An error occurred. See log for details.")
        return geo_df

    def create_shp_for_each_bin(self) -> None:
        """
        Creates a shapefile for each bin. If the ignore attribute of bin is True or it
        the shapefile already exists, it is skipped.
        """
        for b in self.bins:
            if b.ignore:
                logger.info("Bin {} is ignored. Skipping.".format(b.enum))
                continue
            shp_file_name = "{}-{}-bin_{}".format(self.name, b.column, str(b.enum))
            shp_file_path = str(SHP_PATH / (shp_file_name + ".shp.zip"))

            # if the file exists, open it
            if os.path.exists(shp_file_path):
                msg = "\nShapefile for bin {} already exists. Skipping...".format(
                    b.enum
                )
                self.print_to_view(msg)
                b.bin_shp_file_name = shp_file_name
                return

            self.print_to_view("\nCreating shapefile for bin: " + str(b.enum))

            # Create a new dataframe with only the rows that fall within the bin.lower
            # and bin.upper. Boundary type
            # determines whether the lower and upper bounds are inclusive or exclusive.
            temp_df: gp.GeoDataFrame | None = None
            if b.boundary_type == "[[":  # Inclusive lower, exclusive upper
                temp_df = self.geo_df[
                    (self.geo_df[b.column] >= b.lower)
                    & (self.geo_df[b.column] < b.upper)
                ]
            elif b.boundary_type == "[]":  # Inclusive lower, inclusive upper
                temp_df = self.geo_df[
                    (self.geo_df[b.column] >= b.lower)
                    & (self.geo_df[b.column] <= b.upper)
                ]
            else:
                raise NotImplementedError(
                    "Boundary type not implemented: " + b.boundary_type
                )

            # If the dataframe there must be no data so skip this bin
            if temp_df.empty:
                self.print_to_view("\nBin {} is empty. Ignoring...".format(b.enum))
                b.ignore = True
                continue

            binned_shp_file_path = BINNED_SHP_PATH / (shp_file_name + ".shp.zip")
            temp_df.to_file(
                filename=str(binned_shp_file_path),
                driver=SHAPEFILE_DRIVER,
                crs=PROJECTION,
            )
            logger.info(f"Saved geo_df to {shp_file_path}")
            b.bin_shp_file_name = shp_file_name

    def delete_shp_file(self) -> None:
        """
        Deletes the shapefile created by _csv_to_shp()
        """
        os.remove(self.shp_file_path)

    def __str__(self):
        return f"Preper for {self.name}, using csv: {self.csv_file_name}"
