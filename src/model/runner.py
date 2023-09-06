import logging
import os
from typing import Callable
from typing import TYPE_CHECKING

import geopandas as gp
import geopy.distance
import shapely as sp
import simplekml
from osgeo import gdal
from osgeo_utils.gdal_polygonize import gdal_polygonize

if TYPE_CHECKING:
    from . import Bin
from .prepar import Preper
from config import (
    BINNED_SHP_PATH,
    INTERPOLATION_OUTPUT_FORMAT,
    INTERPOLATION_OUTPUT_TYPE,
    INTERPOLATION_ALGORITHM,
)
from config import KML_PATH
from config import POLYGON_SHP_PATH
from config import TIF_PATH

logger = logging.getLogger(__name__)


class Runner:
    """
    Creates individual shapefiles for each bin, interpolates them, polygonizes them,
    Files are saved in the temp_files directory and deleted when done. The run_all()
    methods in order.
    :param preper: Preper object that contains data pointing to the shapefile to be used
    :param radius_width_metres: Search ellipse width in metres
    :param radius_height_metres: Search ellipse height in metres
    :param pixel_size_metres: Pixel width/height in metres
    :param smoothing: Smoothing tolerance for the interpolation. Greater values result
    :param simplification_tolerance: for the polygonization. Greater values result in
    :param angle: Angle in degrees. Measured counter-clockwise from the positive
    :param print_to_frontend: Function to print to the frontend
    """

    preper: Preper
    radius_x_metres: int
    radius_y_metres: int
    pixel_size_metres: int
    smoothing: int
    simplification_tolerance: int
    angle: int
    print_to_frontend: Callable[[str], None]

    def __init__(
        self,
        preper: Preper,
        radius_width_metres: int,
        radius_height_metres: int,
        pixel_size_metres: int,
        smoothing: int,
        simplification_tolerance: int = 10,
        angle: int = 0,
        print_to_frontend: Callable[[str], None] = None,
    ):
        # Get attributes from Preper
        self.name = preper.name
        self.bins = preper.bins
        self.save_directory = preper.save_directory
        self.geo_df = preper.geo_df
        self.dataset_height = preper.dataset_height
        self.dataset_width = preper.dataset_width
        self.bins = preper.bins
        self.preper = preper

        # User-defined attributes
        self.radius_x_metres = radius_width_metres
        self.radius_y_metres = radius_height_metres
        self.pixel_size_metres = pixel_size_metres
        self.smoothing = smoothing
        self.simplification_tolerance = simplification_tolerance
        self.angle = angle

        self._print_to_frontend = print_to_frontend

    def print_to_frontend(self, text: str) -> None:
        """
        Prints to the frontend if the print_to_frontend attribute is not None
        :param text: Text to print
        """
        logger.info(text)
        if self._print_to_frontend:
            self._print_to_frontend(text)

    def run_all(self, delete_files_when_done=True) -> None:
        """
        Runs all the methods in order
        :param delete_files_when_done: Whether to delete the files when done
        """
        self.run_interpolation_for_each_bin()
        self.run_polygonize_for_each_bin()
        self.create_kml_for_each_bin()

        if delete_files_when_done:
            self.delete_files()

    def run_interpolation_for_each_bin(
        self,
        radius_x_metres=None,
        radius_y_metres=None,
        pixel_size_metres=None,  # in metres
        smoothing=None,
    ) -> None:
        """
        Runs the interpolation for each bin. If bin.ignore is True, it is skipped. When
        search ellipse, the radius_x_metres and radius_y_metres are used. These are in
        to degrees using the centre of the dataset as the reference point. This is
        earth, which is not perfectly spherical.
        This method contains run_interpolation() which is the method that actually runs
        is called for each bin.
        """

        def run_interpolation(
            *,
            input_shp_name: str,
            target_column: str,
            output_tif_name: str,
            output_format: str = INTERPOLATION_OUTPUT_FORMAT,
            output_type=INTERPOLATION_OUTPUT_TYPE,
            dataset_width: int = 0,
            dataset_height: int = 0,
            z_increase=None,
            z_multiply=None,
            output_bounds: list = None,
            algorithm: str = INTERPOLATION_ALGORITHM,
            power: int = None,
            smoothing: float = None,
            radius: float = None,
            radius_width: float = None,
            radius_height: float = None,
            angle: int = None,
            max_points: int = None,
            min_points: int = None,
            max_points_per_quadrant: int = 0,
            min_points_per_quadrant: int = 0,
            nodata: float = None,
            where: str = None,
            sql: str = None,
        ) -> str:
            """
            Runs the gdal.Grid function to interpolate a raster from a shapefile.
            :param input_shp_name: The name of the shapefile to be interpolated.
            :param target_column: The column in the shapefile to be interpolated.
            :param output_tif_name: The name of the output tif file.
            :param output_format: The format of the output file.
            :param output_type: The type of the output file. Default is Byte.
            :param dataset_width: The width of the output file.
            :param dataset_height: The height of the output file.
            :param z_increase: The amount to increase the z values by. Z values are the
            :param z_multiply: The amount to multiply the z values by. Z values are the
            :param output_bounds: The bounds of the output file.
            :param algorithm: The algorithm to use for interpolation. Options are
            :param power: The power to use for the inverse distance to a power
            :param smoothing: The smoothing to use for the average distance algorithm.
            :param radius: The radius to use for the average distance algorithm.
            :param radius_width: The radius1 to use for the average distance algorithm.
            :param radius_height: The radius2 to use for the average distance
            :param angle: The angle to use for the average distance algorithm. This is
            :param max_points: The max_points to use for the average distance algorithm.
            :param min_points: The min_points to use for the average distance algorithm.
            :param max_points_per_quadrant: The max_points_per_quadrant to use for the
            :param min_points_per_quadrant: The min_points_per_quadrant to use for the
            :param nodata: The nodata value to use for the average distance algorithm.
            :param where: The where clause to use for the average distance algorithm.
            :param sql: The sql to use for the average distance algorithm.
            """

            def _get_output_bounds() -> list:
                """
                Returns the output bounds for the gdal.Grid function. 18 decimal places
                nanometres. This is unneccessarily precise, but why not? Gdal.Grid will
                Gdal accepts the bounds in the order min_x, max_x, min_y, max_y.
                """
                min_x, min_y, max_x, max_y = output_bounds
                return [
                    "%.18g" % min_x,
                    "%.18g" % max_x,
                    "%.18g" % min_y,
                    "%.18g" % max_y,
                ]

            def _get_algorithm_str() -> str:
                """
                Returns the algorithm string for the gdal.Grid function.
                """
                s = f"{algorithm}:"
                s += f"power={power}:" if power else ""
                s += f"smoothing={smoothing}:" if smoothing else ""
                s += f"radius={radius}:" if radius else ""
                s += f"radius1={radius_width}:" if radius_width else ""
                s += f"radius2={radius_height}:" if radius_height else ""
                s += f"angle={angle}:" if angle else ""
                s += f"max_points={max_points}:" if max_points else ""
                s += f"min_points={min_points}:" if min_points else ""
                s += (
                    f"max_points_per_quadrant={max_points_per_quadrant}:"
                    if max_points_per_quadrant
                    else ""
                )
                s += (
                    f"min_points_per_quadrant={min_points_per_quadrant}:"
                    if min_points_per_quadrant
                    else ""
                )
                s += f"nodata={nodata}:" if nodata else ""
                return s

            new_options = []
            if output_format is not None:
                new_options += ["-of", output_format]
            if output_type is not None:
                new_options += ["-ot", output_type]
            if dataset_width != 0 or dataset_height != 0:
                new_options += ["-outsize", str(dataset_width), str(dataset_height)]
            if output_bounds is not None:
                new_options += ["-txe"] + _get_output_bounds()
            if algorithm is not None:
                new_options += ["-a", _get_algorithm_str()]
            if target_column is not None:
                new_options += ["-zfield", target_column]
            if z_increase is not None:
                new_options += ["-z_increase", str(z_increase)]
            if z_multiply is not None:
                new_options += ["-z_increase", str(z_multiply)]
            if sql is not None:
                new_options += ["-sql", str(sql)]
            if where is not None:
                new_options += ["-where", str(where)]

            # Create the grid options object
            grid_options = gdal.GridOptions(options=new_options)

            dest_name = str(TIF_PATH / (output_tif_name + ".tif"))
            src_ds = str(BINNED_SHP_PATH / (input_shp_name + ".shp.zip"))

            logger.info(
                "Running interpolation on: {} \nOptions: {} \nSaving to: {}".format(
                    str(src_ds), str(new_options), str(dest_name)
                )
            )

            # Run the interpolation
            gdal.Grid(destName=dest_name, srcDS=src_ds, options=grid_options)

            return output_tif_name

        # If no values are passed, use the values from when Runner was initialized
        radius_x_metres = radius_x_metres or self.radius_x_metres
        radius_y_metres = radius_y_metres or self.radius_y_metres
        pixel_size_metres = pixel_size_metres or self.pixel_size_metres
        smoothing = smoothing or self.smoothing

        # Convert the radius from metres to degrees using centre of the dataset as
        # the reference point
        min_x, min_y, max_x, max_y = self.geo_df.total_bounds
        centre_of_geo_df = (
            ((max_x - min_x) / 2 + min_x),
            ((max_y - min_y) / 2 + min_y),
        )  # (longitude, latitude)

        # dest_coord_x is radius_x_metres metres eastwards. Point() object is
        # (latitude, longitude)
        distance_x = geopy.distance.distance(meters=radius_x_metres)
        dest_coord_x: geopy.Point = distance_x.destination(
            (centre_of_geo_df[1], centre_of_geo_df[0]), bearing=90  # East
        )
        radius_x_degrees = dest_coord_x[1] - centre_of_geo_df[0]  # longitude
        # dest_coord_y is radius_y_metres metres northwards. Point() object is
        # (latitude, longitude)
        distance_y = geopy.distance.distance(meters=radius_y_metres)
        dest_coord_y: geopy.Point = distance_y.destination(
            (centre_of_geo_df[1], centre_of_geo_df[0]), bearing=0  # North
        )
        radius_y_degrees = dest_coord_y[0] - centre_of_geo_df[1]  # latitude

        # Run the interpolation for each bin
        for b in self.bins:
            if b.ignore:
                continue

            msg = "\nRunning interpolation for bin: " + str(b.enum)
            self.print_to_frontend(msg)

            tif_file_name = run_interpolation(
                input_shp_name=b.bin_shp_file_name,
                target_column=b.column,
                output_tif_name=b.bin_shp_file_name,
                algorithm="average",
                radius_width=radius_x_degrees,
                radius_height=radius_y_degrees,
                smoothing=smoothing,
                dataset_width=int(self.dataset_width / pixel_size_metres),
                dataset_height=int(self.dataset_height / pixel_size_metres),
                angle=int(self.angle),
            )
            b.tif_file_name = tif_file_name

    def run_polygonize_for_each_bin(
        self,
        simplification_tolerance: int = None,
    ) -> None:
        """
        Runs the polygonization of each bin. If bin.ignore is True, it is skipped.
        """
        # Run polygonize for each bin
        for b in self.bins:
            if b.ignore:
                continue

            self.print_to_frontend("\nRunning polygonize for bin: " + str(b.enum))

            output_shp_name = b.tif_file_name + "_polygons"

            input_tif_path = TIF_PATH / (b.tif_file_name + ".tif")

            output_shp_path = POLYGON_SHP_PATH / (output_shp_name + ".shp.zip")

            gdal_polygonize(
                src_filename=str(input_tif_path),
                dst_filename=str(output_shp_path),
                band_number=1,
                dst_layername=None,
                dst_fieldname=None,
                mask="default",
                connectedness8=False,
                options=None,
                quiet=True,
            )
            logger.info(
                "Polygonized " + str(input_tif_path) + " to " + str(output_shp_path)
            )

            b.polygon_shp_file_name = output_shp_name

    def create_kml_for_each_bin(self) -> None:
        """
        Creates a kml for each bin. If the ignore of bin is True, it is skipped. The
        before the kml is created. The simplification tolerance is in metres and
        are simplified. This method contains the
        to create the kml from a shapely Polygon or MultiPolygon.
        """

        # Convert the simplification tolerance from metres to degrees
        x1, y1 = self.geo_df.total_bounds[0:2]  # lon, lat
        # Get the coordinate a certain distance away. Returns a tuple of (lat, lon)
        dest_coord = geopy.distance.distance(
            meters=self.simplification_tolerance,
        ).destination(
            (y1, x1), bearing=90  # 90 degrees is east
        )
        x2 = dest_coord[1]  # lon
        simplification_tolerance_degrees = x2 - x1

        # Create kml for each bin
        for bin in self.bins:
            if bin.ignore:
                continue
            self.print_to_frontend("\nMaking kml for bin: " + str(bin.enum))

            # Open the polygon shapefile using geopandas
            polygon_shp_file_path = POLYGON_SHP_PATH / (
                bin.polygon_shp_file_name + ".shp.zip"
            )
            polygon_df: gp.GeoDataFrame | None = None
            if not os.path.exists(polygon_shp_file_path):
                logger.exception("File does not exist: %s", polygon_shp_file_path)
                continue
            try:
                polygon_df = gp.read_file(
                    str(polygon_shp_file_path), driver="ESRI Shapefile"
                )
                logger.info(
                    f"Opened shapefile from {polygon_shp_file_path}. "
                    f"Shape (rows, columns): {self.geo_df.shape}"
                )
            except Exception as e:
                logger.error("Can't open shapefile. %s", e)
                self.print_to_frontend("A problem occurred. Please try again.")

            if polygon_df is None:
                self.print_to_frontend(
                    f"The polygon shapefile for bin {bin.enum} is empty. Skipping"
                )
                continue

            # Remove the polygons with zero values. These are not of value to the user.
            polygon_df = polygon_df[polygon_df["DN"] != 0]
            if polygon_df.empty:
                msg = "\nBin {} is empty. Ignoring...".format(bin.enum)
                logger.info(msg)
                self.print_to_frontend(msg)
                bin.ignore = True
                continue

            # Unify and simplify the polygons
            united: sp.Polygon | sp.MultiPolygon = sp.unary_union(
                polygon_df["geometry"]
            )
            united = united.simplify(tolerance=simplification_tolerance_degrees)

            def _get_kml_colour(b: "Bin") -> str:
                """
                KML colour uses the format AABBGGRR, where AA is the alpha value, BB is
                value, and RR is the red value. The alpha value is the opacity of the
                transparent and FF is fully opaque. Values are specified in
                scaled from 0-100 to 0-255 and converted to hexadecimal before being
                """
                from simplekml import Color

                colour_without_hash = b.colour[1:]
                alpha = hex(int(b.opacity / 100 * 255))[
                    2:
                ]  # first two characters are 0x
                return Color.hexa(colour_without_hash + alpha)

            # Create the kml object and a MultiGeometry object
            bin.kml_file_name = bin.polygon_shp_file_name
            kml = simplekml.Kml()
            multi_geometry: simplekml.MultiGeometry = kml.newmultigeometry(
                name=bin.description
            )
            # Polygon or MultiPolygon objects can be added to the MultiGeometry object
            if isinstance(united, sp.Polygon):
                united = multi_geometry.newpolygon(
                    name=bin.description,
                    outerboundaryis=list(united.exterior.coords),
                )
                united.style.polystyle.color = _get_kml_colour(bin)
                united.style.polystyle.outline = 0
            elif isinstance(united, sp.MultiPolygon):
                for polygon in united.geoms:
                    pol = multi_geometry.newpolygon(
                        name=bin.description,
                        outerboundaryis=list(polygon.exterior.coords),
                    )
                    pol.style.polystyle.color = _get_kml_colour(bin)
                    pol.style.polystyle.outline = 0
            else:
                raise TypeError("Polygon must be a shapely Polygon or MultiPolygon")

            # Save the kml file
            if save_directory := self.save_directory:
                file_path = str(save_directory / (bin.kml_file_name + ".kml"))
            else:
                file_path = str(KML_PATH / (bin.kml_file_name + ".kml"))

            try:
                kml.save(file_path)
            except Exception as e:
                logger.exception("Can't save kml file. %s", e)
                self.print_to_frontend("A problem occurred. Please try again.")
            logger.info("Saved kml file to " + file_path)

    def delete_files(
        self,
        delete_binned_shp_files: bool = True,
        delete_polygon_shp_files: bool = True,
        delete_tif_files: bool = True,
        delete_kml_files: bool = False,
    ) -> None:
        """
        Deletes the files created by the Runner.
        :param delete_binned_shp_files: If True, deletes the binned shp files
        :param delete_polygon_shp_files: If True, deletes the polygon shp files
        :param delete_tif_files: If True, deletes the tif files
        :param delete_kml_files: If True, deletes the kml files
        """
        # Remove the temporary files
        for bin in self.bins:
            if bin.ignore:
                continue

            self.print_to_frontend(
                "\nRemoving temporary files for bin: " + str(bin.enum)
            )

            if delete_binned_shp_files and hasattr(bin, "bin_shp_file_name"):
                binned_shp_file_path = (
                    str(BINNED_SHP_PATH / bin.bin_shp_file_name) + ".shp.zip"
                )
                self._delete_file(file_path=binned_shp_file_path)

            if delete_tif_files and hasattr(bin, "tif_file_name"):
                tif_file_path = str(TIF_PATH / bin.tif_file_name) + ".tif"
                self._delete_file(file_path=tif_file_path)

            if delete_polygon_shp_files and hasattr(bin, "polygon_shp_file_name"):
                polygon_shp_file_path = (
                    str(POLYGON_SHP_PATH / bin.polygon_shp_file_name) + ".shp.zip"
                )
                self._delete_file(file_path=polygon_shp_file_path)

            if delete_kml_files and hasattr(bin, "kml_file_name"):
                kml_file_path = str(KML_PATH / bin.kml_file_name) + ".kml"
                self._delete_file(file_path=kml_file_path)

    def _delete_file(self, file_path: str) -> None:
        """
        Deletes the file at the given file path.
        """
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info("Removed file: " + file_path)
        else:
            logger.info("File not found: " + file_path)

    def __str__(self):
        return (
            f"Runner class for: {self.name}, using csv_file: "
            f"{self.preper.csv_file_name}"
        )
