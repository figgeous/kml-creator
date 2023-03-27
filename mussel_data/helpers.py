import os
from dataclasses import dataclass
from typing import Any

import geopandas as gp
import pandas as pd
import shapely as sp
import simplekml
from constants import *
from osgeo import gdal


@dataclass
class Bin:
    enum: int
    column: str
    description: str
    lower: int
    upper: int
    colour: str
    ignore: bool = False
    boundary_type: str = "[["
    bin_shp_file_name: str = None
    tif_file_name: str = None
    polygon_shp_file_name: str = None
    kml_file_name: str = None


class Runner:
    bins: list[Bin]
    geo_df: gp.GeoDataFrame
    dataset_name: str

    def __init__(self, bins, geo_df, dataset_name):
        self.bins = bins
        self.geo_df = geo_df
        self.dataset_name = dataset_name

    def run_all(self, delete_files_when_done=True):
        self.create_shp_for_each_bin()
        self.run_interpolation_for_each_bin()
        self.run_polygonize_for_each_bin()
        self.create_kml_for_each_bin()
        if delete_files_when_done:
            self.delete_files()

    def create_shp_for_each_bin(self) -> None:
        ## Create a new shapefile for each bin
        for bin in self.bins:
            if bin.ignore:
                continue
            print("Creating shapefile for bin: " + str(bin.enum))
            temp_df = None
            if bin.boundary_type == "[[":
                temp_df = self.geo_df[
                    (self.geo_df[bin.column] >= bin.lower)
                    & (self.geo_df[bin.column] < bin.upper)
                ]
            elif bin.boundary_type == "[]":
                temp_df = self.geo_df[
                    (self.geo_df[bin.column] >= bin.lower)
                    & (self.geo_df[bin.column] <= bin.upper)
                ]
            else:
                raise NotImplementedError(
                    "Boundary type not implemented: " + bin.boundary_type
                )

            if temp_df.empty:
                logging.info("Bin {} is empty. Ignoring.".format(bin.enum))
                bin.ignore = True
                continue
            bin_file_name = "{}-{}-bin_{}".format(
                self.dataset_name, bin.column, str(bin.enum)
            )
            if os.path.exists(str(SHP_PATH / (bin_file_name + ".shp"))):
                os.remove(str(SHP_PATH / (bin_file_name + ".shp")))
            save_geodataframe_to_shp(geo_df=temp_df, file_name=bin_file_name)
            bin.bin_shp_file_name = bin_file_name

    def run_interpolation_for_each_bin(self):
        print("Running interpolation for each bin")
        ## Run the interpolation for each bin
        dataset_width, dataset_height = get_geodf_dimensions(geo_df=self.geo_df)
        radius1_metres = 60
        radius2_metres = 10
        pixel_size = 10  # in metres
        (
            radius1_degrees,
            radies2_degrees,
        ) = coordinate_difference_in_metres_to_degrees_x_and_y(
            geo_df=self.geo_df,
            lon_metres=radius1_metres,
            lat_metres=radius2_metres,
        )
        for bin in self.bins:
            if bin.ignore:
                continue
            print("Running interpolation for bin: " + str(bin.enum))
            tif_file_name = run_interpolation(
                input_shp_name=bin.bin_shp_file_name,
                target_column=bin.column,
                output_tif_name=bin.bin_shp_file_name,
                algorithm="average",
                radius1=radius1_degrees,
                radius2=radies2_degrees,
                dataset_width=int(dataset_width / pixel_size),
                dataset_height=int(dataset_height / pixel_size),
            )
            bin.tif_file_name = tif_file_name

    def run_polygonize_for_each_bin(self):
        ## Run polygonize for each bin
        for bin in self.bins:
            if bin.ignore:
                continue
            print("Running polygonize for bin: " + str(bin.enum))
            output_shp_name = bin.tif_file_name + "_polygons"
            run_polygonize(
                input_tif=bin.tif_file_name,
                output_shp=output_shp_name,
                # mask='none',
                # options=["-mask", tif_name]
            )
            bin.polygon_shp_file_name = output_shp_name

    def create_kml_for_each_bin(self):
        ## Unify and simplify polygons for each bin and make kml
        for bin in self.bins:
            if bin.ignore:
                continue
            print("Making kml for bin: " + str(bin.enum))
            polygon_df = open_shp_with_geopandas(file_name=bin.polygon_shp_file_name)
            polygon_df = polygon_df[polygon_df["DN"] != 0]
            # Unify the polygons
            united = sp.unary_union(polygon_df["geometry"])
            # Simplify the polygon to reduce the number of points
            simplification_tolerance_metres = 10
            x1, y1 = united.bounds[0], united.bounds[1]
            x2, _ = get_coordinate_a_distance_away(
                start_coord=(x1, y1), distance_in_metres=simplification_tolerance_metres
            )
            simplification_tolerance_degrees = x2 - x1
            united = united.simplify(tolerance=simplification_tolerance_degrees)
            kml_file_name = make_kml_from_one_polygon_or_multipolygon(
                file_name=bin.polygon_shp_file_name,
                polygon=united,
                bin=bin,
            )
            bin.kml_file_name = kml_file_name

    def delete_files(self):
        # remove the temporary files
        for bin in self.bins:
            if bin.ignore:
                continue
            print("Removing temporary files for bin: " + str(bin.enum))
            shp_file_path = str(SHP_PATH / bin.bin_shp_file_name) + ".shp.zip"
            tif_file_path = str(TIF_PATH / bin.tif_file_name) + ".tif"
            polygon_shp_file_path = (
                str(SHP_PATH / bin.polygon_shp_file_name) + ".shp.zip"
            )

            if os.path.exists(shp_file_path):
                os.remove(shp_file_path)
                logging.info("Removed file: " + shp_file_path)
            else:
                logging.info("File not found: " + shp_file_path)

            if os.path.exists(tif_file_path):
                os.remove(tif_file_path)
                logging.info("Removed file: " + tif_file_path)
            else:
                logging.info("File not found: " + tif_file_path)

            if os.path.exists(polygon_shp_file_path):
                os.remove(polygon_shp_file_path)
                logging.info("Removed file: " + polygon_shp_file_path)
            else:
                logging.info("File not found: " + polygon_shp_file_path)


### For opening files
def open_csv_as_dataframe(
    *, file_name: str, sep: str = ",", index_col: Any = 0
) -> pd.DataFrame:
    file_name = file_name + ".csv"
    file_path = CSV_PATH / file_name
    try:
        df = pd.read_csv(file_path, sep=sep, index_col=index_col)
        logging.info("Opened csv from %s. Shape: %s", file_path, df.shape)
    except Exception as e:
        logging.error("Can't open csv. %s", e)
    return df


def open_shp_with_geopandas(*, file_name: str) -> gp.GeoDataFrame:
    file_path = str(SHP_PATH / (file_name + ".shp.zip"))
    geo_df: gp.GeoDataFrame = None
    try:
        geo_df = gp.read_file(file_path)
        logging.info(
            f"Opened shapefile from {file_path}. Shape (rows, columns): {geo_df.shape}"
        )
    except Exception as e:
        logging.error("Can't open shapefile. %s", e)
    return geo_df


def open_tif_with_gdal(*, file_name: str) -> gdal.Dataset:
    assert ".tif" not in file_name
    file_path = TIF_PATH / (file_name + ".tif")
    gdal_dataset = gdal.Open(str(file_path), gdal.GA_ReadOnly)
    assert gdal_dataset
    logging.info(f"Opened tif file from {file_path}. Run plot_raster() to view")
    return gdal_dataset


### For saving


def save_geodataframe_to_shp(geo_df: gp.GeoDataFrame, file_name: str) -> None:
    file_path = SHP_PATH / (file_name + ".shp.zip")
    geo_df.to_file(filename=str(file_path), driver="ESRI Shapefile", crs="EPSG:4326")
    logging.info(f"Saved geo_df to {file_path}")


### Display and measurement functions
def get_distance(coords: tuple[float, float, float, float]) -> float:
    """
    Calculates the number of metres between coordinate points.
    Must be (lon_1, lat_1, lon_2, lat_2)
    """
    from geopy.point import Point as geopy_Point
    import geopy.distance

    p1 = geopy_Point(longitude=coords[0], latitude=coords[1])
    p2 = geopy_Point(longitude=coords[2], latitude=coords[3])
    return geopy.distance.geodesic(p1, p2).m


def get_coordinate_a_distance_away(
    start_coord: tuple[float, float], distance_in_metres: int, bearing: int = 90
) -> tuple[float, float]:
    """
    ::params:: start_coord: Must be (lon, lat)
    return in (lon, lat) form
    """
    import geopy.distance

    # p1 = geopy_Point(longitude=start_coord[0], latitude=start_coord[1])
    dest_coord = geopy.distance.distance(meters=distance_in_metres).destination(
        (start_coord[1], start_coord[0]), bearing=bearing
    )
    return dest_coord[1], dest_coord[0]


def get_geodf_dimensions(geo_df: gp.GeoDataFrame) -> tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = (
        geo_df.total_bounds[0],
        geo_df.total_bounds[1],
        geo_df.total_bounds[2],
        geo_df.total_bounds[3],
    )
    dataset_width = get_distance((min_lon, min_lat, min_lon, max_lat))
    dataset_height = get_distance((min_lon, min_lat, max_lon, min_lat))
    logging.info(
        f"The dataset covers an area of {round(dataset_width)} m in dataset_width and {round(dataset_height)} m in dataset_height"
    )

    coord_x_one_metre_away_at_bottom, _ = get_coordinate_a_distance_away(
        start_coord=(min_lon, min_lat),
        distance_in_metres=1,
        bearing=90,
    )
    _, coord_y_one_metre_away_at_bottom = get_coordinate_a_distance_away(
        start_coord=(min_lon, min_lat),
        distance_in_metres=1,
        bearing=0,
    )
    distance_for_same_lon_at_top = get_distance(
        (min_lon, max_lat, coord_x_one_metre_away_at_bottom, max_lat)
    )
    distance_for_same_lat_at_right = get_distance(
        (max_lon, min_lat, max_lon, coord_y_one_metre_away_at_bottom)
    )
    logging.info(
        "1.0 m metre of longitude at the bottom left is {} m at the top left."
        "\n1.0 m metre of latitude at the bottom left is {} m at the bottom "
        "right.".format(
            round(distance_for_same_lon_at_top, 4),
            round(distance_for_same_lat_at_right, 3),
        )
    )
    return dataset_width, dataset_height


def coordinate_difference_in_metres_to_degrees(
    start_coord: tuple[float, float], distance_in_metres: int, bearing: int = 90
) -> tuple[float, float]:
    """
    Calculates the number of degrees away from a start point, given a certain bearing and a distance in metres.
    ::start_coord: Must be in (lon, lat) form
    ::param bearing:: in degrees, so 0 is North, 90 is East, 180 is South, 270 or -90 is West.
    """
    import geopy.distance

    dest_coord = geopy.distance.distance(meters=distance_in_metres).destination(
        (start_coord[1], start_coord[0]), bearing=bearing
    )
    return (dest_coord[1] - start_coord[0]), (dest_coord[0] - start_coord[1])


def coordinate_difference_in_metres_to_degrees_x_and_y(
    geo_df: gp.GeoDataFrame, lon_metres: int, lat_metres: int
) -> tuple[float, float]:
    """
    Calculates the number of degrees away from a start point, given a certain number of
    metres north (lat_metres) and the same for a number of metres east (lon_metres).
    """
    start_coord = (geo_df.total_bounds[0], geo_df.total_bounds[1])
    metres_of_lon, _ = coordinate_difference_in_metres_to_degrees(
        start_coord=start_coord, distance_in_metres=lon_metres
    )
    _, metres_of_lat = coordinate_difference_in_metres_to_degrees(
        start_coord=start_coord, distance_in_metres=lat_metres, bearing=0
    )
    return metres_of_lon, metres_of_lat


def print_tif_metadata(*, tif_name: str) -> None:
    metadata = os.popen("gdalinfo " / TIF_PATH / (tif_name + ".tif")).read()
    print(metadata)


def plot_raster(
    *,
    tif_name: str,
    fig_size: tuple[int, int] = (9, 6),
) -> None:
    """
    Plots a raster image of a tif file using matplotlib.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from osgeo import gdal_array

    dataset = open_tif_with_gdal(file_name=tif_name)
    # Allocate our array using the first band's datatype
    image_datatype = dataset.GetRasterBand(1).DataType

    image = np.zeros(
        (dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount),
        dtype=gdal_array.GDALTypeCodeToNumericTypeCode(image_datatype),
    )
    # Loop over all bands in dataset
    for b in range(dataset.RasterCount):
        # Remember, GDAL index is on 1, but Python is on 0 -- so we add 1 for our GDAL calls
        band = dataset.GetRasterBand(b + 1)
        # Read in the band's data into the third dimension of our array
        image[:, :, b] = band.ReadAsArray()

    plt.rcParams["figure.figsize"] = [fig_size[0], fig_size[1]]
    plt.rcParams["figure.autolayout"] = True

    plt.imshow(image[:, :, 0], origin="lower", extent=[-1, 1, -1, 1], aspect=1)
    plt.colorbar()
    plt.show()


### Basic data processing
def dataframe_to_shp(*, input_df, output_file_name: str) -> None:
    """
    Adds a column of shapely.geometry.point.Point objects corresponding to the lat and
    lon of each row. A geopandas.GeoDataFrame is a subclass of pandas.DataFrame.
    """
    from shapely.geometry import Point

    input_df["geometry"] = input_df.apply(
        lambda x: Point((float(x.lon), float(x.lat))), axis=1
    )
    geo_df = gp.GeoDataFrame(input_df, geometry="geometry", crs="EPSG:4326")
    geo_df = geo_df.reset_index()
    geo_df = geo_df.drop(columns=["index", "lat", "lon"])

    output_file_name += ".shp.zip"
    output_file_path = SHP_PATH / output_file_name
    try:
        geo_df.to_file(
            output_file_path,
            driver="ESRI Shapefile",
            projection="EPSG:4326",
        )
        logging.info(
            "pandas.DataFrame converted to geopandas.GeoDataFrame and saved to"
            + str(output_file_path)
        )
    except Exception as e:
        logging.error(
            "Could not save geopandas.GeoDataFrame to " + str(output_file_path)
        )
        logging.error(e)


def add_square_buffer_to_geo_df(geo_df: gp.GeoDataFrame, size: float = 0.00005):
    """
    Adds a column of square polygons with centre at Point. Size in degrees
    """
    # Buffer the points using a square cap style
    # https://gis.stackexchange.com/questions/314949/creating-square-buffers-around-points-using-shapely
    geo_df["buffer"] = geo_df["geometry"].buffer(size, cap_style=3)
    return geo_df


### Processing tif files
def run_polygonize(
    input_tif: str,
    input_band: int = 1,
    output_shp: str = None,
    output_layer: str = None,
    output_field: str = None,
    mask: str = "default",
    connectedness8: bool = False,
    options: list = None,
) -> None:
    from osgeo_utils.gdal_polygonize import gdal_polygonize

    input_tif_path = TIF_PATH / (input_tif + ".tif")

    if output_shp:
        output_shp_path = SHP_PATH / (output_shp + ".shp.zip")
    else:
        output_shp_path = SHP_PATH / (input_tif + ".shp.zip")

    gdal_polygonize(
        src_filename=str(input_tif_path),
        dst_filename=str(output_shp_path),
        band_number=input_band,
        dst_layername=output_layer,
        dst_fieldname=output_field,
        mask=mask,
        connectedness8=connectedness8,
        options=options,
        quiet=True,
    )
    logging.info("Polygonized " + str(input_tif_path) + " to " + str(output_shp_path))


### KML creation
def make_kml_from_geo_df_single_bin(
    polygon_df: gp.GeoDataFrame,
    kml_file_name: str,
    bin: Bin,
    grouping_col: str = "geometry",
) -> str:
    """
    Takes all polygons in a geo_df and writes them into a kml file
    """
    kml = simplekml.Kml()
    multipolodd = kml.newmultigeometry(name=bin.description)
    for polygon in polygon_df[grouping_col]:
        if polygon.exterior.coords:
            pol = multipolodd.newpolygon(
                name="polygon",
                outerboundaryis=list(polygon.exterior.coords),
            )
            pol.style.polystyle.color = bin.colour
            pol.style.polystyle.outline = 0
    file_path = str(KML_PATH / (kml_file_name + ".kml"))
    kml.save(file_path)
    logging.info("Saved kml file to " + file_path)
    return file_path


def make_kml_from_one_polygon_or_multipolygon(
    file_name: str, polygon: sp.Polygon | sp.MultiPolygon, bin: Bin
) -> str:
    kml = simplekml.Kml()
    multipolodd = kml.newmultigeometry(name="MultiPoly")

    if isinstance(polygon, sp.Polygon):
        polygon = multipolodd.newpolygon(
            name="polygon",
            outerboundaryis=list(polygon.exterior.coords),
        )
        polygon.style.polystyle.color = bin.colour
        polygon.style.polystyle.outline = 0
        file_path = str(KML_PATH / (file_name + ".kml"))
    elif isinstance(polygon, sp.MultiPolygon):
        for polygon in polygon.geoms:
            pol = multipolodd.newpolygon(
                name="polygon",
                outerboundaryis=list(polygon.exterior.coords),
            )
            pol.style.polystyle.color = bin.colour
            pol.style.polystyle.outline = 0
            file_path = str(KML_PATH / (file_name + ".kml"))
    else:
        raise TypeError("Polygon must be a shapely Polygon or MultiPolygon")

    kml.save(file_path)
    logging.info("Saved kml file to " + file_path)
    return file_name


### Algorithm
def run_interpolation(
    *,
    # For gdal_grid
    input_shp_name: str,
    target_column: str,
    output_tif_name: str,
    # Below are associated with GdalOptions
    output_format: str = "Gtiff",
    output_type="Byte",
    dataset_width: int = 0,
    dataset_height: int = 0,
    z_increase=None,
    z_multiply=None,
    outputBounds: list = None,
    algorithm: str = "invdist",
    power: int = None,
    smoothing: float = None,
    radius: float = None,
    radius1: float = None,
    radius2: float = None,
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
    Keyword arguments are :
        input_shp_name ---
        target_column ---
        output_tif_name ---
        output_format --- output format ("GTiff", etc...)
        output_type --- output type (gdalconst.GDT_Byte, etc...)
        dataset_width --- dataset_width of the output raster in pixel
        dataset_height --- dataset_height of the output raster in pixel
        output_res --- resolution of output file
        outputBounds --- assigned output bounds: [ulx, uly, lrx, lry]

        #Related to algorithm
        algorithm --- algorithm to use, e.g. "invdist", "nearest", "average", "linear"
        power --- power used by algorithm
        smoothing --- smoothing used by algorithm
        radius1 ---
        radius2 ---
        angle ---
        max_points ---
        min_points ---
        no_data ---

        #Not implemented:
        outputSRS --- assigned output SRS
        layers --- list of layers to convert
        spatFilter --- spatial filter as (minX, minY, maxX, maxY) bounding box
    """
    assert ".shp.zip" not in input_shp_name
    assert ".tif" not in output_tif_name

    def _get_output_bounds() -> list:
        return [
            "%.18g" % outputBounds[0],
            "%.18g" % outputBounds[2],
            "-tye",
            "%.18g" % outputBounds[1],
            "%.18g" % outputBounds[3],
        ]

    def _get_algorithm_str() -> str:
        s = f"{algorithm}:"
        s += f"power={power}:" if power else ""
        s += f"smoothing={smoothing}:" if smoothing else ""
        s += f"radius={radius}:" if radius else ""
        s += f"radius1={radius1}:" if radius1 else ""
        s += f"radius2={radius2}:" if radius2 else ""
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
    if outputBounds is not None:
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

    grid_options = gdal.GridOptions(options=new_options)

    dest_name = str(TIF_PATH / (output_tif_name + ".tif"))
    src_ds = str(SHP_PATH / (input_shp_name + ".shp.zip"))

    logging.info(
        "Running interpolation on: {} \nOptions: {} \nSaving to: {}".format(
            str(src_ds), str(new_options), str(dest_name)
        )
    )

    idw = gdal.Grid(destName=dest_name, srcDS=src_ds, options=grid_options)
    idw = None
    return output_tif_name
