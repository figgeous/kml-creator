from dataclasses import dataclass
from typing import Any

import geopandas as gp
import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely as sp
import simplekml
from geopy.point import Point as geopy_Point
from osgeo import gdal
from osgeo import gdal_array
from osgeo import ogr
from shapely.geometry import Point

from .constants import *


@dataclass
class Bin:
    column: str
    enum: int
    description: str
    lower: int
    upper: int
    colour: str
    boundary_type: str = "[["
    bin_shp_file_name: str = None
    tif_file_name: str = None
    polygon_shp_file_name: str = None
    kml_file_name: str = None


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


def open_shp_with_gdal(*, file_name: str) -> ogr.DataSource:
    assert ".shp" not in file_name
    file_path = SHP_PATH + file_name + ".shp.zip"
    ogr_datasource = ogr.Open(file_path)
    assert ogr_datasource
    logging.info(f"Opened shapefile from {file_path}")
    return ogr_datasource


def open_shp_with_geopandas(*, file_name: str) -> gp.GeoDataFrame:
    file_name = file_name + ".shp.zip"
    file_path = SHP_PATH / file_name
    try:
        geo_df = gp.read_file(file_path)
        logging.info(
            f"Opened shapefile from {file_path}. Shape (rows, columns): {geo_df.shape}"
        )
    except Exception as e:
        logging.error("Can't open shapefile. %s", e)
    return geo_df


def open_or_create_and_open_shp_with_geopandas(
    file_name: str,
    bins: tuple[Bin],
    target_col_for_enum: str,
    make_new_geo_df: bool = False,
    sep: str = ",",
) -> tuple[gp.GeoDataFrame, int, int]:
    make_new_geo_df = False
    geo_df = None
    try:
        geo_df = open_shp_with_geopandas(file_name=file_name)
    except Exception:
        logging.exception("Count not load shapefile.")

    if make_new_geo_df or geo_df.empty:
        df = open_csv_as_dataframe(file_name=file_name, sep=sep, index_col=0)
        df = add_bin_enum_to_df(df=df, bins=bins, column=target_col_for_enum)
        geo_df = dataframe_to_shp(input_df=df)
        save_geodataframe_to_shp(
            geo_df=geo_df,
            file_name=file_name,
        )
    width, height = get_geodf_dimensions(geo_df=geo_df)
    return geo_df, width, height


def open_tif_with_gdal(*, file_name: str) -> gdal.Dataset:
    assert ".tif" not in file_name
    file_path = TIF_PATH / (file_name + ".tif")
    gdal_dataset = gdal.Open(str(file_path), gdal.GA_ReadOnly)
    assert gdal_dataset
    logging.info(f"Opened tif file from {file_path}. Run plot_raster() to view")
    return gdal_dataset


### For saving
def save_dataframe_to_csv(
    *, df: pd.DataFrame, file_name: str, sep: str = ",", index_col: str = None
) -> pd.DataFrame:
    file_path = CSV_PATH + file_name + ".csv"
    return df.to_csv(path_or_buf=file_path, sep=sep)


def save_geodataframe_to_shp(geo_df: gp.GeoDataFrame, file_name: str) -> None:
    file_path = SHP_PATH / (file_name + ".shp.zip")
    geo_df.to_file(filename=str(file_path), driver="ESRI Shapefile", crs="EPSG:4326")
    logging.info(f"Saved geo_df to {file_path}")


### Display and measurement functions
def get_coordinate_a_distance_away(
    start_coord: tuple[float, float], distance_in_metres: int, bearing: int = 90
) -> tuple[float, float]:
    """
    ::params:: start_coord: Must be (lon, lat)
    return in (lon, lat) form
    """
    # p1 = geopy_Point(longitude=start_coord[0], latitude=start_coord[1])
    dest_coord = geopy.distance.distance(meters=distance_in_metres).destination(
        (start_coord[1], start_coord[0]), bearing=bearing
    )
    return dest_coord[1], dest_coord[0]


### Display and measurement functions


def get_geodf_dimensions(geo_df: gp.GeoDataFrame) -> tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = (
        geo_df.total_bounds[0],
        geo_df.total_bounds[1],
        geo_df.total_bounds[2],
        geo_df.total_bounds[3],
    )
    width = get_distance((min_lon, min_lat, min_lon, max_lat))
    height = get_distance((min_lon, min_lat, max_lon, min_lat))
    logging.info(
        f"The dataset covers an area of {round(width)} m in width and {round(height)} m in height"
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
    return width, height


def get_distance(coords: tuple[float, float, float, float]) -> float:
    """
    Calculates the number of metres between coordinate points.
    Must be (lon_1, lat_1, lon_2, lat_2)
    """
    p1 = geopy_Point(longitude=coords[0], latitude=coords[1])
    p2 = geopy_Point(longitude=coords[2], latitude=coords[3])
    return geopy.distance.geodesic(p1, p2).m


def get_coordinate_a_distance_away(
    start_coord: tuple[float, float], distance_in_metres: int, bearing: int = 90
) -> tuple[float, float]:
    """
    Must be (lon, lat)
    """
    dest_coord = geopy.distance.distance(meters=distance_in_metres).destination(
        (start_coord[1], start_coord[0]), bearing=bearing
    )
    return dest_coord[1], dest_coord[0]


def coordinate_difference_in_metres_to_degrees(
    start_coord: tuple[float, float], distance_in_metres: int, bearing: int = 90
) -> tuple[float, float]:
    """
    Calculates the number of degrees away from a start point, given a certain bearing and a distance in metres.
    ::start_coord: Must be in (lon, lat) form
    ::param bearing:: in degrees, so 0 is North, 90 is East, 180 is South, 270 or -90 is West.
    """
    dest_coord = geopy.distance.distance(meters=distance_in_metres).destination(
        (start_coord[1], start_coord[0]), bearing=bearing
    )
    return (dest_coord[1] - start_coord[0]), (dest_coord[0] - start_coord[1])


def coordinate_difference_in_metres_to_degrees_x_and_y(
    geo_df: gp.GeoDataFrame, lon_metres: int, lat_metres: int
) -> tuple[float, float]:
    """
    Calculates the number of degrees away from a start point, given a certain number of
    metres north (lat_metres) and the same the for a number of metres east (lon_metres).
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


def print_first_feature_of_shp(file):
    """
    Should print out something like:
    {"type": "Feature", "geometry": {"type": "Point",
    "coordinates": [8.943549, 56.996942]}, "properties":
    {"bm_dens": 10, "bm_size": 58}, "id": 0}
    """
    shape = file.GetLayer(0)
    # first feature of the shapefile
    feature = shape.GetFeature(1)
    first = feature.ExportToJson()
    print(first)  # (GeoJSON format)


def plot_raster(
    *,
    tif_name: str,
    fig_size: tuple[int, int] = (9, 6),
) -> None:
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
def grid_to_coordinate(*, grid_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Converts a grid format to a list of values and their coordinates. The grid is
    2 dimensional with latitude on rows and longitude on columns. Lists of items and
    their coordinates are easier to process. Target column is the resulting column
    name.
    """
    grid_df = grid_df.reset_index()
    melt = grid_df.melt(
        id_vars=["index"], var_name=grid_df.columns[1], value_name=target_column
    )
    melt.columns = ["lat", "lon", target_column]
    melt = melt.apply(pd.to_numeric)
    return melt


def dataframe_to_shp(*, input_df, output_file_name: str) -> None:
    """
    Adds a column of shapely.geometry.point.Point objects corresponding to the lat and
    lon of each row. A geopandas.GeoDataFrame is a subclass of pandas.DataFrame.
    """
    # combine lat and lon column to a shapely Point() object
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


def add_bin_enum_to_df(df: pd.DataFrame, bins: tuple[Bin], column: str) -> pd.DataFrame:
    """
    Adds a column to df with a binned version of column. Each row in df that falls
    within bin.lower_bound and bin.upper_bound is given the value bin.enum. The new
    column is the original column +"_b"
    """
    conditions, categories = [], []
    for bin in bins:
        conditions.append(
            (df[column] >= bin.lower_bound) & (df[column] < bin.upper_bound)
        )
        categories.append(bin.enum)
    df[column + "_s"] = np.select(conditions, categories)
    return df


def add_square_buffer_to_geo_df(geo_df: gp.GeoDataFrame, size: float = 0.00005):
    """
    Adds a column of square polygons with centre at Point. Size in degrees
    """
    # Buffer the points using a square cap style
    # https://gis.stackexchange.com/questions/314949/creating-square-buffers-around-points-using-shapely
    geo_df["buffer"] = geo_df["geometry"].buffer(size, cap_style=3)
    return geo_df


def shp_of_polygons_to_binned_dict(
    *, bins: tuple[Bin], shp: object = None, shp_file_name: str = None
) -> dict[int, dict[sp.Polygon]]:
    """
    Use either shp instance or shp_file_name. Creates a dictionary of keys related to
    bin.enum and nested dictionary values containing polygons that fall within each bin.
    """
    if shp_file_name:
        shp = open_shp_with_geopandas(file_name=shp_file_name)
    polygon_dict = {}
    for bin in bins:
        polygon_dict[bin.enum] = shp[shp["DN"] == bin.enum]["geometry"].to_dict()
    logging.info("Converted shapefile to binned polygon dictionary")
    return polygon_dict


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
    assert ".tif" not in input_tif
    if output_shp:
        assert ".shp" not in output_shp
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
    )
    logging.info("Polygonized " + str(input_tif_path) + " to " + str(output_shp_path))


### KML creation


def _save_kml(kml: simplekml.Kml, file_name: str):
    file_path = str(KML_PATH / (file_name + ".kml"))
    kml.save(file_path)
    logging.info("Saved kml file to " + file_path)


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


def make_kml_from_geo_df_multi_bin(
    file_name: str,
    bins: tuple[Bin],
    geo_df: gp.GeoDataFrame = None,
    grouping_col: str = None,
) -> None:
    """
    Takes all polygons in a geo_df and writes them into a kml file
    """
    kml = simplekml.Kml()
    for bin in bins:
        multipolodd = kml.newmultigeometry(name=bin.description)
        for polygon in geo_df[geo_df[grouping_col] == bin.enum]["buffer"]:
            pol = multipolodd.newpolygon(
                name="polygon",
                outerboundaryis=list(polygon.exterior.coords),
            )
            pol.style.polystyle.color = bin.colour
            pol.style.polystyle.outline = 0
    _save_kml(kml=kml, file_name=file_name)


def make_kml_from_one_multipolygon(
    file_name: str, multi_poly: sp.MultiPolygon, bin: Bin
) -> None:
    assert ".kml" not in file_name
    kml = simplekml.Kml()
    multipolodd = kml.newmultigeometry(name="MultiPoly")
    for polygon in multi_poly.geoms:
        pol = multipolodd.newpolygon(
            name="polygon",
            outerboundaryis=list(polygon.exterior.coords),
        )
        pol.style.polystyle.color = bin.colour
        pol.style.polystyle.outline = 0
    _save_kml(kml=kml, file_name=file_name)


def make_kml_from_binned_multipolygon_dict(
    *,
    binned_multipolygon_dict: dict[str, sp.MultiPolygon],
    file_name: str,
    bins: tuple[Bin],
    ignore_bin: list[int] = None,
    one_file_per_bin: bool = False,
) -> None:
    """
    Use in conjunction with unite_polygons_to_binned_multipolygon_dict()
    """
    kml = simplekml.Kml()
    multipolodd = kml.newmultigeometry(name="MultiPoly")
    for bin in bins:
        if bin.enum in ignore_bin:
            continue
        for polygon in binned_multipolygon_dict[bin.enum].geoms:
            pol = multipolodd.newpolygon(
                name="polygon",
                outerboundaryis=list(polygon.exterior.coords),
            )
            pol.style.polystyle.color = bin.colour
            pol.style.polystyle.outline = 0
        if one_file_per_bin:
            _save_kml(kml=kml, file_name=file_name + "_bin_" + str(bin.enum))
    if not one_file_per_bin:
        _save_kml(kml=kml, file_name=file_name)


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
    width: int = 0,
    height: int = 0,
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
        width --- width of the output raster in pixel
        height --- height of the output raster in pixel
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

    # Not implemented settings include: creationOptions, layers
    new_options = []
    if output_format is not None:
        new_options += ["-of", output_format]
    if output_type is not None:
        new_options += ["-ot", output_type]
    if width != 0 or height != 0:
        new_options += ["-outsize", str(width), str(height)]
    if outputBounds is not None:
        new_options += ["-txe"] + _get_output_bounds()
    # Maybe include outputSRS later?
    # if outputSRS is not None:
    #     new_options += ['-a_srs', str(outputSRS)]
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


def run_interpolation_jupyter(
    *,
    # For gdal_grid
    input_shp_name: str,
    target_column: str,
    output_tif_name: str,
    # Below are associated with GdalOptions
    output_format: str = "Gtiff",
    output_type="Byte",
    width: int = 0,
    height: int = 0,
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
        width --- width of the output raster in pixel
        height --- height of the output raster in pixel
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

    # Not implemented settings include: creationOptions, layers
    new_options = []
    if output_format is not None:
        new_options += ["-of", output_format]
    if output_type is not None:
        new_options += ["-ot", output_type]
    if width != 0 or height != 0:
        new_options += ["-outsize", str(width), str(height)]
    if outputBounds is not None:
        new_options += ["-txe"] + _get_output_bounds()
    # Maybe include outputSRS later?
    # if outputSRS is not None:
    #     new_options += ['-a_srs', str(outputSRS)]
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
