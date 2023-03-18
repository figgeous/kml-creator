from dataclasses import dataclass
from typing import Any

import geopandas as gp
import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely as sp
import simplekml
from osgeo import gdal
from osgeo import gdal_array
from osgeo import ogr
from shapely.geometry import Point

from constants import *


@dataclass
class Bin:
    column: str
    enum: int
    description: str
    lower_bound: int
    upper_bound: int
    colour: str = None


### For opening files
def open_csv_as_dataframe(
    *, file_name: str, sep: str = ",", index_col: Any = False
) -> pd.DataFrame:
    assert ".csv" not in file_name
    file_path = CSV_PATH + file_name + ".csv"
    df = pd.read_csv(file_path, sep=sep, index_col=index_col)
    print(f"Opened csv from {file_path}. Shape: {df.shape}")
    return df

def open_shp_with_gdal(*, file_name: str) -> ogr.DataSource:
    assert ".shp" not in file_name
    file_path = SHP_PATH + file_name + ".shp.zip"
    ogr_datasource = ogr.Open(file_path)
    assert ogr_datasource
    print(f"Opened shapefile from {file_path}")
    return ogr_datasource

def open_shp_with_geopandas(*, file_name:str) -> gp.GeoDataFrame:
    assert ".shp" not in file_name
    file_path = SHP_PATH+file_name+'.shp.zip'
    geo_df = gp.read_file(file_path)
    print(f"Opened shapefile from {file_path}. Shape: {geo_df.shape}")
    return geo_df

def open_tif_with_gdal(*, file_name: str) -> gdal.Dataset:
    assert ".tif" not in file_name
    file_path = TIF_PATH + file_name + ".tif"
    gdal_dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
    assert gdal_dataset
    print(f"Opened tif file from {file_path}. Run plot_raster() to view")
    return gdal_dataset


### For saving
def save_dataframe_to_csv(
    *, df: pd.DataFrame, file_name: str, sep: str = ",", index_col: str = None
) -> pd.DataFrame:
    assert ".csv" not in file_name
    file_path = CSV_PATH + file_name + ".csv"
    return df.to_csv(path_or_buf=file_path, sep=sep)

def save_geodataframe_to_shp(geo_df:gp.GeoDataFrame, file_name:str) -> None:
    assert ".csv" not in file_name
    file_path = SHP_PATH+file_name+'.shp.zip'
    geo_df.to_file(filename=file_path, driver='ESRI Shapefile', crs='EPSG:4326')
    print(f"Saved geo_df to {file_path}")


### Display functions
def get_distance(
        coords: tuple[int, int, int, int]) -> float:
    """
    Calculates the number of metres between coordinate points.
    Must be (lat_1, lon_1, lat_2, lon_2)
    """
    distance = geopy.distance.geodesic((coords[0], coords[1]), (coords[2], coords[3])).m
    return distance


def print_tif_metadata(*, tif_name: str) -> None:
    metadata = os.popen("gdalinfo " + TIF_PATH + tif_name + ".tif").read()
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
        fig_size: tuple[int, int] = None,
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
    if fig_size:
        plt.figure(figsize=(fig_size, fig_size[0], fig_size[1]))
    plt.imshow(image[:, :, 0], origin="lower")
    plt.colorbar()

def get_geodf_dimensions(geo_df:gp.GeoDataFrame) -> (int, int):
    min_lon, min_lat, max_lon, max_lat = geo_df.total_bounds[0], geo_df.total_bounds[1], geo_df.total_bounds[2], geo_df.total_bounds[3]
    width = get_distance((min_lat, min_lon, min_lat, max_lon))
    height = get_distance((min_lat, min_lon, max_lat, min_lon))
    print(f"geo_df is {round(width)} m in width and {round(height)} m in height")
    return width, height

def rough_coord_diff_to_metres(coord_diff:float):
    return coord_diff*1000000


### Basic data processing
def grid_to_coordinate(*, grid_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Converts a grid format to a list of values and their coordinates. The grid is
    2 dimensional with latitude on rows and longitude on columns. Lists of items and
    their coordinates are easier to process.
    """
    melt = grid_df.melt(
        id_vars=["lat"], var_name=grid_df.columns[1], value_name=target_column
    )
    melt = melt.rename(columns={melt.columns[1]: "lon"})
    melt[["lat"]] = melt[["lat"]].apply(pd.to_numeric)
    return melt


def dataframe_to_shp(*, input_df, output_file_path: str = None) -> gp.GeoDataFrame:
    """
    Adds a column of shapely.geometry.point.Point objects corresponding to the lat and
    lon of each row. A geopandas.GeoDataFrame is a subclass of pandas.DataFrame.
    """
    # combine lat and lon column to a shapely Point() object
    input_df["geometry"] = input_df.apply(
        lambda x: Point((float(x.lon), float(x.lat))), axis=1
    )
    geo_df = gp.GeoDataFrame(input_df, geometry="geometry")
    geo_df = geo_df.reset_index()
    geo_df = geo_df.drop(columns=['index',"lat", "lon"])
    if output_file_path:
        assert ".shp.zip" not in output_file_path
        geo_df.to_file(SHP_PATH + output_file_path + ".shp.zip", driver="ESRI Shapefile")
    return geo_df


def add_bin_enum_to_df(df:pd.DataFrame, bins:tuple[Bin], column:str) -> pd.DataFrame:
    """
    Adds a column to df with a binned version of column. Each row in df that falls
    within bin.lower_bound and bin.upper_bound is given the value bin.enum. The new
    column is the original column +"_b"
    """
    conditions, categories = [], []
    for bin in bins:
        conditions.append((df[column] >= bin.lower_bound) & (df[column] < bin.upper_bound))
        categories.append(bin.enum)
    df[column+"_s"] = np.select(conditions, categories)
    return df


def add_square_buffer_to_geo_df(geo_df:gp.GeoDataFrame, size:float=0.00005):
    """
    Adds a column of square polygons with centre at Point. Size in degrees
    """
    # Buffer the points using a square cap style
    #https://gis.stackexchange.com/questions/314949/creating-square-buffers-around-points-using-shapely
    geo_df["buffer"] = geo_df["geometry"].buffer(size, cap_style = 3)
    return geo_df


def shp_of_polygons_to_binned_dict(*, bins:tuple[Bin], shp:object=None, shp_file_name:str=None) -> dict[int, dict[sp.Polygon]]:
    """
    Use either shp instance or shp_file_name. Creates a dictionary of keys related to
    bin.enum and nested dictionary values containing polygons that fall within each bin.
    """
    if shp_file_name:
        shp = open_shp_with_geopandas(file_name=shp_file_name)
    polygon_dict = {}
    for bin in bins:
        polygon_dict[bin.enum] = shp[shp['DN'] == bin.enum]["geometry"].to_dict()
    print("Converted shapefile to binned polygon dictionary")
    return polygon_dict


def unite_polygons_to_binned_multipolygon_dict(*, geo_df:gp.GeoDataFrame, grouping_col:str, bins:tuple[Bin]) -> dict[str,sp.MultiPolygon]:
    """
    Takes a geo_df containing polygons, unites any overlapping polygons and groups the
    resulting multipolygons into a binned dict.
    """
    multipolygon_dict = dict()
    for bin in bins:
        column = geo_df[geo_df[grouping_col] == bin.enum]["buffer"]
        multipolygon_dict[bin.enum] = sp.unary_union(geo_df[geo_df[grouping_col] == bin.enum]["buffer"])
        print("Bin {}: uniting {} polygons to {} polygons".format(bin.enum, len(column), len(multipolygon_dict[bin.enum].geoms)))
    return multipolygon_dict


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

    input_tif_path = TIF_PATH + input_tif + ".tif" if input_tif else None

    if output_shp:
        output_shp_path = SHP_PATH + output_shp + ".shp.zip"
    else:
        output_shp_path = SHP_PATH + input_tif + ".shp.zip"

    gdal_polygonize(
        src_filename=input_tif_path,
        dst_filename=output_shp_path,
        band_number=input_band,
        dst_layername=output_layer,
        dst_fieldname=output_field,
        mask=mask,
        connectedness8=connectedness8,
        options=options,
    )


### KML creation

def _save_kml(kml:simplekml.Kml, file_name:str):
    file_path = KML_PATH+file_name+".kml"
    kml.save(file_path)
    print("Saved kml file to "+file_path)

def make_kml_from_geo_df(
        file_name:str,
        bins:tuple[Bin],
        geo_df:gp.GeoDataFrame = None,
        grouping_col:str = None,
        ) -> None:
    """
    Takes all polygons in a geo_df and writes them into a kml file
    """
    assert ".kml" not in file_name

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

def make_kml_from_one_multipolygon(file_name:str, multi_poly:sp.MultiPolygon, bin:Bin) -> None:
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
    _save_kml(kml=kml,file_name=file_name)


def make_kml_from_binned_multipolygon_dict(
        *,
        binned_multipolygon_dict: dict[str,sp.MultiPolygon],
        file_name:str,
        bins:tuple[Bin],
        ignore_bin:int = None,
        one_file_per_bin:bool=False) -> None:
    """
    Use in conjunction with unite_polygons_to_binned_multipolygon_dict()
    """
    assert ".kml" not in file_name
    kml = simplekml.Kml()
    multipolodd = kml.newmultigeometry(name="MultiPoly")
    for bin in bins:
        if bin.enum == ignore_bin: continue
        for polygon in binned_multipolygon_dict[bin.enum].geoms:
            pol = multipolodd.newpolygon(
                name="polygon",
                outerboundaryis=list(polygon.exterior.coords),
            )
            pol.style.polystyle.color = bin.colour
            pol.style.polystyle.outline = 0
        if one_file_per_bin:
            _save_kml(kml=kml, file_name=file_name+"_bin_"+str(bin.enum))
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
    output_res: list = None,
    outputBounds: list = None,
    algorithm: str = "invdist",
    power: int = 1,
    smoothing: float = None,
    radius1: float = None,
    radius2: float = None,
    angle: int = None,
    max_points: int = 0,
    min_points: int = 0,
    max_points_per_quadrant: int = 0,
    min_points_per_quadrant: int = 0,
    nodata: float = None,
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

    # Not implemented settings include: creationOptions, layers, SQLStatement, z_increase, z_multiply
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
    # Maybe include spatFilter later?
    # if spatFilter is not None:
    #     new_options += ['-spat', str(spatFilter[0]), str(spatFilter[1]), str(spatFilter[2]), str(spatFilter[3])]
    if output_res is not None:
        new_options += ["-tr", str(output_res[0]), str(output_res[1])]
    print("Options: ", new_options)

    grid_options = gdal.GridOptions(options=new_options)

    output_tif_name += f"-{algorithm}-{radius1}-{radius2}"
    dest_name = TIF_PATH + output_tif_name + ".tif"
    src_ds = SHP_PATH + input_shp_name + ".shp.zip"

    print("Running interpolation on: " + src_ds)
    print("Saving to: " + dest_name)
    idw = gdal.Grid(destName=dest_name, srcDS=src_ds, options=grid_options)
    idw = None
    return output_tif_name

