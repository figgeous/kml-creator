# new_main_name_full = "only_bm_seg-bm_dens--average-986.6431"
import numpy as np
from osgeo import gdalconst


def apply_bins_to_tif(
    file_name: str, bins: tuple[Bin], file_name_suffix: str = "_binned"
) -> str:
    in_dataset = open_tif_with_gdal(file_name=file_name)
    if in_dataset is None:
        print("Could not open image file")

    in_band = in_dataset.GetRasterBand(1)
    rows = in_dataset.RasterYSize
    cols = in_dataset.RasterXSize

    # create the output image
    driver = in_dataset.GetDriver()
    in_band_as_array = in_band.ReadAsArray()

    # plot distribution of values in array
    plt.hist(in_band_as_array.flatten(), bins=100)

    for bin in bins:
        print(bin.upper_bound_norm, bin.lower_bound_norm)
        if bin.boundary_type == "[]":
            cond1 = in_band_as_array <= bin.upper_bound_norm
        else:
            cond1 = in_band_as_array < bin.upper_bound_norm
        cond2 = in_band_as_array >= bin.lower_bound_norm
        in_band_as_array[cond1 & cond2] = bin.enum
        # get max value in all arrays

    # Create out dataset
    new_file_name = file_name + file_name_suffix
    new_file_path = TIF_PATH + new_file_name + ".tif"
    out_dataset = driver.Create(new_file_path, cols, rows, 1, gdalconst.GDT_Byte)
    if out_dataset is None:
        print("Could not create output tif file")
    out_band = out_dataset.GetRasterBand(1)
    out_band_as_array = in_band_as_array
    out_band.WriteArray(out_band_as_array, 0, 0)

    # flush data to disk, set the NoData value and calculate stats
    out_band.FlushCache()
    print(f"Saved tif to: {TIF_PATH+file_name+file_name_suffix+'.tif'}")
    # out_band.SetNoDataValue(-99)

    # georeference the image and set the projection
    out_dataset.SetGeoTransform(in_dataset.GetGeoTransform())
    out_dataset.SetProjection(in_dataset.GetProjection())

    del out_dataset
    return new_file_name


new_main_name_full_binned = apply_bins_to_tif(
    file_name=new_main_name_full, bins=bm_dens_bins
)


def print_nth_line_of_raster(file_name: str, line_number: int = 0):
    in_dataset = open_tif_with_gdal(file_name=file_name)
    if in_dataset is None:
        print("Could not open image file")

    in_band = in_dataset.GetRasterBand(1)

    # out_data = np.zeros((rows,cols), np.int16)

    in_band_as_array = in_band.ReadAsArray()
    print(in_band_as_array[line_number])


line_number = 345
print_nth_line_of_raster(file_name=new_main_name_full, line_number=line_number)

print_nth_line_of_raster(
    file_name=new_main_name_full + "_binned", line_number=line_number
)


def unite_polygons_to_binned_multipolygon_dict(
    *,
    bins: tuple[Bin],
    geo_df: gp.GeoDataFrame = None,
    polygon_dict=None,
    grouping_col: str = None,
    target_column: str = None,
) -> dict[str, sp.MultiPolygon]:
    """
    Takes a geo_df containing polygons, unites any overlapping polygons and groups the
    resulting multipolygons into a binned dict.
    """
    multipolygon_dict = dict()
    for bin in bins:
        multipolygon_dict[bin.enum] = sp.unary_union(
            geo_df[geo_df[grouping_col] == bin.enum][target_column]
        )
        print(
            "Bin {}: uniting {} polygons to {} polygons".format(
                bin.enum,
                len(geo_df[geo_df[grouping_col] == bin.enum][target_column]),
                len(multipolygon_dict[bin.enum].geoms),
            )
        )
        # # If geo_df as input
        # column = geo_df[geo_df[grouping_col] == bin.enum][target_column]
        # multipolygon_dict[bin.enum] = sp.unary_union(
        #     geo_df[geo_df[grouping_col] == bin.enum][target_column]
        # )
        # print(
        #     "Bin {}: uniting {} polygons to {} polygons".format(
        #         bin.enum, len(column), len(multipolygon_dict[bin.enum].geoms)
        #     )
        # )
    return multipolygon_dict


def open_shp_with_gdal(*, file_name: str):
    from osgeo import ogr

    file_path = SHP_PATH + file_name + ".shp.zip"
    ogr_datasource = ogr.Open(file_path)
    assert ogr_datasource
    logging.info(f"Opened shapefile from {file_path}")
    return ogr_datasource


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
    dataset_width, dataset_height = get_geodf_dimensions(geo_df=geo_df)
    return geo_df, dataset_width, dataset_height


def save_dataframe_to_csv(
    *, df: pd.DataFrame, file_name: str, sep: str = ",", index_col: str = None
) -> pd.DataFrame:
    file_path = CSV_PATH + file_name + ".csv"
    return df.to_csv(path_or_buf=file_path, sep=sep)


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


def add_bin_enum_to_df(df: pd.DataFrame, bins: tuple[Bin], column: str) -> pd.DataFrame:
    """
    Adds a column to df with a binned version of column. Each row in df that falls
    within bin.lower_bound and bin.upper_bound is given the value bin.enum. The new
    column is the original column +"_b"
    """
    import numpy as np

    conditions, categories = [], []
    for bin in bins:
        conditions.append(
            (df[column] >= bin.lower_bound) & (df[column] < bin.upper_bound)
        )
        categories.append(bin.enum)
    df[column + "_s"] = np.select(conditions, categories)
    return df


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
            file_path = str(KML_PATH / (file_name + ".kml"))
            kml.save(file_path)
            logging.info("Saved kml file to " + file_path)
    if not one_file_per_bin:
        file_path = str(KML_PATH / (file_name + ".kml"))
        kml.save(file_path)
        logging.info("Saved kml file to " + file_path)


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
    file_path = str(KML_PATH / (file_name + ".kml"))
    kml.save(file_path)
    logging.info("Saved kml file to " + file_path)
