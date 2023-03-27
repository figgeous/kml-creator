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
