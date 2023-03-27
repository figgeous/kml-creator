from helpers import *

create_new_shp = False
delete_files_when_done = False
main_name = "only_bm_seg"
target_column = "bm_dens"

bins = (
    Bin(1, "bm_dens", "Blue mussels 0-20", 0, 20, None, True),
    Bin(2, "bm_dens", "Blue mussels 20+", 20, 40, simplekml.Color.hex("5CFF21")),
    Bin(3, "bm_dens", "Blue mussels 40+", 40, 60, simplekml.Color.hex("3B9C17"), "[]"),
)

if create_new_shp or not os.path.exists(str(CSV_PATH / (main_name + ".csv"))):
    df = open_csv_as_dataframe(file_name=main_name)
    dataframe_to_shp(input_df=df, output_file_name=main_name)

geo_df = open_shp_with_geopandas(file_name=main_name)
width, height = get_geodf_dimensions(geo_df=geo_df)

# Create a new shapefile for each bin
for bin in bins:
    print("Creating shapefile for bin: " + str(bin.enum))
    temp_df = None
    if bin.boundary_type == "[[":
        temp_df = geo_df[
            (geo_df["bm_dens"] >= bin.lower) & (geo_df["bm_dens"] < bin.upper)
        ]
    elif bin.boundary_type == "[]":
        temp_df = geo_df[
            (geo_df["bm_dens"] >= bin.lower) & (geo_df["bm_dens"] <= bin.upper)
        ]
    else:
        raise NotImplementedError("Boundary type not implemented: " + bin.boundary_type)
    bin_file_name = "{}-{}-bin_{}".format(main_name, bin.column, str(bin.enum))
    save_geodataframe_to_shp(geo_df=temp_df, file_name=bin_file_name)
    bin.bin_shp_file_name = bin_file_name

# Run the interpolation for each bin
print("Running interpolation for each bin")
radius1_metres = 60
radius2_metres = 10
pixel_size = 10  # in metres
radius1_degrees, radies2_degrees = coordinate_difference_in_metres_to_degrees_x_and_y(
    geo_df=geo_df,
    lon_metres=radius1_metres,
    lat_metres=radius2_metres,
)
for bin in bins:
    print("Running interpolation for bin: " + str(bin.enum))
    tif_file_name = run_interpolation(
        input_shp_name=bin.bin_shp_file_name,
        target_column=target_column,
        output_tif_name=bin.bin_shp_file_name,
        algorithm="average",
        radius1=radius1_degrees,
        radius2=radies2_degrees,
        width=int(width / pixel_size),
        height=int(height / pixel_size),
    )
    bin.tif_file_name = tif_file_name

# Run polygonize for each bin
for bin in bins:
    print("Running polygonize for bin: " + str(bin.enum))
    output_shp_name = bin.tif_file_name + "_polygons"
    run_polygonize(
        input_tif=bin.tif_file_name,
        output_shp=output_shp_name,
        # mask='none',
        # options=["-mask", tif_name]
    )
    bin.polygon_shp_file_name = output_shp_name

## Unify and simplify polygons for each bin and make kml
for bin in bins:
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


# remove the temporary files
if delete_files_when_done:
    for bin in bins:
        print("Removing temporary files for bin: " + str(bin.enum))
        shp_file_path = str(SHP_PATH / bin.bin_shp_file_name) + ".shp.zip"
        tif_file_path = str(TIF_PATH / bin.tif_file_name) + ".tif"
        polygon_shp_file_path = str(SHP_PATH / bin.polygon_shp_file_name) + ".shp.zip"

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
