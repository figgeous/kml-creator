from mussel_data.helpers import *

create_new_shp = False
main_name = "only_bm_seg"
bm_dens_col_name = "bm_dens"
bm_dens_bins = (
    Bin(bm_dens_col_name, 1, "First bin", 0, 20, simplekml.Color.yellow),
    Bin(bm_dens_col_name, 2, "Second bin", 20, 40, simplekml.Color.orange),
    Bin(bm_dens_col_name, 3, "Third bin", 40, 60, simplekml.Color.red, "[]"),
)

if create_new_shp:
    df = open_csv_as_dataframe(file_name=main_name)
    dataframe_to_shp(input_df=df, output_file_name=main_name)

geo_df = open_shp_with_geopandas(file_name=main_name)
width, height = get_geodf_dimensions(geo_df=geo_df)

radius1_metres = 60
radius2_metres = 10
pixel_size = 10 #in metres
radius1_degrees, radies2_degrees = coordinate_difference_in_metres_to_degrees_x_and_y(
    geo_df=geo_df,
    lon_metres=radius1_metres,
    lat_metres=radius2_metres,)
target_column = bm_dens_col_name
#
# Create a new shapefile for each bin
for bin in bm_dens_bins:
    temp_df = None
    if bin.boundary_type == "[[":
        temp_df = geo_df[(geo_df["bm_dens"] >= bin.lower) & (geo_df["bm_dens"] < bin.upper)]
    elif bin.boundary_type == "[]":
        temp_df = geo_df[(geo_df["bm_dens"] >= bin.lower) & (geo_df["bm_dens"] <= bin.upper)]
    else:
        raise NotImplementedError("Boundary type not implemented: " + bin.boundary_type)
    bin_file_name ="{}-{}-bin_{}".format(main_name, bin.column, str(bin.enum))
    save_geodataframe_to_shp(geo_df=temp_df, file_name=bin_file_name)
    bin.bin_shp_file_name = bin_file_name

# Run the interpolation for each bin
for bin in bm_dens_bins:
    # main_name="only_bm_seg_bin_3"
    output_tif_name = "Trial-bin_"+str(bin.enum)
    print(
        [main_name, target_column, radius1_degrees, radies2_degrees, width / pixel_size,
         height / pixel_size])
    bin_name_full = run_interpolation(
        input_shp_name=bin.bin_shp_file_name,
        target_column=target_column,
        output_tif_name=output_tif_name,
        algorithm="average",
        radius1=radius1_degrees,
        radius2=radies2_degrees,
        width=int(width/pixel_size),
        height=int(height/pixel_size),
    )
    bin.tif_file_name = bin_name_full

# Run polygonize for each bin
for bin in bm_dens_bins:
    tif_name = bin.tif_file_name
    output_shp_name = bin.tif_file_name + "_polygons"
    run_polygonize(
        input_tif=tif_name,
        output_shp=output_shp_name,
        # mask='none',
        # options=["-mask", tif_name]
        )
    bin.polygon_shp_file_name = output_shp_name

# make kml files
for bin in bm_dens_bins:
    polygon_df = open_shp_with_geopandas(file_name=bin.polygon_shp_file_name)
    polygon_df = polygon_df[polygon_df["DN"] != 0]
    kml_file_name = make_kml_from_geo_df_single_bin(
        polygon_df=polygon_df,
        kml_file_name=bin.polygon_shp_file_name,
        bin=bin)
    bin.kml_file_name = kml_file_name

# remove the temporary files
for bin in bm_dens_bins:
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