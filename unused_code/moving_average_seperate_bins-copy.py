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
pixel_size = 10  # in metres
radius1_degrees, radies2_degrees = coordinate_difference_in_metres_to_degrees_x_and_y(
    geo_df=geo_df,
    lon_metres=radius1_metres,
    lat_metres=radius1_metres,
)
target_column = bm_dens_col_name

# Run the interpolation for each bin
for bin in bm_dens_bins:
    if bin.boundary_type == "[]":
        where_string = (
            f"{target_column} >= {bin.lower} and {target_column} <= {bin.upper}"
        )
    elif bin.boundary_type == "[[":
        where_string = (
            f"{target_column} >= {bin.lower} and {target_column} < {bin.upper}"
        )
    else:
        raise NotImplementedError(f"Boundary type not implemented: {bin.boundary_type}")
    output_tif_name = "Trial-bin_" + str(bin.enum)
    print(main_name, target_column, where_string)
    bin_name_full = run_interpolation(
        input_shp_name=main_name,
        target_column=target_column,
        output_tif_name=output_tif_name,
        algorithm="average",
        radius1=radius1_degrees,
        radius2=radies2_degrees,
        width=int(width / pixel_size),
        height=int(height / pixel_size),
        where=where_string,
    )
    bin.tif_file_name = bin_name_full


for bin in bm_dens_bins:
    tif_name = bin.tif_file_name
    output_shp_name = bin.tif_file_name + "_polygons"
    run_polygonize(
        input_tif=tif_name,
        output_shp=output_shp_name,
        mask="none",
        options=["-mask", tif_name],
    )
    bin.polygon_shp_file_name = output_shp_name

# make kml files
for bin in bm_dens_bins:
    polygon_df = open_shp_with_geopandas(file_name=bin.polygon_shp_file_name)
    polygon_df = polygon_df[polygon_df["DN"] != 0]
    print(polygon_df["DN"].value_counts())
    make_kml_from_geo_df_single_bin(
        polygon_df=polygon_df, kml_file_name=bin.polygon_shp_file_name, bin=bin
    )
