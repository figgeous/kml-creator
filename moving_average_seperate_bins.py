# from scripts.scripts import *
# # open text.txt and print first line
# with open("test.txt", "r") as f:
#     print(f.readline())
# use pandas to open only_bm_seg.csv
# df = pd.read_csv("csv_files/only_bm_seg.csv")
# print(df.head())
# # use geopandas to open only_bm_seg.shp
# import geopandas as gp
# geo_df = gp.read_file("mussel_data/shapefiles/only_bm.shp.zip")
# print(geo_df.head())
# print current working directory
# main_name = "only_bm_seg"
# open_csv_as_dataframe(file_name="main_name")
# bm_dens_col_name = "bm_dens"
# bm_dens_bins = (
#     Bin(bm_dens_col_name, 1,"First bin", 0, 20, simplekml.Color.orange),
#     Bin(bm_dens_col_name, 2,"Second bin", 20, 40, simplekml.Color.orange),
#     Bin(bm_dens_col_name, 3,"Third bin", 40,60, simplekml.Color.red, "[]"),
#     )
# file_name = main_name
# file_path = "/"+SHP_PATH + file_name + ".shp.zip"
# try:
#     geo_df = gp.read_file(file_path)
#     print(f"Opened shapefile from {file_path}. Shape: {geo_df.shape}")
# except Exception as e:
#     print("Can't open shapefile.", e)
# geo_df = open_shp_with_geopandas(file_name=main_name)
# width, height = get_geodf_dimensions(geo_df=geo_df)
# # Create dictionary of geo_df in bins
# geo_df_dict = {}
# for bin in bm_dens_bins:
#     if bin.bounds == "[]":
#         geo_df_dict[bin.enum] = geo_df[(geo_df[bm_dens_col_name] >= bin.lower_bound) & (geo_df[bm_dens_col_name] <= bin.upper_bound)]
#     else:
#         geo_df_dict[bin.enum] = geo_df[(geo_df[bm_dens_col_name] >= bin.lower_bound) & (geo_df[bm_dens_col_name] < bin.upper_bound)]
# # print firt bin
# print(geo_df_dict[1])
