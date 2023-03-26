import simplekml

from mussel_data.helpers import *


main_name = "only_bm"
bm_dens_col_name = "bm_dens"
open_csv_as_dataframe(file_name=main_name)
bm_dens_bins = (
    Bin(bm_dens_col_name, 1, "First bin", 0, 20, simplekml.Color.orange),
    Bin(bm_dens_col_name, 2, "Second bin", 20, 40, simplekml.Color.orange),
    Bin(bm_dens_col_name, 3, "Third bin", 40, 60, simplekml.Color.red, "[]"),
)

dataframe_to_shp

# geo_df = open_shp_with_geopandas(file_name=main_name)
# width, height = get_geodf_dimensions(geo_df=geo_df)
# print(width, height)
# # Create dictionary of geo_df in bins
# geo_df_dict = {}
# for bin in bm_dens_bins:
#     if bin.bounds == "[]":
#         geo_df_dict[bin.enum] = geo_df[(geo_df[bm_dens_col_name] >= bin.lower_bound) & (geo_df[bm_dens_col_name] <= bin.upper_bound)]
#     else:
#         geo_df_dict[bin.enum] = geo_df[(geo_df[bm_dens_col_name] >= bin.lower_bound) & (geo_df[bm_dens_col_name] < bin.upper_bound)]
# # print firt bin
# print(geo_df_dict[1])
