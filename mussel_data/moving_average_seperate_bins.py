from mussel_data.helpers import *

use_existing_shp = True
main_name = "only_bm_seg"
bm_dens_col_name = "bm_dens"
bm_dens_bins = (
    Bin(bm_dens_col_name, 1, "First bin", 0, 20, simplekml.Color.orange),
    Bin(bm_dens_col_name, 2, "Second bin", 20, 40, simplekml.Color.orange),
    Bin(bm_dens_col_name, 3, "Third bin", 40, 60, simplekml.Color.red, "[]"),
)

if not use_existing_shp:
    df = open_csv_as_dataframe(file_name=main_name)
    dataframe_to_shp(input_df=df, output_file_name=main_name)

geo_df = open_shp_with_geopandas(file_name=main_name)

print(geo_df.head())

