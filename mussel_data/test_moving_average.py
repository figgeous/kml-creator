from helpers import *

create_new_shp = False
# delete_files_when_done = False
dataset_name = "only_bm_seg"

bins = [
    Bin(1, "bm_dens", "Blue mussel density 0-20", 0, 20, None, True),
    Bin(
        2, "bm_dens", "Blue mussels density 20+", 20, 40, simplekml.Color.hex("5CFF21")
    ),
    Bin(
        3,
        "bm_dens",
        "Blue mussels density 40+",
        40,
        60,
        simplekml.Color.hex("3B9C17"),
        False,
        "[]",
    ),
    Bin(4, "bm_size", "Blue mussels size 0-2 cm", 0, 50, None, True),
    Bin(
        5, "bm_size", "Blue mussels size 5-6 cm", 50, 60, simplekml.Color.hex("ffbf80")
    ),
    Bin(
        6,
        "bm_size",
        "Blue mussels size 6-9 cm",
        60,
        90,
        simplekml.Color.hex("ffa64d"),
        False,
        "[]",
    ),
]

# Create the shapefile if it doesn't exist or if create_new_shp is True
if create_new_shp or not os.path.exists(str(CSV_PATH / (dataset_name + ".csv"))):
    df = open_csv_as_dataframe(file_name=dataset_name)
    dataframe_to_shp(input_df=df, output_file_name=dataset_name)

geo_df = open_shp_with_geopandas(file_name=dataset_name)
runner = Runner(bins=bins, geo_df=geo_df, dataset_name=dataset_name)
runner.create_shp_for_each_bin()
runner.run_interpolation_for_each_bin()
runner.run_polygonize_for_each_bin()
runner.create_kml_for_each_bin()
runner.delete_files()
# runner.run_all()
