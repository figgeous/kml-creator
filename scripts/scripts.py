import pandas as pd
import matplotlib.pyplot as plt

from constants import *

def grid_to_coordinate(*, grid_df:pd.DataFrame, target_column:str) -> pd.DataFrame:
    melt = grid_df.melt(id_vars=["lat"], var_name=grid_df.columns[1], value_name=target_column)
    melt = melt.rename(columns={melt.columns[1]:"lon"})
    melt[["lat"]] = melt[["lat"]].apply(pd.to_numeric)
    return melt

def dataframe_to_shp(*, input_df, output_file_path:str=None):
    import geopandas
    from shapely.geometry import Point
    # combine lat and lon column to a shapely Point() object
    input_df['geometry'] = input_df.apply(lambda x: Point((float(x.lon), float(x.lat))), axis=1)
    gpdf = geopandas.GeoDataFrame(input_df, geometry='geometry')
    if output_file_path:
        assert ".shp.zip" not in output_file_path
        gpdf.to_file(SHP_PATH+output_file_path+'.shp.zip', driver='ESRI Shapefile')
    return gpdf

def open_shp(*,file_name:str):
    from osgeo import ogr
    assert ".shp.zip" not in file_name
    file = ogr.Open(SHP_PATH+file_name+".shp.zip")
    assert file
    return file

def get_pandas_from_csv(*,csv_name:str,sep:str=",",index_col:str=None) -> pd.DataFrame:
    assert ".csv" not in csv_name
    return pd.read_csv(CSV_PATH+csv_name+".csv",sep=sep,index_col=index_col)

def pandas_to_csv(*,df:pd.DataFrame, csv_name:str, sep:str=",", index_col:str=None) -> pd.DataFrame:
    assert ".csv" not in csv_name
    return df.to_csv(CSV_PATH+csv_name+".csv",sep=sep)