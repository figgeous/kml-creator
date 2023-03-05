from osgeo import gdalconst,ogr,gdal,gdal_array
import os
from constants import *

def gdal_open_tif(*,tif_name:str):
    return gdal.Open(TIF_PATH+tif_name+".tif", gdal.GA_ReadOnly)

def gdal_print_metadata(*,tif_name:str) -> None:
    metadata = os.popen("gdalinfo "+TIF_PATH+tif_name+".tif").read()
    print(metadata)

def gdal_print_first_feature(file):
    """
    Should print out something like:
    {"type": "Feature", "geometry": {"type": "Point", "coordinates": [8.943549, 56.996942]}, "properties": {"bm_dens": 10, "bm_size": 58}, "id": 0}
    """
    shape = file.GetLayer(0)
    #first feature of the shapefile
    feature = shape.GetFeature(1)
    first = feature.ExportToJson()
    print(first) # (GeoJSON format)
    
def gdal_open_shp(*,shp_name:str):
    assert ".shp.zip" not in shp_name
    return ogr.Open(SHP_PATH+shp_name+".shp.zip")

def gdal_run_invdist(
    *,
    input_shp_name:str,
    target_column:str,
    output_tif_name:str,
    output_format:str="Gtiff",
    output_type=gdalconst.GDT_Int16,
    power:int=1,
    smoothing:float=None,
    radius1:float=None,
    radius2:float=None,
    angle:int=None,
    max_points:int=0,
    min_points:int=0,
    nodata:float=None,
    ):
    assert ".shp.zip" not in input_shp_name
    assert ".tif" not in output_tif_name
    
    algorithm_str = f"invdist:power={power}"
    algorithm_str += f":smoothing={smoothing}" if smoothing else ""
    algorithm_str += f":radius1={radius1}" if radius1 else ""
    algorithm_str += f":radius2={radius2}" if radius2 else ""
    algorithm_str += f":angle={angle}" if angle else ""
    algorithm_str += f":max_points={max_points}" if max_points else ""
    algorithm_str += f":min_points={min_points}" if min_points else ""
    algorithm_str += f":nodata={nodata}" if nodata else ""
    
    grid_options = gdal.GridOptions(
        format=output_format,
        zfield=target_column,
        algorithm=algorithm_str,
        outputType=output_type,)
    dest_name = TIF_PATH+output_tif_name+".tif"
    src_ds = SHP_PATH+input_shp_name+".shp.zip"
    print("Running interpolation on: "+src_ds)
    print("Saving to: "+dest_name)
    idw = gdal.Grid(
        destName=dest_name,
        srcDS=src_ds,
        options=grid_options)
    idw = None
    
def plot_raster(*,tif_name:str) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    dataset = gdal_open_tif(tif_name=main_name)
    # Allocate our array using the first band's datatype
    image_datatype = dataset.GetRasterBand(1).DataType

    image = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount),
                    dtype=gdal_array.GDALTypeCodeToNumericTypeCode(image_datatype))
    # Loop over all bands in dataset
    for b in range(dataset.RasterCount):
        # Remember, GDAL index is on 1, but Python is on 0 -- so we add 1 for our GDAL calls
        band = dataset.GetRasterBand(b + 1)
        print(band)
        # Read in the band's data into the third dimension of our array
        image[:, :, b] = band.ReadAsArray()

    plt.imshow(image[:, :, 0],)
    plt.colorbar()