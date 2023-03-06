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
    # For gdal_grid
    input_shp_name:str,
    target_column:str,
    output_tif_name:str,
    # Below are associated with GdalOptions
    output_format:str="Gtiff",
    output_type="Byte",
    width:int=0, height:int=0,
    output_res:list = None,
    outputBounds:list = None,
    algorithm:str="invdist",
    power:int=1,
    smoothing:float=None,
    radius1:float=None,
    radius2:float=None,
    angle:int=None,
    max_points:int=0,
    min_points:int=0,
    nodata:float=None,
    ) -> str:
    assert ".shp.zip" not in input_shp_name
    assert ".tif" not in output_tif_name
    
    def _get_algorithm_str() -> str:
        s = f"invdist:power={power}" if power else ""
        s += f":smoothing={smoothing}" if smoothing else ""
        s += f":radius1={radius1}" if radius1 else ""
        s += f":radius2={radius2}" if radius2 else ""
        s += f":angle={angle}" if angle else ""
        s += f":max_points={max_points}" if max_points else ""
        s += f":min_points={min_points}" if min_points else ""
        s += f":nodata={nodata}" if nodata else ""
        return s
    
    # Not implemented settings include: creationOptions, layers, SQLStatement, z_increase, z_multiply
    new_options = []
    if output_format is not None:
        new_options += ['-of', output_format]
    if output_type is not None:
        new_options += ['-ot', output_type]
    if width != 0 or height != 0:
        new_options += ['-outsize', str(width), str(height)]
    if outputBounds is not None:
        new_options += ['-txe','%.18g' % outputBounds[0], '%.18g' % outputBounds[2], '-tye', '%.18g' % outputBounds[1], '%.18g' % outputBounds[3]]
    # Maybe include outputSRS later?
    # if outputSRS is not None:
    #     new_options += ['-a_srs', str(outputSRS)]
    if algorithm is not None:
        new_options += ['-a', _get_algorithm_str()]
    if target_column is not None:
        new_options += ['-zfield', target_column]
    #Maybe include spatFilter later?
    # if spatFilter is not None:
    #     new_options += ['-spat', str(spatFilter[0]), str(spatFilter[1]), str(spatFilter[2]), str(spatFilter[3])]
    if output_res is not None:
        new_options += ['-tr', str(output_res[0]), str(output_res[1])]
    print("Options: ",new_options)
    
    grid_options = gdal.GridOptions(
        options=new_options
    ) 

    output_tif_name += f"-invdist-{power}-{smoothing}-{radius1}-{radius2}-{angle}-{max_points}-{min_points}"
    dest_name = TIF_PATH+output_tif_name+".tif"
    src_ds = SHP_PATH+input_shp_name+".shp.zip"
    
    print("Running interpolation on: "+src_ds)
    print("Saving to: "+dest_name)
    idw = gdal.Grid(
        destName=dest_name,
        srcDS=src_ds,
        options=grid_options)
    idw = None
    return output_tif_name
    
def plot_raster(*,tif_name:str) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    dataset = gdal_open_tif(tif_name=tif_name)
    # Allocate our array using the first band's datatype
    image_datatype = dataset.GetRasterBand(1).DataType

    image = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount),
                    dtype=gdal_array.GDALTypeCodeToNumericTypeCode(image_datatype))
    # Loop over all bands in dataset
    for b in range(dataset.RasterCount):
        # Remember, GDAL index is on 1, but Python is on 0 -- so we add 1 for our GDAL calls
        band = dataset.GetRasterBand(b + 1)
        # Read in the band's data into the third dimension of our array
        image[:, :, b] = band.ReadAsArray()

    plt.imshow(image[:, :, 0], origin='lower')
    plt.colorbar()