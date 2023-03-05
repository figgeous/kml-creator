from osgeo import gdal
from osgeo_utils.rgb2pct import *

import os

### Metadata
# # metadata = os.popen('gdalinfo ./output.tif').read()
# metadata = os.popen('gdalinfo ./ours.tiff').read()
# print(metadata)
# metadata = None

### Read shapefile
from osgeo import ogr

file = ogr.Open("only_bm.shp")

shape = file.GetLayer(0)
#first feature of the shapefile
feature = shape.GetFeature(0)
first = feature.ExportToJson()
print(first)
# print(first) # (GeoJSON format)
# # {"geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [25.0, 10.0], [50.0, 50.0]]}, "type": "Feature", "properties": {"FID": 0.0}, "id": 0}







# # Reduce colour palette down to 3 colours
# input = "output.tif"
# number_of_colour = 3
# output = input+"_pct_"+str(number_of_colour)
# pct = rgb2pct(input, None, output, 3)

# # Run Grid again.
# ds = gdal.Open('sample_pct_3.tif')
# idw = gdal.Grid("sample_pct_3_grid.tif", ds, format="GTiff", zfield="Blue musse", algorithm="invdist:power=3:smoothing=0.0015:radius=1")
# idw = None

# # Translate Type=... to Type = Byte
# ds = gdal.Open('ours.tiff')
# ds = gdal.Translate('output.tif', ds, options = "-ot Byte -expand gray")


