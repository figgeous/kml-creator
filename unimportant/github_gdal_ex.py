#This file is a copy paste from stackoverflow. We might be able to get some ideas from it.

from osgeo import gdal
import os

shapefiles = [folder + '/' + file for file in os.listdir(folder) if 'shp' in file]
output = 'output'
if not os.path.exists(output):
    os.mkdir(output)
RefImage = ref_image
gdalformat = 'GTiff'
datatype = gdal.GDT_Byte
burnVal = 1 #value for the output image pixels

# Get projection info from reference image
RasImage = gdal.Open(RefImage, gdal.GA_ReadOnly)

for Shapefile in shapefiles:
    ds = ogr.Open(Shapefile)
    ds_layer = ds.GetLayer()
# convert a shapefile to raster
def convertShpToRaster(inputfile, band, size):
    shp = ogr.Open(inputfile)
    lyr = shp.GetLayer(0)
    lyrname = lyr.GetName()
    #color = plt.gray()

    subprocess.call('gdal_rasterize -ts {0} {1} -l {2} -burn {3} ./{4} ./{5}.tif' \
                .format(size[0], size[1], lyrname, band, inputfile, output + '/' + lyrname))

def inputToShapefile():
    # input color, resolution and shapefile location
    #color = plt.gray()
    band = RasImage.GetRasterBand(1)
    size = RasImage.RasterXSize, RasImage.RasterYSize

    # call the function
    for inputfile in shapefiles:
        convertShpToRaster(inputfile, band, size)   

inputToShapefile()

gdal.Polygonize( srcband, None, dst_layer, -1, [], callback=None )