import dataclasses


@dataclasses.dataclass
class Bin:
    """
    A class to represent a bin. A bin is a range of values that are grouped together.
    0 - 10, 10 - 20, 20 - 30, etc. The bin class is used to store the information about
    description, the range of values, the colour, etc. The attributes ending in
    Preper or the Runner during the processing of the data.
    :param enum: The bin number. This is used to identify the bin in the code.
    :param column: The column name that the bin is associated with.
    :param description: The description of the bin.
    :param lower: The lower bound of the bin.
    :param upper: The upper bound of the bin.
    :param bin_shp_file_name: The name of the bin shapefile.
    :param tif_file_name: The name of the tif file.
    :param polygon_shp_file_name: The name of the polygon shapefile.
    :param kml_file_name: The name of the kml file.
    :param colour: The colour of the bin.
    :param opacity: The opacity of the bin.
    :param ignore: Whether or not the bin should be ignored.
    :param boundary_type: The boundary type of the bin. This is used to determine
    are inclusive or exclusive. The first character is the lower bound and "[" means
    The second character is the upper bound and "]" means inclusive and "[" means
    that the lower bound is inclusive and the upper bound is exclusive.
    """

    enum: int
    column: str
    description: str
    lower: int
    upper: int
    bin_shp_file_name: str = None
    tif_file_name: str = None
    polygon_shp_file_name: str = None
    kml_file_name: str = None
    colour: str = "D10000"
    opacity: int = 100
    ignore: bool = False
    boundary_type: str = "[["
