# CSV to KML Converter

**CSV to KML Converter** interpolates disparate geographic data points into continuous fields of mapped vectors. It translates your CSV data into Google's Keyhole Markup Language (KML), making it easy to import into Google Maps.

## Features:
- Uses GDAL's moving average interpolation algorithm to transform discrete data points into continuous fields, providing more comprehensive mapped areas.
- Allows users to create separate layers, each with its own color and opacity, corresponding to specific value ranges in given dataset columns.
- Customize GDAL's algorithm settings via the GUI to define the search ellipsis's horizontal and vertical extents and set the degree of smoothing/simplification.

![KML_Creator_screenshot](https://github.com/figgeous/kml-creator/assets/56649407/792a230c-9a49-431f-b76f-52d31f3782b1)

## Sample Output:

See a [sample output](https://www.google.com/maps/d/edit?mid=1k_8PufWkvy9zPsLz70ibuwcq5unQ2rI&usp=sharing) of the CSV to KML conversion.

## System Requirements:
- **Operating System**: Linux (tested on Ubuntu 22)
- **Programming Language**: Python (tested on Python 3.10)
- **Dependency**: GDAL

## Installation and Running:
1. Install the necessary packages:
2. pip install -r requirements.txt
3. Run the application: python __make__.py
   
## Contact & Collaboration:
Interested in collaborating or have questions? Drop an email at [thomas.oneill@gmail.com](mailto:thomas.oneill@gmail.com).


