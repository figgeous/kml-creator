import filecmp
import logging
import os
import shutil
import sys
import unittest
from pathlib import Path

from src.model.bin import Bin
from src.model.prepar import Preper
from src.model.runner import Runner
from tests.context.bins_for_testing import single_bin
from tests.context.bins_for_testing import test_bins

mussel_data_folder = Path(os.path.abspath(".")) / Path("src")
sys.path.insert(0, str(mussel_data_folder))

from config import CSV_PATH, BINNED_SHP_PATH, POLYGON_SHP_PATH, TEST_PATH
from config import KML_PATH
from config import TEST_CONTEXT_PATH
from config import TEST_CSV_DIMENSIONS
from config import TEST_CSV_NAME
from config import TEST_KML_NAME
from config import TEST_POLYGON_SHP_NAME
from config import TEST_SHP_NAME
from config import TEST_TIF_NAME
from config import TIF_PATH

logging.basicConfig(
    filename=str(TEST_PATH / "tests.log"),
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)
logger.info("Starting app")


class SetupTest(unittest.TestCase):
    """
    Setup and teardown for tests
    """

    def setUp(self):
        source_csv_path = str(TEST_CONTEXT_PATH / (TEST_CSV_NAME + ".csv"))
        dest_csv_path = str(CSV_PATH / (TEST_CSV_NAME + ".csv"))
        logger.info(f"Copying {source_csv_path} to {dest_csv_path}")
        shutil.copy(source_csv_path, dest_csv_path)
        self.copied_csv_path = dest_csv_path
        self.assertTrue(os.path.exists(self.copied_csv_path))

    def tearDown(self):
        # Delete copied csv file
        os.remove(self.copied_csv_path)
        logger.info("Remove csv path: " + self.copied_csv_path)


class TestObjectInitialization(SetupTest):
    """
    Test that the models initialize correctly
    """

    def test_bin_class_initializes(self):
        """
        Test that the Bin class initializes
        """
        bin = Bin(
            enum=1,
            column="value",
            description="test description",
            lower=0,
            upper=666,
        )
        self.assertEqual(bin.enum, 1)
        self.assertEqual(bin.column, "value")
        self.assertEqual(bin.description, "test description")
        self.assertEqual(bin.lower, 0)
        self.assertEqual(bin.upper, 666)
        # Test default values
        self.assertEqual(bin.boundary_type, "[[")
        self.assertEqual(bin.colour, "D10000")
        self.assertEqual(bin.opacity, 100)
        self.assertEqual(bin.ignore, False)

    def test_preper_class_initialize(self):
        """
        Test that the Preper class initializes
        """
        preper = Preper(
            name="test name",
            csv_file_path=CSV_PATH / str(TEST_CSV_NAME + ".csv"),
            bins=test_bins,
        )
        self.assertEqual(preper.name, "test name")
        self.assertEqual(preper.csv_file_name, TEST_CSV_NAME)
        self.assertEqual(preper.bins, test_bins)

    def test_runner_class_initializes(self):
        """
        Test that the Runner class initializes
        """
        preper = Preper(
            name="test name",
            csv_file_path=CSV_PATH / str(TEST_CSV_NAME + ".csv"),
            bins=test_bins,
        )
        runner = Runner(
            preper=preper,
            radius_width_metres=60,
            radius_height_metres=60,
            pixel_size_metres=10,
            smoothing=10,
        )
        self.assertEqual(runner.name, "test name")
        self.assertEqual(runner.preper.csv_file_name, TEST_CSV_NAME)
        self.assertEqual(runner.preper.bins, test_bins)
        self.assertEqual(runner.preper.name, "test name")
        self.assertEqual(runner.preper.geo_df.shape, TEST_CSV_DIMENSIONS)


class TestWholeProcess(SetupTest):
    preper: Preper
    runner: Runner
    name: str

    """
    Test the whole process, from csv to kml. Each Runner method is tested in turn.
    """

    def setUp(self):
        """
        Set up the test by copying the test csv file to the csv folder and creating a Preper and Runner object
        """
        super().setUp()
        self.name = "test_name"
        self.preper = Preper(
            name=self.name,
            csv_file_path=CSV_PATH / str(TEST_CSV_NAME + ".csv"),
            bins=single_bin,
        )
        self.runner = Runner(
            preper=self.preper,
            radius_width_metres=60,
            radius_height_metres=60,
            pixel_size_metres=10,
            smoothing=10,
        )

    def tearDown(self):
        """
        Tear down the test by deleting the files created by the Preper and Runner object
        """
        self.preper.delete_shp_file()
        self.runner.delete_files(delete_kml_files=True)
        super().tearDown()

    def test_expected_shp_generated(self):
        """
        Test that create_shp_for_each_bin creates the expected binned shapefile
        """
        self.preper.create_shp_for_each_bin()
        created_shp_name = single_bin[0].bin_shp_file_name
        path_to_created_shp = str(BINNED_SHP_PATH / (created_shp_name + ".shp.zip"))
        path_to_expected_shp = str(TEST_CONTEXT_PATH / (TEST_SHP_NAME + ".shp.zip"))
        comparison = filecmp.cmp(
            path_to_created_shp, path_to_expected_shp, shallow=False
        )
        # Compare two files are roughly the same size
        size_diff = abs(
            os.path.getsize(path_to_created_shp) - os.path.getsize(path_to_expected_shp)
        )
        self.assertTrue(size_diff < 50)  # 50 bytes

    def test_expected_tif_generated(self):
        """
        Test that run_interpolation_for_each_bin creates the expected tif file
        """
        self.preper.create_shp_for_each_bin()
        self.runner.run_interpolation_for_each_bin()
        created_tif_name = single_bin[0].tif_file_name
        path_to_created_tif = str(TIF_PATH / (created_tif_name + ".tif"))
        path_to_expected_tif = str(TEST_CONTEXT_PATH / (TEST_TIF_NAME + ".tif"))
        comparison = filecmp.cmp(
            path_to_created_tif, path_to_expected_tif, shallow=False
        )
        self.assertTrue(comparison)

    def test_expected_polygon_shp_generated(self):
        """
        Test that run_polygonize_for_each_bin creates the expected polygon shapefile
        """
        self.preper.create_shp_for_each_bin()
        self.runner.run_interpolation_for_each_bin()
        self.runner.run_polygonize_for_each_bin()
        created_shp_name = single_bin[0].polygon_shp_file_name
        path_to_created_shp = str(POLYGON_SHP_PATH / (created_shp_name + ".shp.zip"))
        path_to_expected_shp = str(
            TEST_CONTEXT_PATH / (TEST_POLYGON_SHP_NAME + ".shp.zip")
        )

        # Compare two files are roughly the same size
        size_diff = abs(
            os.path.getsize(path_to_created_shp) - os.path.getsize(path_to_expected_shp)
        )
        self.assertTrue(size_diff < 50)  # 50 bytes

    def test_expected_kml_generated(self):
        """
        Test that run_all creates the expected kml file
        """
        self.preper.create_shp_for_each_bin()
        self.runner.run_interpolation_for_each_bin()
        self.runner.run_polygonize_for_each_bin()
        self.runner.create_kml_for_each_bin()
        created_kml_name = single_bin[0].kml_file_name
        path_to_created_kml = str(KML_PATH / (created_kml_name + ".kml"))
        path_to_expected_kml = str(TEST_CONTEXT_PATH / (TEST_KML_NAME + ".kml"))

        # Compare two files are roughly the same size
        size_diff = abs(
            os.path.getsize(path_to_expected_kml) - os.path.getsize(path_to_created_kml)
        )
        self.assertTrue(size_diff < 50)  # 50 bytes


if __name__ == "__main__":
    unittest.main()
