import json
import logging
import os
import threading
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

import pandas as pd
from model.bin import Bin
from model.prepar import Preper
from model.runner import Runner

from config import BINNED_SHP_PATH
from config import COLUMNS_TO_DROP
from config import KML_PATH
from config import POLYGON_SHP_PATH
from config import SAVED_STATE
from config import TIF_PATH

if TYPE_CHECKING:
    from view.gui_main import KmlCreatorGui

logger = logging.getLogger(__name__)


class Presenter:
    """
    Presenter class handles the communication between the view and the model. The view
    presenter by calling methods from the presenter. The presenter communicates with
    from the view, and vice versa. Likewise, the presenter communicates with the model
    model, and vice versa.
    When the Presenter is instantiated, it clears any temporary files that may have
    session. It also loads the GUI state from the saved state JSON file, if it exists.
    :param view: The view that the presenter is controlling.
    """

    view: "KmlCreatorGui"

    def __init__(self, view):
        self.view = view
        self.clear_temp_files()

    def clear_temp_files(self) -> None:
        """
        Clear any temporary files that may have been left over from a previous session.
        Uses a list of folders to iterate through and delete all files in each folder,
        except for ".gitkeep". .gitkeep is used to keep the folders in the repository,
        but is not needed for the program to run. Note that the SHP_PATH folder is not
        included in the list of folders to clean up, as this contains the shapefiles
        that the user might want to keep for later use.
        """
        # List of folders to clean up
        folders = [BINNED_SHP_PATH, POLYGON_SHP_PATH, TIF_PATH, KML_PATH]

        for folder in folders:
            for file_path in folder.iterdir():
                if file_path.name != ".gitkeep":
                    if file_path.is_file():  # Not a folder
                        file_path.unlink()
                        logger.info(f"Deleted file {file_path}")

    def load_gui_state(self, csv_file_path: Path = None) -> None:
        """
        Load the GUI state from the saved state JSON file. If the saved state JSON file
        saved state JSON file does not contain a csv_file_path, then the GUI state is
        longer exists or if there is an error loading the saved state JSON file, then
        :param csv_file_path: The path to the CSV file to load the GUI state from. If
        is loaded from the saved state JSON file.
        """
        rows, settings, file_path = [], {}, None
        if csv_file_path:
            file_path = csv_file_path
        else:
            # Open the saved state file
            try:
                with open(str(SAVED_STATE), "r") as f:
                    state = json.load(f)
            except FileNotFoundError:
                logger.exception("No saved state found")
                return
            try:
                saved_file_path = state.get("csv_file_path", None)
                if saved_file_path in [None, "", "."]:
                    return
                # Check if csv file specified in saved state still exists
                if not os.path.exists(saved_file_path):
                    return
                file_path = Path(saved_file_path)
                rows = state.get("rows", [])
                settings = state.get("settings", {})
            except FileNotFoundError:
                logger.exception("No saved state found")
            except Exception as e:
                logger.exception(e)

        if file_path is None:
            return

        target_columns, column_min_max = self.get_target_columns_and_default_values(
            csv_file_path=file_path
        )
        # Target columns are mandatory so we need to check if they exist before loading
        # the state
        if target_columns in [None, []]:
            return

        # Load the GUI state
        self.view.set_target_columns(target_columns=target_columns)
        self.view.set_col_min_max_values(column_min_max=column_min_max)
        self.view.set_csv_file_path(file_path)
        self.view.set_saved_rows(rows=rows)
        self.view.set_settings(settings=settings)
        self.view.create_table_and_accessories()

    def save_gui_state(self) -> None:
        """
        Save the GUI state to the saved state JSON file. If the saved state JSON file
        created. If the saved state JSON file does exist, then it is overwritten. The
        are used to get the GUI state (e.g. csv_file_path, rows, settings, etc.). If
        they are not saved.

        """
        state: dict[str, str | dict[str, str]] = dict()
        if csv_file_path := self.view.get_csv_file_path():
            state["csv_file_path"] = str(csv_file_path)
        if rows := self.view.get_rows(ensure_filled=False):
            state["rows"] = rows
        if settings := self.view.get_settings():
            state["settings"] = settings

        # Save the GUI state to the saved state JSON file
        try:
            with open(str(SAVED_STATE), "w") as f:
                json.dump(state, f)
            logger.info("Saved state to JSON file")
        except FileNotFoundError:
            logger.exception("No saved state found")
        except Exception as e:
            logger.exception(e)

    def get_target_columns_and_default_values(
        self, csv_file_path: Path
    ) -> (list, dict):
        """
        Get the target columns and column min/max values from the CSV file. The target
        the user can select in the GUI's combobox. The column min/max values are the
        each column in the CSV file. These values are used for the lower and upper
        :param csv_file_path: The path to the CSV file to get the target columns and
        """
        # When loading from a saved state, the file may no longer exist
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError
        try:
            with open(csv_file_path, "r") as f:
                # load csv file into pandas
                first_line = f.readline()
                if ";" in first_line:
                    csv_df = pd.read_csv(csv_file_path, index_col=0, sep=";")
                else:
                    csv_df = pd.read_csv(csv_file_path, index_col=0, sep=",")
        except Exception as e:
            logger.exception(e)
            raise e

        # Define columns to drop. They should only be dropped if they exist in the csv
        # file
        cols_to_drop = [col for col in COLUMNS_TO_DROP if col in csv_df.columns]

        csv_df = csv_df.drop(columns=cols_to_drop)
        column_names: list[str] = csv_df.columns.to_list()
        columns_min_and_max: dict[str, tuple[int, int]] = dict()
        # Get min and max values for each column
        for column_name in column_names:
            columns_min_and_max[column_name] = (
                csv_df[column_name].min(),
                csv_df[column_name].max(),
            )
        return column_names, columns_min_and_max

        # Try to load the file

    def initialize_and_run_process(self, data, save_directory) -> None:
        """
        Initialize and run the process in a separate thread. This is done so that the
        GUI does not freeze while the process is running.
        :param data: The data to pass to the process (row data, settings, etc.).
        :param save_directory: The directory to save the output kml files to.
        """
        t = threading.Thread(target=self.run_process, args=(data, save_directory))
        t.start()
        return

    def print_to_view_console(self, text: str) -> None:
        """
        Print text to the view's console.
        """
        logger.info(text)
        self.view.print_to_console(text=text)

    def run_process(self, data, save_directory) -> None:
        """
        Run the process. This is done in a separate thread so that the GUI does not
        running. Bin objects are created from the data and then a Preper and Runner are
        a shapefile for the Runner, which then creates further shapefiles (one for each
        interpolation, running polygonization and converting the polygon shapefiles to
        :param data: The data fetched from the GUI (row data, settings, etc.).
        :param save_directory: The directory to save the output kml files to.
        """
        bins_dict: dict[str, Any] = data["bins"]
        settings_dict: dict[str, Any] = data["settings"]
        bins: list[Bin] = []

        # Create bins
        for k, v in bins_dict.items():
            bins.append(
                Bin(
                    enum=k,
                    column=v["target_column"],
                    description=v["description"],
                    lower=v["lower_bound"],
                    upper=v["upper_bound"],
                    colour=v["colour"],
                    opacity=v["opacity"],
                    ignore=False,
                    boundary_type="[]",
                )
            )

        # Create Preper and Runner
        preper = Preper(
            name=settings_dict["name"],
            save_directory=save_directory,
            csv_file_path=self.view.get_csv_file_path(),
            bins=bins,
            print_to_view=self.print_to_view_console,
        )
        runner = Runner(
            preper=preper,
            print_to_frontend=self.print_to_view_console,
            radius_width_metres=settings_dict["radius_width"],
            radius_height_metres=settings_dict["radius_height"],
            smoothing=settings_dict["smoothing"],
            pixel_size_metres=settings_dict["pixel_size"],
            angle=settings_dict["angle"],
            simplification_tolerance=settings_dict["simplification"],
        )

        self.print_to_view_console("\nCreating shapefile for each bin...")
        preper.create_shp_for_each_bin()

        self.print_to_view_console("\nRunning interpolation for each bin...")
        runner.run_interpolation_for_each_bin()
        self.print_to_view_console("...done.")

        self.print_to_view_console("\nRunning polygonization for each bin...")
        runner.run_polygonize_for_each_bin()
        self.print_to_view_console("...done.")

        self.print_to_view_console("\nCreating KML for each bin...")
        runner.create_kml_for_each_bin()
        self.print_to_view_console("...done.")

        self.print_to_view_console("\nDeleting temporary files...")
        runner.delete_files()
        self.print_to_view_console("...done.")

        self.view.submit.submit_finish()
        self.print_to_view_console("\nFinished.")
