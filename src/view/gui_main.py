import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Any

from presenter.presenter import Presenter
from view.gui_button_frame import ButtonFrame
from view.gui_console import ConsoleFrame
from view.gui_header import Header
from view.gui_settings import SettingsFrame
from view.gui_submit import Submit

from .gui_table import TableFrame

logger = logging.getLogger(__name__)


class KmlCreatorGui(tk.Tk):
    """
    The main GUI window.
    """

    presenter: "Presenter"
    settings: dict[str, int]
    csv_file_path: Path = Path()
    saved_rows: list[dict[str, Any]] = []
    header_frame: Header
    table_frame: TableFrame
    console_frame: ConsoleFrame
    button_frame: ButtonFrame
    settings_frame: SettingsFrame
    submit: Submit
    submit_frame: Submit

    def __init__(self):
        # Call the parent constructor
        super().__init__()

        # Set the title of the window
        self.title("KML Creator")

        # Set the default font, size and window size
        self.option_add("*Font", "Verdana 10")
        self.geometry("400x75")

        # Create the header
        self.header_frame = Header(master=self)

        # Create a Presenter object to facilitate communication between the view and
        # the model. The presenter will
        # also load the GUI state from the saved_state JSON file.
        self.presenter = Presenter(view=self)
        self.presenter.load_gui_state()

        # Set up callback for when the window is closed. When this happens, the GUI
        # state is saved to the saved_state
        # JSON file.
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_table_and_accessories(self) -> None:
        """
        Create the table and the accessories (buttons, settings, console).
        """
        if not hasattr(self, "csv_file_path"):
            return

        # Create the table
        self.table_frame = TableFrame(
            parent=self,
        )

        # Create the buttons ("Select all", "Deselect all", "Add row", "Delete row",
        # "Submit")
        self.button_frame = ButtonFrame(master=self)
        self.button_frame.pack(side=tk.TOP, fill=tk.Y, anchor=tk.W)

        # Create the settings ("Name", "Pixel Size", "Radius width", Radius height",
        # "Smoothing", "Angle")
        self.settings_frame = SettingsFrame(master=self)
        self.settings_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        # Create the console
        self.console_frame = ConsoleFrame(master=self)
        self.console_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        # Set the window size to fit the table
        self.geometry(f"{self.table_frame.width + 10}x800")

    @staticmethod
    def ask_user_file_directory() -> Path | None:
        """
        Open a file dialog window to allow the user to select a directory.
        """
        file_path_str: str = filedialog.askdirectory()
        # If user cancels, return None
        if file_path_str == ():
            return None
        return Path(file_path_str)

    @staticmethod
    def ask_user_file_path(filetypes=None) -> Path | None:
        """
        Open a file dialog window to allow the user to select a file. The file must
        When the user cancels the file dialog, file_path is an empty tuple. Return if
        :param filetypes: The file types to allow the user to select. Defaults to CSV
        files.
        """
        if filetypes is None:
            filetypes = [("CSV files", "*.csv")]
        file_path: str = filedialog.askopenfilename(filetypes=filetypes)
        if file_path == ():
            return
        return Path(file_path)

    def get_rows(
        self, ensure_filled=True, selected_only=False
    ) -> dict[str, Any] | None:
        """
        Get the rows from the table.
        :param ensure_filled: If True, displays a message box if any required fields are
         empty, and returns None.
         :param selected_only: If True, only return rows that have been selected
         (checkbox is ticked).
        """
        if hasattr(self, "table_frame"):
            return self.table_frame.get_rows(
                ensure_filled=ensure_filled, selected_only=selected_only
            )

    def get_csv_file_path(self) -> Path | None:
        """
        Get the path to the CSV file.
        """
        if hasattr(self, "header_frame"):
            return self.header_frame.csv_file_path

    def get_settings(self) -> dict[str, int | str] | None:
        """
        Get the settings from the settings frame ("Name", "Pixel Size", "Radius width",
        Radius height", "Smoothing",
        "Angle").
        """
        if hasattr(self, "settings_frame"):
            return self.settings_frame.get_settings()

    def set_csv_file_path(self, file_path=None) -> None:
        """
        Set the path to the CSV file.
        :param file_path: The path to the CSV file.
        """
        if file_path is None:
            file_path: Path = self.ask_user_file_path()
        # Update the file label with the path to the selected file
        self.header_frame.csv_file_path_label.config(text=str(file_path))
        # Convert to pathlib object
        self.header_frame.csv_file_path = file_path

    def set_target_columns(self, target_columns) -> None:
        """
        Set the target columns.
        :param target_columns: The target columns fetched from the column names in the
        CSV file. Certain columns are
        excluded (e.g. "lat", "lon").
        """
        self.target_columns = target_columns

    def set_saved_rows(self, rows) -> None:
        """
        Set the saved rows.
        :param rows: The rows fetched from the saved_state JSON file.
        """
        self.saved_rows = rows

    def set_col_min_max_values(self, column_min_max) -> None:
        """
        Provide the minimum and maximum values for each column in the CSV file.
        :param column_min_max: The minimum and maximum values for each column in the
        CSV file.
        """
        self.column_min_max = column_min_max

    def set_settings(self, settings) -> None:
        """
        Set the settings in the settings frame ("Name", "Pixel Size", "Radius width",
        Radius height", "Smoothing",
        "Angle").
        :param settings: The saved settings fetched from the saved_state JSON file.
        """
        # TODO: Set settings.
        pass

    def print_to_console(self, text) -> None:
        """
        Print text to the console.
        :param text: The text to print to the console.
        """
        self.console_frame.console_text.configure(state="normal")
        self.console_frame.console_text.insert(tk.END, text)
        self.console_frame.console_text.configure(state="disabled")

    def submit(self):
        """
        The method that is called when the "Submit" button is clicked. Creates a Submit
        rows from the table, and then calls the submit method in the presenter. A
        submission is in progress.
        """
        self.submit = Submit(master=self)

    def on_close(self) -> None:
        """
        The method that is called when the window is closed. Saves the GUI state to a
        JSON file before closing.
        """
        try:
            self.presenter.save_gui_state()
        except Exception as e:
            logger.error(e)
        self.destroy()
