import tkinter as tk
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from view.gui_main import KmlCreatorGui


class Header(tk.Frame):
    """
    Header frame for the GUI.
    :param master: The parent widget.
    """

    master: "KmlCreatorGui"
    csv_file_path: Path = Path()
    load_csv_text: tk.Label
    load_csv_button: tk.Button
    csv_file_path_label: tk.Label

    def __init__(self, master: "KmlCreatorGui"):
        super().__init__()
        self.master = master
        self.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        self.load_csv_text = tk.Label(self, text="Load file", anchor=tk.W)
        self.load_csv_text.pack(side=tk.TOP, fill=tk.X, expand=True)

        self.load_csv_button = tk.Button(
            self,
            text="CSV file",
            command=self.load_csv_button,
        )
        self.load_csv_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.csv_file_path_label = tk.Label(self, text="No file selected")
        self.csv_file_path_label.pack(side=tk.LEFT, padx=10, pady=5)

    def load_csv_button(self) -> None:
        """
        Callback for the load_csv_button. If the user presses cancel, the function
        returns and nothing happens.
        """
        csv_file_path = self.master.ask_user_file_path()
        if csv_file_path in [None, ""]:
            return
        self.master.presenter.load_gui_state(csv_file_path=csv_file_path)
