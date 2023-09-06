import tkinter as tk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from view.gui_main import KmlCreatorGui


class SettingsFrame(tk.Frame):
    """
    The settings frame. The settings are stored in a dictionary, with the key being the
    being a tuple containing the default value and the datatype of the value. Any
    the new ones are created.
    :param master: The parent widget.
    """

    master: "KmlCreatorGui"
    settings: dict[str, tuple[str | int, type]]
    settings_title: tk.Label
    name_label: tk.Label
    name_entry: tk.Entry
    pixel_size_label: tk.Label
    pixel_size_entry: tk.Entry
    radius_width_label: tk.Label
    radius_width_entry: tk.Entry
    radius_height_label: tk.Label
    radius_height_entry: tk.Entry
    angle_label: tk.Label
    angle_entry: tk.Entry
    smoothing_label: tk.Label
    smoothing_entry: tk.Entry
    simplification_label: tk.Label
    simplification_entry: tk.Entry

    def __init__(self, master: "KmlCreatorGui"):
        super().__init__(master)
        self.master = master

        # Destroy the old settings if they exist
        if hasattr(self.master, "settings_frame"):
            self.master.settings_frame.destroy()

        # Settings with default values
        self.settings = {
            "name": ("Batch 1", str),
            "pixel_size": (10, int),
            "radius_width": (60, int),
            "radius_height": (60, int),
            "angle": (0, int),
            "smoothing": (10, float),
            "simplification": (10, float),
        }

        # Create a title for the other settings and anchor it to the left
        self.settings_title = tk.Label(self, text="Settings", anchor=tk.W)
        self.settings_title.pack(side=tk.TOP, fill=tk.X)

        # Create the settings and pack them
        for setting in self.settings:
            label = setting.replace("_", " ").title()
            setattr(
                self,
                setting + "_label",
                tk.Label(self, text=label),
            )
            getattr(self, setting + "_label").pack(side=tk.LEFT, padx=10, pady=5)
            setattr(self, setting + "_entry", tk.Entry(self, width=10))
            getattr(self, setting + "_entry").pack(side=tk.LEFT, padx=10, pady=5)
            getattr(self, setting + "_entry").insert(0, self.settings[setting][0])

    def get_settings(self) -> dict[str, int | str]:
        """
        Get the settings from the settings frame.
        """
        settings = {}
        for setting in self.settings:
            datatype = self.settings[setting][1]
            value = getattr(self, setting + "_entry").get()
            if datatype == int:
                settings[setting] = int(value)
            elif datatype == str:
                settings[setting] = value  # Already a string
            elif datatype == float:
                settings[setting] = float(value)
            else:
                raise TypeError(f"Type {self.settings[setting][1]} not supported")
        return settings
