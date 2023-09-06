import tkinter as tk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from view.gui_main import KmlCreatorGui


class ConsoleFrame(tk.Frame):
    """
    A frame that contains a console for the user to view the output of the program
    :param master: The parent widget
    """

    master: "KmlCreatorGui"
    console_title: tk.Label
    console_text: tk.Text
    scrollbar: tk.Scrollbar

    def __init__(self, master: "KmlCreatorGui"):
        super().__init__()
        self.parent = master

        # Destroy the old console if it exists
        if hasattr(master, "console_frame"):
            self.parent.console_frame.destroy()

        # Create a title for the console and anchor it to the left.
        self.console_title = tk.Label(self, text="Console", anchor=tk.W)
        self.console_title.pack(side=tk.TOP, fill=tk.X)

        # Create a text widget for the console. Readonly.
        self.console_text = tk.Text(self, state=tk.DISABLED)
        self.console_text.pack(side=tk.TOP, fill=tk.X)

        # Create a scrollbar for the console
        self.scrollbar = tk.Scrollbar(self, command=self.console_text.yview)
        self.scrollbar.pack(side="right", fill="y")

        # Configure the console to use the scrollbar
        self.console_text.config(yscrollcommand=self.scrollbar.set)
