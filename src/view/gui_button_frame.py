import tkinter as tk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from view.gui_main import KmlCreatorGui


class ButtonFrame(tk.Frame):
    """
    The frame containing the buttons for the GUI. Any existing buttons are destroyed
    before the new ones are created.
    :param master: The parent widget.
    """

    master: "KmlCreatorGui"
    select_all_button: tk.Button
    deselect_all_button: tk.Button
    add_button: tk.Button
    remove_button: tk.Button
    submit_button: tk.Button

    def __init__(self, master: "KmlCreatorGui"):
        super().__init__(master)
        self.parent = master

        # Destroy the buttons if they already exist
        if hasattr(self.parent, "button_frame"):
            self.parent.button_frame.destroy()

        button_data: list[tuple[str, callable]] = [
            ("Select All", lambda: master.table_frame.select_all(True)),
            ("Deselect All", lambda: master.table_frame.select_all(False)),
            ("Add Row", master.table_frame.add_row),
            ("Remove Row", master.table_frame.remove_rows),
            ("Submit", master.submit),
        ]

        for text, command in button_data:
            button = tk.Button(self, text=text, command=command)
            setattr(self, text.lower().replace(" ", "_") + "_button", button)
            button.pack(side=tk.LEFT, padx=10, pady=10, anchor=tk.W)
