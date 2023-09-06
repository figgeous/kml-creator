import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from tkinter.colorchooser import askcolor
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from view.gui_table_row import TableRow


@dataclass
class RowField:
    col_name: str
    widget: tk.Checkbutton | tk.Entry | tk.Label | ttk.Combobox
    datatype: Any


class TargetColumnCombobox(ttk.Combobox):
    """
    A combobox that displays the target columns. If the user has loaded in a saved row,
    the combobox will display the target column that was selected when the row was
    saved.
    :param parent: The parent widget.
    :param values: The values to display in the combobox.
    """

    def __init__(self, parent: "TableRow", values: list):
        super().__init__(parent.canvas, values=values)
        self.parent = parent
        if parent.saved_row_data and (col := "target_column") in parent.saved_row_data:
            self.current(parent.target_columns.index(parent.saved_row_data[col]))
        else:
            self.current(0)
        self.bind("<<ComboboxSelected>>", self._update_min_and_max_bounds)

    def _update_min_and_max_bounds(self, *args):
        self.parent.lower_bound_entry.update_to_default_value()
        self.parent.upper_bound_entry.update_to_default_value()


class LowerAndUpperBoundEntry(tk.Entry):
    """
    A text entry box that displays the min and max bounds of the target column. The
    and is erased when the user clicks on the entry box. If the user clicks away from
    anything, the default value is restored.
    :param parent: The parent widget.
    """

    _default: str

    def __init__(self, parent, **kwargs):
        super().__init__(parent.canvas, **kwargs)
        self.parent: TableRow = parent
        self._get_default_value()
        self.bind("<FocusIn>", self.erase_default_value)
        self.bind("<FocusOut>", self.restore_default_value_if_empty)

    def _get_default_value(self):
        min_and_max = self.parent.target_col_min_max[
            self.parent.target_column_combo_box.get()
        ]
        self._default = f"Min: {min_and_max[0]}, Max: {min_and_max[1]}"

    def update_to_default_value(self):
        self._get_default_value()
        self.delete(0, tk.END)
        self.insert(0, self._default)
        self.config(fg="grey")

    def erase_default_value(self, event=None):
        if self.get() == self._default:
            self.delete(0, tk.END)
        self.config(fg="black")

    def restore_default_value_if_empty(self, event=None):
        if self.get() == "":
            self.update_to_default_value()


class ColourEntry(tk.Entry):
    """
    A text entry box that displays the colour picked from the colour picker dialog.
    :param parent: The parent widget.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent.canvas, **kwargs)
        self.parent: TableRow = parent
        self.bind("<Button-1>", lambda event: self._choose_color())

    def set_colour(self, colour):
        """
        Sets the colour of the entry box to the colour passed in.
        """
        self._choose_color(colour=colour)

    def _choose_color(self, colour: str = None):
        """
        Displays the colour picker dialog and sets the text in the entry box to the hex
        :param colour: The colour to set the entry box to. If None, the colour picker
        """
        if not colour:
            colour = askcolor()[1]  # returns a tuple (None, '#ffffff')
        # If a color is selected, set the text in the text entry to the hex value of
        # the colour
        self.delete(0, tk.END)
        self.insert(0, str(colour))
        # set entry box colour to the colour picked
        self.config(bg=colour, fg=colour)


class CheckbuttonWithVar(tk.Checkbutton):
    """
    A checkbutton that has a BooleanVar associated with it. The BooleanVar is set to
    True when the checkbutton is checked, and False when it is unchecked.
    """

    var: tk.BooleanVar

    def __init__(self, *args, **kwargs):
        self.var = tk.BooleanVar()
        super().__init__(*args, variable=self.var, **kwargs)

    def get(self):
        return self.var.get()
