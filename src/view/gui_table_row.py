import logging
import tkinter as tk
from typing import Any
from typing import TYPE_CHECKING

from view.gui_table_row_fields import CheckbuttonWithVar
from view.gui_table_row_fields import ColourEntry
from view.gui_table_row_fields import LowerAndUpperBoundEntry
from view.gui_table_row_fields import RowField
from view.gui_table_row_fields import TargetColumnCombobox

if TYPE_CHECKING:
    from view.gui_table import TableFrame

logger = logging.getLogger(__name__)


class TableRow:
    """
    A row in the table. Each row contains the columns reflected in TableFrame.headers.
    :param master: The parent widget.
    :param saved_row_data: The saved row data, if any.
    """

    master: "TableFrame"

    # Entry widgets
    check_box: CheckbuttonWithVar
    bin_number_label: tk.Label
    target_column_combo_box: TargetColumnCombobox
    entry_colour: ColourEntry
    opacity_entry: tk.Entry
    description_entry: tk.Entry
    lower_bound_entry: LowerAndUpperBoundEntry
    upper_bound_entry: LowerAndUpperBoundEntry

    # Saved row data
    saved_row_data: dict[str, Any]

    # Cell and button dimensions
    cell_width: int
    cell_height: int
    button_width: int
    button_height: int

    def __init__(self, master: "TableFrame", saved_row_data=None):
        self.row_num = len(master.rows)
        self.saved_row_data = saved_row_data

        # Get the parent attributes
        self.parent = master
        self.cell_width = master.cell_width
        self.cell_height = master.cell_height
        self.canvas = master.canvas
        self.target_columns = master.target_columns
        self.target_col_min_max = master.column_min_max

        # Create entry widgets
        self.row_fields = []
        self.create_entries()

    def make_canvas_window(self, num, item) -> None:
        """
        Create a canvas window for the given widget.
        :param num: The column number of the widget.
        :param item: The widget to create a canvas window for.
        """
        canvas_y_padding: int = (
            5  # 5 pixels of padding on the left and right of the widget
        )
        canvas_x_padding: int = (
            5  # 5 pixels of padding on the top and bottom of the widget
        )
        canvas_x_offset: int = 10  # 10 pixels of offset from the right edge of the cell
        x1: int = num * self.cell_width
        y1: int = (self.row_num + 1) * self.cell_height
        self.canvas.create_window(
            x1 + canvas_y_padding,
            y1 + canvas_x_padding,
            width=self.cell_width - canvas_x_offset,
            window=item,
            anchor="nw",
        )

    def create_entries(self) -> None:
        """
        Create the entry widgets for this row. First, the widgets are created, setting
        value or a default value. Then, the canvas windows are created for each widget
        the row_fields list.
        """
        # Assign the entry widgets to the row
        self.check_box = CheckbuttonWithVar()
        self.bin_number_label = tk.Label(self.canvas, text=f"{self.row_num + 1}")
        self.target_column_combo_box = TargetColumnCombobox(
            parent=self, values=self.target_columns
        )
        self.description_entry = tk.Entry(self.canvas)
        self.lower_bound_entry = LowerAndUpperBoundEntry(
            parent=self,
        )
        self.upper_bound_entry = LowerAndUpperBoundEntry(
            parent=self,
        )
        self.entry_colour = ColourEntry(parent=self)
        self.opacity_entry = tk.Entry(self.canvas)

        # Make the canvas window for each widget and append each widget to the
        # row_fields list.
        # Select column
        if self.saved_row_data and (col := "select") in self.saved_row_data:
            self.check_box.var.set(self.saved_row_data[col])
        else:
            self.check_box.var.set(False)
        self.make_canvas_window(num=len(self.row_fields), item=self.check_box)
        self.row_fields.append(
            RowField(col_name="select", widget=self.check_box, datatype=bool)
        )
        # Bin Number column
        self.make_canvas_window(num=len(self.row_fields), item=self.bin_number_label)
        self.row_fields.append(
            RowField(col_name="bin_number", widget=self.bin_number_label, datatype=int)
        )

        # Target Column column
        self.make_canvas_window(
            num=len(self.row_fields), item=self.target_column_combo_box
        )
        self.row_fields.append(
            RowField(
                col_name="target_column",
                widget=self.target_column_combo_box,
                datatype=str,
            )
        )

        # Description column
        if self.saved_row_data and (col := "description") in self.saved_row_data:
            if self.saved_row_data[col]:
                self.description_entry.insert(0, self.saved_row_data[col])
        self.make_canvas_window(num=len(self.row_fields), item=self.description_entry)
        self.row_fields.append(
            RowField(
                col_name="description", widget=self.description_entry, datatype=str
            )
        )

        # Lower and Upper Bounds columns
        for col, widget in [
            ("lower_bound", self.lower_bound_entry),
            ("upper_bound", self.upper_bound_entry),
        ]:
            if self.saved_row_data and col in self.saved_row_data:
                if self.saved_row_data[col] is not None:  # allow 0 as a valid value
                    widget.insert(0, str(self.saved_row_data[col]))
                else:
                    widget.update_to_default_value()
            else:
                widget.update_to_default_value()
            self.make_canvas_window(num=len(self.row_fields), item=widget)
            self.row_fields.append(RowField(col_name=col, widget=widget, datatype=int))

        # Colour column
        if self.saved_row_data and (col := "colour") in self.saved_row_data:
            if colour := self.saved_row_data[col]:
                self.entry_colour.set_colour(colour)
        self.make_canvas_window(num=len(self.row_fields), item=self.entry_colour)
        self.row_fields.append(
            RowField(col_name="colour", widget=self.entry_colour, datatype=str)
        )

        # Opacity Column column
        if self.saved_row_data and (col := "opacity") in self.saved_row_data:
            if self.saved_row_data[col] is not None:  # allow 0 opacity
                self.opacity_entry.insert(0, str(self.saved_row_data[col]))
        self.make_canvas_window(num=len(self.row_fields), item=self.opacity_entry)
        self.row_fields.append(
            RowField(col_name="opacity", widget=self.opacity_entry, datatype=int)
        )

    def get_row_data(self) -> dict[str, Any]:
        """
        Returns a dict containing the data from the row. A try/except block is used to
        datatype. If this fails, the field is not added to the row_data dict, as the
        invalid data (e.g. a string in a field that should contain an int).
        """
        row_data = {}
        bin_number = None
        for row_field in self.row_fields:
            # The bin number uses a different function to get the data from the widget
            if row_field.col_name == "bin_number":
                bin_number = row_field.widget.cget("text")
                continue

            row_data[row_field.col_name] = self._convert_field_to_datatype(row_field)
        return {bin_number: row_data}

    def destroy_entries(self) -> None:
        """
        Destroys all the entry widgets in the row_fields list.
        """
        for entry in self.row_fields:
            entry.widget.destroy()
        self.row_fields = []

    def is_selected(self) -> bool:
        """
        Returns True if the row is selected, False otherwise.
        """
        return self.row_fields[0].widget.var.get()

    def contains_empty_entry(self) -> bool:
        """
        Returns True if any of the entry widgets in the row_fields list are empty, False
         otherwise. Ignores the
        Bin Number and Select columns.
        """
        for row_field in self.row_fields:
            if row_field.col_name in ["bin_number", "selected"]:
                continue
            if row_field.widget.get() is None:
                return True
            row_field_value = self._convert_field_to_datatype(row_field)
            if row_field_value is not None:
                continue
            else:
                return True
        return False

    def _convert_field_to_datatype(self, row_field: RowField) -> Any:
        """
        Converts the data in the row_field to the correct datatype. If the conversion
        fails, None is returned.
        """
        if row_field.col_name == "bin_number":
            return row_field.widget.cget("text")

        value = row_field.widget.get()
        value = None if value == "" else value
        try:
            if isinstance(row_field.datatype(), int):
                value = int(value)
            elif isinstance(row_field.datatype(), float):
                value = float(value)
        except ValueError:
            logger.exception(
                f"Failed to convert {row_field.col_name} to {row_field.datatype()}"
            )
            value = None
        finally:
            return value
