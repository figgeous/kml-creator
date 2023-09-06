import logging
import tkinter as tk
from tkinter import messagebox
from typing import Any

from view.gui_table_row import RowField
from view.gui_table_row import TableRow
from view.gui_table_row_fields import CheckbuttonWithVar

logger = logging.getLogger(__name__)


class TableFrame(tk.Frame):
    """
    The frame containing the table for the GUI. Any existing table is destroyed before
    If there are no saved rows (loaded in from the Presenter), a single empty row is
    :param master: The parent widget.
    """

    headers: list[str] = [
        "Select",
        "Bin Number",
        "Target Column",
        "Description",
        "Lower Bound",
        "Upper Bound",
        "Colour",
        "Opacity (%)",
    ]
    rows = []
    width: int
    cell_width: int = 180
    cell_height: int = 35

    target_columns: list[str]
    saved_rows: dict[str, dict[str, Any]]
    column_min_max: dict[str, tuple[float, float]]
    rows: list[TableRow]
    canvas_frame: tk.Frame
    canvas: tk.Canvas
    title: tk.Label
    scrollbar_v: tk.Scrollbar
    scrollbar_h: tk.Scrollbar

    def __init__(self, parent):
        # Call the parent constructor
        super().__init__()

        self.parent = parent
        self.rows = []

        # Destroy the table if it already exists
        if hasattr(self.parent, "table_frame"):
            self.parent.table_frame.destroy()

        # Pack the table frame into the parent
        self.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        # Get the target columns, saved rows, and column min/max from the parent
        self.target_columns = parent.target_columns
        self.saved_rows = parent.saved_rows
        self.column_min_max = parent.column_min_max

        # Create a title for the table and anchor to the left
        self.title = tk.Label(self, text="Table", anchor="w")
        self.title.pack(side=tk.TOP, fill=tk.X)

        # Create a canvas to hold the table, scrollbars, and headers
        self.canvas = tk.Canvas(self)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.create_scrollbars()
        self.create_headers()

        # Draw the rows
        if self.saved_rows:
            # If there are saved rows, draw them
            for row in self.saved_rows.values():
                # Check that each saved row has all the headers
                try:
                    self.add_row(row)
                except Exception as e:
                    logger.error(e)
        else:
            # Add a row if there are no saved rows
            self.add_row()

        self.update_scrollregion()

    def create_scrollbars(self) -> None:
        """
        Create the vertical and horizontal scrollbars for the table.
        """
        scrollbar_v = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar_v.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.config(yscrollcommand=scrollbar_v.set)
        scrollbar_h = tk.Scrollbar(
            self, orient=tk.HORIZONTAL, command=self.canvas.xview
        )
        # scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)
        scrollbar_h.place(relx=0, rely=1, relwidth=1, anchor="sw")
        self.canvas.config(xscrollcommand=scrollbar_h.set)

    def create_headers(self) -> None:
        """
        Create the column headers for the table.
        """
        # Draw the table headers
        for i, header in enumerate(self.headers):
            x1 = i * self.cell_width
            y1 = 0
            x2 = (i + 1) * self.cell_width
            y2 = self.cell_height
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="gray", outline="black")
            self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=header)

        # Set required width and height of the table
        self.width = len(self.headers) * self.cell_width + 10  # +10 for scrollbar

    def add_row(self, saved_row=None) -> None:
        """
        Add a row to the table and update the scroll region.
        """
        new_row = TableRow(master=self, saved_row_data=saved_row)

        self.rows.append(new_row)

        self.update_scrollregion()

    def remove_rows(self) -> None:
        """
        Remove the selected rows from the table. If no rows are selected, an error
        Once the rows are removed, the remaining rows are re-drawn and the scroll
        """
        rows_to_remove = []
        for row in self.rows:
            if row.is_selected():
                row.destroy_entries()
                rows_to_remove.append(row.row_num)
        if not rows_to_remove:
            messagebox.showerror("Error", "Please select a row to remove")
            return

        rows_to_remove.sort(reverse=True)
        for row_num in rows_to_remove:
            self.rows.pop(row_num)

        remaining_rows = self.get_rows(ensure_filled=False, selected_only=False)

        # Destroy all rows
        for row in self.rows:
            row.destroy_entries()
        self.rows = []

        # Re-draw the rows
        for row in remaining_rows.values():
            self.add_row(row)

        self.update_scrollregion()

    def get_rows(
        self, ensure_filled=True, selected_only=False
    ) -> dict[str, Any] | None:
        """
        Get the data from the table rows and return it as a dictionary. If
        will be displayed to the user if any fields are empty. If selected_only is
        selected rows will be returned.
        :param ensure_filled: If True, ensure all fields are filled before returning
        :param selected_only: If True, only return the data from the selected rows
        """
        if selected_only:
            rows = []
            for row in self.rows:
                if row.is_selected():
                    rows.append(row)
        else:
            rows = self.rows

        # Ensure all fields are filled
        if ensure_filled:
            for row in rows:
                if row.contains_empty_entry():
                    messagebox.showerror("Error", "Please fill in all fields")
                    return

        # Create a dictionary of the table data where the key is the bin
        # number and the value is a dictionary of the row data
        data = {}
        for row in rows:
            data.update(row.get_row_data())
        return data

    def select_all(self, select_all: bool) -> None:
        """
        Select or deselect all rows in the table.
        :param select_all: If True, select all rows. If False, deselect all rows.
        """
        for row in self.rows:
            row: TableRow
            select_box: RowField = row.row_fields[0]
            assert isinstance(
                select_box.widget, CheckbuttonWithVar
            )  # helps type checker
            if select_all:
                select_box.widget.select()
                select_box.widget.var.set(True)
            else:
                select_box.widget.deselect()
                select_box.widget.var.set(False)

    def update_scrollregion(self) -> None:
        """
        Update the scroll region of the canvas to fit the table.
        """
        region = self.canvas.bbox(tk.ALL)
        region = (region[0], region[1], region[2], region[3] + 20)
        self.canvas.config(scrollregion=region)
