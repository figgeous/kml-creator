import logging
import tkinter as tk
from typing import TYPE_CHECKING

from config import SPINNER_SPEED

if TYPE_CHECKING:
    from view.gui_main import KmlCreatorGui

logger = logging.getLogger(__name__)


class Submit:
    """
    The class that collects the data from the GUI (table and settings) and submits it
    to the presenter. Also displays a spinner while the process is running.
    :param master: The parent widget.
    """

    master: "KmlCreatorGui"
    spinner_title: tk.Label
    spinner: tk.Label

    def __init__(self, master: "KmlCreatorGui"):
        self.master = master

        # Clear parent.console_frame.console
        self.master.console_frame.console_text.config(state=tk.NORMAL)
        self.master.console_frame.console_text.delete("1.0", tk.END)
        data = {}

        # Get the save directory from the user
        save_directory = master.ask_user_file_directory()
        # If user presses cancel, return.
        if not save_directory:
            logger.info("User cancelled file dialog")
            return
        logger.info(f"Save directory: {save_directory}")

        # Get the row data from the table
        try:
            if bins := master.get_rows(ensure_filled=True, selected_only=True):
                data["bins"] = bins
            else:
                return
        except ValueError as e:
            logger.error(e)
            self.master.print_to_console(
                "There was a problem collecting the data from the table."
            )
            return

        # Get the data from the settings
        data["settings"] = master.settings_frame.get_settings()
        logger.info(f"Settings: {data['settings']}")

        if data:
            self.submit_start_spinner()
            master.presenter.initialize_and_run_process(
                data=data, save_directory=save_directory
            )

    def submit_start_spinner(self) -> None:
        """
        Start the spinner animation.
        """
        # Disable the Submit button so the user can't click it again
        self.master.button_frame.submit_button.config(state="disabled")

        # Create the spinner title
        self.spinner_title = tk.Label(self.master.button_frame, text="Processing...")
        self.spinner_title.pack(side=tk.LEFT, pady=10)
        # Create the spinner
        self.spinner = tk.Label(self.master.button_frame, text="--")
        self.spinner.pack(side=tk.LEFT, pady=10)
        # Start the spinner animation
        self.submit_animate_spinner()

    def submit_animate_spinner(self) -> None:
        """
        Animate the spinner. The animation is done by rotating the spinner 30 degrees
        every number of milliseconds set by config.SPINNER_SPEED.
        """
        # Rotate the spinner by 30 degrees
        text = self.spinner.cget("text")
        spin_dict = {"--": "\\", "\\": "|", "|": "/", "/": "--"}
        self.spinner.config(text=spin_dict[text])

        # Schedule the next rotation after SPINNER_SPEED milliseconds
        if self.master.button_frame.submit_button.cget("state") == "disabled":
            self.master.table_frame.after(SPINNER_SPEED, self.submit_animate_spinner)

    def submit_finish(self) -> None:
        """
        Finish the submit process. This is called by the presenter when the process is
        finished.
        """
        # Remove the spinner and spinner title
        self.spinner_title.pack_forget()
        self.spinner.pack_forget()

        # Re-enable the submit button
        self.master.button_frame.submit_button.config(state="normal")
