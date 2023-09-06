import logging

from config import DEVELOPMENT
from config import TEMP_FILES
from src.view.gui_main import KmlCreatorGui

if DEVELOPMENT:
    logging.basicConfig(
        filename=str(TEMP_FILES / "app.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )
logger = logging.getLogger(__name__)
logger.info("Starting app")

if __name__ == "__main__":
    view = KmlCreatorGui()
    view.mainloop()
