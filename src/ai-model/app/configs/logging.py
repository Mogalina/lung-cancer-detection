import logging
import os


def setup_logging(log_file: str = "app.log", log_level: int = logging.INFO) -> None:
    """
    Set up logging for the entire application. Logs to stdout and to a file.

    Args:
        log_file (str): The filename where logs will be written.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
    """
    if logging.getLogger().hasHandlers():
        logging.getLogger().handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # File handler
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(os.path.join("logs", log_file), mode="a")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Root logger setup
    logging.basicConfig(
        level=log_level,
        handlers=[console_handler, file_handler]
    )
