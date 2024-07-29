import logging
from absl import logging as absl_logging
import logging.handlers
from rich.logging import RichHandler

RICH_FORMAT = "[%(filename)s:%(lineno)s] >> %(message)s"
FILE_HANDLER_FORMAT = "[%(asctime)s] %(levelname)s [%(filename)s:%(funcName)s:%(lineno)s] >> %(message)s"

def init_logger():
    absl_logging.get_absl_handler().setLevel(logging.FATAL)
    logging.basicConfig(
        level="INFO",
        format=RICH_FORMAT,
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logger = logging.getLogger("rich")
    return logger

def add_file_handler(log_path:str):
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(FILE_HANDLER_FORMAT))
    logger.addHandler(file_handler)


logger = init_logger()

LOG_INFO = logger.info
LOG_DEBUG = logger.debug
LOG_WARN = logger.warning
LOG_ERROR = logger.error
LOG_CRITICAL = logger.critical



if __name__ == "__main__":
    LOG_INFO("information test")
    LOG_DEBUG("debug test")
    LOG_WARN("warn test")
    LOG_ERROR("error test")
    LOG_CRITICAL("critical test")