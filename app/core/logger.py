import logging
import sys


logger = logging.getLogger("ContextTreeAPILogger")
logger.setLevel(logging.INFO)


# handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create a console handler (stdout)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)


file_handler = logging.FileHandler("app.log")
# file_handler = logging.FileHandler("~/logs/app.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)
