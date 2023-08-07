# FILE: Constant and variable definitions for all other files


# DEPENDENCIES
from datetime import datetime


# CONSTANTS
API_KEY = "014900ea22d31ab04fe6c120b522cf3e-9e0629096736f16aeb23db95f23aa057"  # Oanda API key
API_REST_BASE_URL = " https://api-fxpractice.oanda.com"  # Oanda REST API base URL
API_STREAM_BASE_URL = "https://stream-fxpractice.oanda.com"  # Oanda streaming API base URL

CURRENCY = "AUD_USD"  # ID of currency to track

BATCH_SIZE = 1000  # Size of batches for data collection
FEATURE_INTERVALS = [5, 10, 20]  # Intervals for features to be calculated
FILE_PATH = "datasets/collected.csv"
LABEL_SCOPE_LENGTH = 100  # Number of future data points to scan for profit


# VARIABLES
account_id = ""  # Account ID (for api module purposes)


# FUNCTIONS
def log(tag: str = "misc",
        content: str = "none",
        verbose: bool = True):

    # FUNCTION: Console print

    # PARAM: tag: str: The type of message being logged.
    # PARAM: content: str: The content of the log message.
    # PARAM: verbose: bool: Whether to print logs.

    # RETURN: (print): (print): The output message.

    if verbose:  # If allowed to print
        timestamp = str(datetime.now())  # Get current timestamp
        tag = tag.upper()  # Tag to uppercase
        result = f"[{tag}]\t({timestamp})\t{content}"  # Create output string
        print(result)  # Print the result
