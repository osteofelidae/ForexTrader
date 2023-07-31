# FILE: Data operations


# DEPENDENCIES
import numpy as np
import shared as s


# FUNCTIONS
def export(data: np.ndarray,
           path: str = s.FILE_PATH,
           verbose: bool = True):

    # FUNCTION: Save data to file

    # PARAM: data: ndarray: Data to save
    # PARAM: path: str: File path to save to
    # PARAM: verbose: bool: Whether to print logs

    s.log(tag="data",
          content=f"Saving dataset to '{path}'...",
          verbose=verbose)  # Log

    np.savetxt(path, data, delimiter=",")  # Save as csv

    s.log(tag="data",
          content=f"Saved dataset to '{path}'.",
          verbose=verbose)  # Log