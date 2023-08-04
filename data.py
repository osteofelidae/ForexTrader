# FILE: Data operations


# DEPENDENCIES
import numpy as np
import shared as s


# FUNCTIONS
def save(data: np.ndarray,
         path: str = s.FILE_PATH,
         verbose: bool = True):

    # FUNCTION: Save data to file

    # PARAM: data: ndarray: Data to save
    # PARAM: path: str: File path to save to
    # PARAM: verbose: bool: Whether to print logs

    # RETURN (file): Writes to 'path'

    s.log(tag="data",
          content=f"Saving dataset to '{path}'...",
          verbose=verbose)  # Log

    np.savetxt(path, data, delimiter=",")  # Save as csv

    s.log(tag="data",
          content=f"Saved dataset to '{path}'.",
          verbose=verbose)  # Log


def load(path: str = s.FILE_PATH,
         verbose: bool = True):

    # FUNCTION: Load ndarray from csv.

    # PARAM: path: str: Path of file
    # PARAM: verbose: bool: Whether to print logs

    # RETURN: data: ndarray: loaded dataset

    s.log(tag="data",
          content=f"Loading dataset from '{path}'...",
          verbose=verbose)  # Log

    data = np.genfromtxt(path, delimiter=",")  # Load dataset

    s.log(tag="data",
          content=f"Loaded dataset from '{path}'.",
          verbose=verbose)  # Log

    return data  # Return dataset


def scope(data: np.ndarray,
          length: int,
          index: int):

    # FUNCTION: get in scope items

    # PARAM: data: ndarray: dataset
    # PARAM: length: int: length of in-scope items
    # PARAM: index: int: index where data becomes out of scope (ending index)

    # OUT: scoped: ndarray: in-scope items

    index = index if index >= 0 else index + len(data)  # Deal with negative indexes (indexing from end)

    start = max([0, index - length])  # Starting index
    end = index  # Ending index

    scoped = data[start:end]  # In-scope items

    return scoped  # Return result


def features(data: np.ndarray,
             intervals: list[int] = s.FEATURE_INTERVALS,
             verbose: bool = True):

    # FUNCTION: Do feature engineering

    # PARAM (required): data: ndarray: dataset
    # PARAM: intervals: list[int]: intervals at which to apply feature engineering
    # PARAM: verbose: bool: whether to print logs

    # RETURN: engineered: ndarray: ndarray with feature engineering applied

    s.log(tag="data",
          content=f"Computing features...",
          verbose=verbose)  # Log

    select = [0, 1]  # Columns to select
    raw = data[:, select]  # Clone data, filtering by given columns

    new = []  # Temp arr to add calculated values to
    for interval in intervals:  # For each interval
        matrix = []  # Temp arr to add rows to
        for row_i in range(raw.shape[0] - 1):  # For each row index except the first
            row_i_t = row_i + 1
            scoped = scope(data=raw, length=interval, index=row_i_t)  # Get in-scope items

            maxs = np.max(scoped, axis=0)  # Get maxes of all columns
            mins = np.min(scoped, axis=0)  # Get mins of all columns
            means = np.mean(scoped, axis=0)  # Get means of all columns
            stds = np.std(scoped, axis=0)  # Get standard deviation

            uppers = means + (2 * stds)  # Upper bollinger band
            lowers = means - (2 * stds)  # Lower bollinger band

            spot = scoped[:, 0]  # Spot prices
            up = 1e-8  # Total profit: set to small number epsilon
            down = 1e-8  # Total loss
            for ind in range(len(spot) - 1):  # Iterate over length of spot array
                if spot[ind] < spot[ind + 1]:  # If profit
                    up += spot[ind + 1] - spot[ind]  # Add to profit total
                elif spot[ind] > spot[ind + 1]:  # If loss
                    down += spot[ind] - spot[ind + 1]  # Add to loss total
            rs = up / down  # Relative strength
            rsi_0 = [(100 - (100 / (1 + rs)))/100]  # Relative strength index for index 0

            spot = scoped[:, 1]  # Spot prices
            up = 1e-8  # Total profit: set to small number epsilon
            down = 1e-8  # Total loss
            for ind in range(len(spot) - 1):  # Iterate over length of spot array
                if spot[ind] < spot[ind + 1]:  # If profit
                    up += spot[ind + 1] - spot[ind]  # Add to profit total
                elif spot[ind] > spot[ind + 1]:  # If loss
                    down += spot[ind] - spot[ind + 1]  # Add to loss total
            rs = up / down  # Relative strength
            rsi_1 = [(100 - (100 / (1 + rs)))/100]  # Relative strength index for index 1

            lag = scoped[0]  # Lagged feature

            row = np.hstack(
                (maxs, mins, means, stds, uppers, lowers, rsi_0, rsi_1, lag))  # Make all calculated values into new row
            matrix.append(row)
        matrix = np.array(matrix)  # Turn matrix into numpy array
        new = np.hstack((new, matrix)) if len(new) > 0 else matrix

    engineered = np.hstack((raw[1:], new))  # Add new values to the side of raw array

    s.log(tag="data",
          content=f"Computed features.",
          verbose=verbose)  # Log

    return engineered  # Return result


# TODO labels
def labels():

    # FUNCTION: Add labels to data

    # TODO RETURN

    # TODO EVERYTHING


# TESTING
if __name__ == "__main__":

    print(features(load(path="datasets/AUD_USD/3.csv")))
