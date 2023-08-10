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


def f_scope(data: np.ndarray,
            length: int,
            index: int):

    # FUNCTION: get in scope items to the front

    # PARAM: data: ndarray: dataset
    # PARAM: length: int: length of in-scope items
    # PARAM: index: int: index where data becomes out of scope (ending index)

    # OUT: scoped: ndarray: in-scope items

    index = index if index >= 0 else index + len(data)  # Deal with negative indexes (indexing from end)

    start = index  # Starting index
    end = min([data.shape[0], index + length])  # Ending index

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


def labels(data: np.ndarray,
           length: int = s.LABEL_SCOPE_LENGTH,
           margin: float = s.PROFIT_MARGIN,
           verbose: bool = True):

    # FUNCTION: Add labels to data

    # PARAM: data: ndarray: dataset
    # PARAM: length: int: Number of future points to scan for profit
    # PARAM: margin: float: Multiplier of profit margin
    # PARAM: verbose: bool: Whether to print logs

    # RETURN: labelled: ndarray: Labels without data

    labelled = []  # Array of labels

    for index_f in range(data.shape[0]-1):  # Iterate over data

        index = index_f + 1

        scoped = f_scope(data=data, index=index, length=length)  # Scope future items

        bids = scoped[:, 0]  # Bid prices
        asks = scoped[:, 1]  # Ask prices

        min_ask = np.min(asks)  # Minimum ask price in scope
        current_bid = bids[0]  # Current bid price in scope

        condition = min_ask * margin < current_bid  # Condition for labels

        if condition:  # If profitable
            labelled.append([1])  # Add 1
        else:  # If not
            labelled.append([0])  # Add 0

    labelled = np.array(labelled)  # Convert to numpy array

    return labelled  # Return result except first term, to match with features


def normalize(data: np.ndarray,
              verbose: bool = True):

    # FUNCTION: normalize data

    # PARAM: data: ndarray: dataset
    # PARAM: verbose: bool: whether to print logs

    # RETURN: normalized: ndarray: normalized dataset

    # TODO verbose

    mean = np.mean(data[:,0])  # Array of means
    std = np.std(data, axis=0)  # Array of standard deviations

    normalized = (data - mean)# / ( std)  # Calculate normalized array

    return normalized  # Return calculated result


def batch_norm(data: np.ndarray,
               batch: int = s.BATCH_SIZE,
               verbose: bool = True):

    # FUNCTION: normalize data in batches

    # PARAM: data: ndarray: Dataset
    # PARAM: batch: int: Batch size
    # PARAM: verbose: bool: Whether to print logs

    # RETURN: normalized: ndarray: Normalized dataset

    s.log(tag="data",
          content=f"Batch normalizing...",
          verbose=verbose)  # Log

    normalized = []
    for row_i in range(len(data)):  # Iterate over row indexes

        scoped = scope(data=data, index=row_i, length=batch) if row_i != 0 else [data[0]]  # Scope items
        means = np.mean(scoped, axis=0)  # Means
        stds = np.std(scoped, axis=0) + 1e-8  # Array of standard deviations
        normalized.append((data[row_i] - means)/(2 *stds))  # Normalize and append result

    s.log(tag="data",
          content=f"Batch normalized.",
          verbose=verbose)  # Log

    return np.array(normalized)  # Return result


# TESTING
if __name__ == "__main__":
    data = load(path="datasets/AUD_USD/3.csv")
    featured = features(data=data)
    labelled = labels(data)
    print(batch_norm(data=featured))
    #print(len(labelled[labelled == 1]),len(labelled[labelled == 0]))
    #print(f_scope(data=data, length=10, index=-2))
