# FILE: Decision tree classifier


# DEPENDENCIES
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import shared as s
import data as d


# FUNCTIONS
def init(x: np.ndarray,
         y: np.ndarray,
         trees: int = 10,
         seed: int = 12,
         verbose: bool = True):

    # FUNCTION: Initialize random forest tree

    # PARAM: trees: int: Number of decision trees to train
    # PARAM: seed: float: Seed for random number generator
    # PARAM: verbose: bool: Whether to print logs

    # TODO RETURN

    y = np.ndarray.flatten(y)  # Make into 1d array

    ensemble = RandomForestClassifier(n_estimators=trees, random_state=seed)  # Initialize model
    ensemble.fit(x, y)  # Train

    return ensemble  # Return result


# TESTING
if __name__ == "__main__":

    INTERVALS_TEMP = [10, 100]  # TODO change
    LENGTH_TEMP = 200
    BATCH_NORM_TEMP = 200

    data = d.load(path="datasets/AUD_USD/3.csv")
    y = d.labels(data, length=LENGTH_TEMP)
    x = d.batch_norm(data=d.features(data=data,
                                     intervals=INTERVALS_TEMP),
                     batch=BATCH_NORM_TEMP)

    model = init(x=x, y=y)

    for i in range(1, 5):  # Use (1, 6) for last huge dataset
        data = d.load(path=f"datasets/AUD_USD/{i}.csv")
        y = d.labels(data, length=LENGTH_TEMP)
        x = d.batch_norm(data=d.features(data=data,
                                         intervals=INTERVALS_TEMP),
                         batch=BATCH_NORM_TEMP)

        y_pred = model.predict(x)

        accuracy = accuracy_score(y, y_pred)
        print(accuracy)

        #a.data_overview(x=x, y=y, raw=data, model=model)