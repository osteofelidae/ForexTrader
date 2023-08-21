# FILE: Sequential evaluation using random forests


# DEPENDENCIES
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import shared as s
import data as d
import trees as t


# TESTING
if __name__ == "__main__":

    INTERVALS_TEMP = [10, 100]  # TODO change
    LENGTH_TEMP = 200
    BATCH_RANGE = 500
    BATCH_NORM_TEMP = BATCH_RANGE
    TRAIN_RANGE = 1000

    data = d.load(path="datasets/AUD_USD/3.csv")
    y = d.labels(data, length=LENGTH_TEMP)
    x = d.batch_norm(data=d.features(data=data,
                                     intervals=INTERVALS_TEMP),
                     batch=BATCH_NORM_TEMP)

    correct = 0.0
    count = 0.0

    for index_t in range(len(x)-2):
        index = index_t + 1
        x_t = d.scope(data=x, index=index, length=BATCH_RANGE)
        y_t = d.scope(data=y, index=index, length=BATCH_RANGE)

        if index_t % TRAIN_RANGE == 0:
            model = t.init(x=x_t, y=y_t)

        if index_t > TRAIN_RANGE:

            prediction = model.predict([x[index + 1]])
            if prediction == y[index + 1]:
                correct += 1
            count += 1
            print(correct/count)
