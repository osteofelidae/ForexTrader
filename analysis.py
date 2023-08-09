# FILE: Copypasta of my standard analysis file

# FILE: data analysis


# DEPENDENCIES
import numpy as np
import tensorflow as tf
import shared as s


# VARIABLES
t_balance = 0
t_crypto = 0
t_holding = False


# FUNCTIONS
def split_data(data: np.ndarray,
               index: int = 1):

    # FUNCTION: Copypasta dependency of data_overview()

    data_height, data_width = data.shape  # Shape of data

    x = data[:, :-1 * index].reshape(data_height, data_width - index)  # x vals
    y = data[:, -1 * index:].reshape(data_height, index)  # y vals

    return x, y  # Return result


def t_buy(point, units):

    # FUNCTION: test buy

    # PARAM: point: data point
    # PARAM: units: amount of currency to buy

    # RETURN: (global var) edit t_balance and t_crypto and t_holding

    global t_holding, t_balance, t_crypto

    t_holding = True
    t_balance -= units * point[1]
    t_crypto += units


def t_sell_all(point):
    # FUNCTION: test sell

    # PARAM: point: data point

    # RETURN: (global var) edit t_balance and t_crypto and t_holding

    global t_holding, t_balance, t_crypto

    t_holding = False
    t_balance += t_crypto * point[0]
    t_crypto = 0


def data_overview(x: np.ndarray,
                  y: np.ndarray,
                  raw: np.ndarray,
                  model: tf.keras.models.Sequential = None):

    # FUNCTION: provide an overview of data

    # PARAM: x: x values
    # PARAM: y: y values
    # PARAM: raw: raw dataset
    # PARAM: profit_index: up to how many examples to calculate profit (since it is slow)
    # PARAM: model: Sequential: trained tensorflow model

    # RETURN: (print) percent of each label

    global t_holding, t_balance, t_crypto

    count = [0.0, 0.0]  # Count features
    for point in y:
        count[int(point)] += 1
    total = sum(count)


    if model is not None:
        data = np.concatenate((x, y), axis=1)  # Combine x and y

        profit = data[data[:, -1] == 1]  # Get rows with profit label
        loss = data[data[:, -1] == 0]

        profit_x, profit_y = split_data(profit)  # Split dataset
        loss_x, loss_y = split_data(loss)

        _, profit_accuracy = model.evaluate(x=profit_x, y=profit_y, verbose=0)
        _, loss_accuracy = model.evaluate(x=loss_x, y=loss_y, verbose=0)
        _, overall_accuracy = model.evaluate(x=x, y=y)

        predictions = model.predict(x, verbose=0)

        count1 = 0
        last_price = 0.0
        for index in range(predictions.shape[0]):

            prediction = predictions[index]
            point = raw[index]

            profit = prediction[0] >= 0.5

            loss = prediction[0] < 0.5

            if profit and (not t_holding):

                t_buy(point=point, units=100)
                count1 = 0
                last_price = point[1]

            elif (loss and t_holding and point[0] > last_price):# or count1 >= s.LABEL_SCOPE_LENGTH:

                t_sell_all(point=point)

            elif loss and t_holding:

                count1 += 1

        t_sell_all(point=point)

        point_count = data.shape[0]

        print(f"Profit: {count[0] / total * 100}% Loss: {count[1] / total * 100}%")
        result = (f"Model accuracy: Profit {profit_accuracy * 100}% \t Loss {loss_accuracy * 100}% \t Overall {overall_accuracy * 100}% \t Profit ${t_balance} over {point_count} points")
        print(result)
        t_balance = 0
        t_crypto = 0
        t_holding = False
        return result


# MAIN
#data = d.d_import(path="datasets/train.csv")
#x, y = d.d_get(data=data)
#data_overview(x=x, y=y)