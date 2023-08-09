# FILE: ML model for predicting forex prices


# DEPENDENCIES
import numpy as np
import tensorflow as tf
import shared as s
import data as d
import analysis as a


# FUNCTIONS
def init(x: np.ndarray,
         y: np.ndarray,
         epochs: int = 500,
         batch_size: int = 512,
         learning_rate: float = 0.001,
         dropout_percent: float = 0,
         regularization_strength: float = 0,
         neurons: int = 20,
         verbose: bool = True,
         tf_verbose: int = 0):

    # FUNCTION: initialize and train model

    # PARAM (required): x: ndarray: x values
    # PARAM (required): y: ndarray: y values
    # PARAM: epochs: int: number of epochs in each iteration of gradient descent
    # PARAM: batch_size: int: number of data points to process before updating model parameters
    # PARAM: learning_rate: float: Obvious.
    # PARAM: dropout_percent: float: what percent of neurons are dropped in dropout layers
    # PARAM: regularization_percent: float: regularization coefficient
    # PARAM: neurons: int: number of neurons in hidden layer
    # PARAM: verbose: bool: whether to print logs
    # PARAM: tf_verbose: int: verbose value for tensorflow functions

    # RETURN: model: trained tensorflow sequential model

    s.log(tag="model",
          content=f"Training model...",
          verbose=verbose)  # Log

    regularizer = tf.keras.regularizers.l2(regularization_strength)

    model = tf.keras.Sequential([  # Define model
        tf.keras.layers.Dense(x.shape[1], activation='sigmoid', input_shape=(x.shape[1],)),
        tf.keras.layers.Dropout(dropout_percent),
        tf.keras.layers.Dense(neurons, activation='sigmoid', kernel_regularizer=regularizer),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # Define optimizer
    loss = tf.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  # Compile

    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=tf_verbose)  # Train

    s.log(tag="model",
          content=f"Trained model.",
          verbose=verbose)  # Log

    return model


# TESTING
if __name__ == "__main__":

    INTERVALS_TEMP = [100, 200]  # TODO change
    LENGTH_TEMP = 200

    data = d.load(path="datasets/AUD_USD/3.csv")
    y = d.labels(data, length=LENGTH_TEMP)
    x = d.normalize(data=d.features(data=data,
                                    intervals=INTERVALS_TEMP))


    d.save(data=x, path="datasets/AUD_USD/testing/x.csv")
    d.save(data=y, path="datasets/AUD_USD/testing/y.csv")

    model = init(x=x, y=y, tf_verbose=1,  # TODO change
                 epochs=5000,
                 batch_size=512,
                 learning_rate=0.005,
                 neurons=20,
                 dropout_percent=0
                 )

    data1 = d.load(path="datasets/AUD_USD/1.csv", verbose=False)
    y1 = d.labels(data1, length=LENGTH_TEMP, verbose=False)
    x1 = d.normalize(data=d.features(data=data1,
                                     intervals=INTERVALS_TEMP, verbose=False), verbose=False)
    a.data_overview(x=x1, y=y1, raw=data1, model=model)

    data1 = d.load(path="datasets/AUD_USD/2.csv", verbose=False)
    y1 = d.labels(data1, length=LENGTH_TEMP, verbose=False)
    x1 = d.normalize(data=d.features(data=data1,
                                     intervals=INTERVALS_TEMP, verbose=False), verbose=False)
    a.data_overview(x=x1, y=y1, raw=data1, model=model)

    data1 = d.load(path="datasets/AUD_USD/3.csv", verbose=False)
    y1 = d.labels(data1, length=LENGTH_TEMP, verbose=False)
    x1 = d.normalize(data=d.features(data=data1,
                                     intervals=INTERVALS_TEMP, verbose=False), verbose=False)
    a.data_overview(x=x1, y=y1, raw=data1, model=model)

    data1 = d.load(path="datasets/AUD_USD/4.csv", verbose=False)
    y1 = d.labels(data1, length=LENGTH_TEMP, verbose=False)
    x1 = d.normalize(data=d.features(data=data1,
                                     intervals=INTERVALS_TEMP, verbose=False), verbose=False)
    a.data_overview(x=x1, y=y1, raw=data1, model=model)

    data1 = d.load(path="datasets/AUD_USD/5.csv", verbose=False)
    y1 = d.labels(data1, length=LENGTH_TEMP, verbose=False)
    x1 = d.normalize(data=d.features(data=data1,
                                     intervals=INTERVALS_TEMP, verbose=False), verbose=False)
    a.data_overview(x=x1, y=y1, raw=data1, model=model)