# FILE: ML model for predicting forex prices


# DEPENDENCIES
import numpy as np
import tensorflow as tf
import shared as s


# FUNCTIONS
def init(x: np.ndarray,
         y: np.ndarray,
         epochs: int = 10000,
         batch_size: int = 2048,
         learning_rate: float = 0.001,
         dropout_percent: float = 0.1,
         regularization_strength: float = 0.02,
         neurons: int = 40,
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
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # Define optimizer
    loss = tf.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  # Compile

    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=tf_verbose)  # Train

    s.log(tag="model",
          content=f"Trained model.",
          verbose=verbose)  # Log

    return model

