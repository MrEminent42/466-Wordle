import tensorflow as tf
import numpy as np
from simple_wordle import ENCODED_WORDS, State

# Create the two model structures
model_q = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(970, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

model_qp = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(970, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

model_q.build((None, 970))
model_qp.build((None, 970))

model_qp.set_weights(model_q.get_weights())

# Define the optimizer and error
optimizer = tf.keras.optimizers.Adam(1e-4)
mean_squared_error = tf.keras.losses.MeanSquaredError()


# Gets the reward assuming the qp policy is taken
def policy_reward(state: State):
    if state.is_complete:
        return


print(create_input(State(0), 0))
