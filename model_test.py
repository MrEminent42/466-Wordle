from typing import Literal, Tuple

import numpy as np

# import tensorflow as tf


def flatten(letters: np.ndarray, success: np.ndarray):
    assert letters.shape == (6, 5, 26) and success.shape == (
        6,
        5,
        2,
    ), "Invalid input shape."
c

# model = tf.keras.models.Sequential([])
