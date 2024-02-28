import tensorflow as tf
import numpy as np
from simple_wordle import ENCODED_WORDS, State
from collections import deque
import random
import pickle
from multiprocessing import Pool

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

model_q.compile(
    optimizer=optimizer,
    loss=mean_squared_error,
)

model_qp.compile(
    optimizer=optimizer,
    loss=mean_squared_error,
)


# Gets the reward assuming the qp policy is taken
def policy_reward(state: State, print_result=False):
    state = state.copy()

    while not state.is_complete:
        inputs = np.array(
            [state.flattened_state_and_action(i) for i in range(len(ENCODED_WORDS))]
        )
        res = model_qp.predict(inputs, verbose=0)
        best_action = np.argmax(res)
        state.guess(ENCODED_WORDS[best_action])

    if print_result:
        print(state)

    return 25 if state.has_won else 0 + state.num_green()


sample_buffer = deque(maxlen=1000)

# print("Creating a sample buffer:")
# for i in range(1000):
#     state = State(np.random.randint(0, len(ENCODED_WORDS)))
#     print(i)

#     while not state.is_complete:
#         sample_buffer.append(state.copy())

#         inputs = np.array(
#             [state.flattened_state_and_action(i) for i in range(len(ENCODED_WORDS))]
#         )
#         res = model_qp.predict(inputs, verbose=0)
#         best_action = np.argmax(res)
#         state.guess(ENCODED_WORDS[best_action])

# random.shuffle(sample_buffer)

# with open("sample_buffer.pickle", "wb") as pickle_file:
#     pickle.dump(sample_buffer, pickle_file)

while True:
    with open("sample_buffer.pickle", "rb") as pickle_file:
        sample_buffer = pickle.load(pickle_file)

    print("Creating test dataset...")
    train, label = [], []
    reward_sum = 0
    for i in range(50):
        print(i)
        start_state = sample_buffer.pop()
        check_actions = np.random.choice(len(ENCODED_WORDS), 10)
        for action in check_actions:
            copy_state = start_state.copy()
            copy_state.guess(ENCODED_WORDS[action])

            reward = policy_reward(copy_state)
            train.append(start_state.flattened_state_and_action(action))
            label.append(reward)

    model_q.fit(
        np.array(train),
        np.array(label),
        batch_size=10,
        epochs=15,
    )

    print("Copying weights...")
    model_qp.set_weights(model_q.get_weights())

    print("Example Game:")
    policy_reward(State(np.random.randint(0, len(ENCODED_WORDS))), print_result=True)
