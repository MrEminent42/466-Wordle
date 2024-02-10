import tensorflow as tf
import numpy as np
from collections import deque
from Wordle import *

game = WordleGame("Hello")

actions = np.loadtxt("./actions.txt", dtype=str)
answers = np.loadtxt("./answers.txt", dtype=str)


def rowToOneHot(row: list[str]):
    oneHot = np.zeros((5, 26))
    for i, char in enumerate(row):
        if char != None:
            oneHot[i][ord(char) - ord("A")] = 1
    return oneHot


def boardToState(board: Board):
    letters = np.zeros((6, 5, 26))
    colors = np.zeros((6, 5, 2))

    for i, row in enumerate(board.get_rows()):
        for j, tile in enumerate(row):
            letters[i][j][[ord(tile.char) - ord("A")]] = 1
            if tile.color == Color.GREEN:
                colors[i][j][1] = 1
            elif tile.color == Color.YELLOW:
                colors[i][j][0] = 1
    return np.concatenate((letters.flatten(), colors.flatten()))


def getReward(colors: list[Color]):
    reward = 0
    win = True
    for color in colors:
        if color == Color.GREEN:
            reward += 1.5
        elif color == Color.YELLOW:
            reward += 1.0
            win = False
        else:
            reward += 0.5
            win = False
    if win:
        reward += 5
    return reward


def step(action: int):
    ## verify action is legit
    if action < 0 or action >= len(actions):
        raise ValueError("Action out of bounds")

    ## take the action (guess)
    guess = actions[action]
    colors = game.guess(guess)
    ## what we need: next_state, reward, done, info (not used)
    reward = getReward(colors)
    next_state = boardToState(game.board)
    done = game.is_complete
    return next_state, reward, done, None


class ReplayBuffer:
    """Experience replay buffer that samples uniformly."""

    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        indexes = np.random.choice(len(self.buffer), num_samples)

        for i in indexes:
            state, action, reward, next_state, done = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )


print("Loaded", len(actions), "actions;", len(answers), "answers.")


model_q = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(840,)),  # 840 inputs
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

model_qp = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(840,)),  # 840 inputs
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

# model_qp = model_q.cop


def select_epsilon_greedy_action(epsilon: float):
    """Take random action with probability epsilon, else take best action."""
    result = np.random.uniform(0, 1)
    if result < epsilon:
        # get random action from actions
        action_ind = np.random.randint(0, len(actions))
        debug("Selected action:", actions[action_ind], "(index", str(action_ind) + ")")
        return actions[action_ind]  # Random action (left or right).

    else:
        # run all possible guesses through the model and select the best one
        best_action_ind = None
        best_q = -np.inf
        for action_ind in range(len(actions)):
            state = boardToState(game.board)
            # q: copilot, why did you expand_dims?
            # state = np.expand_dims(state, axis=0)
            # a: to make the input shape (1, 840) instead of (840,)
            q = model_qp.predict(state)
            if q > best_q:
                best_q = q
                best_action_ind = action_ind
        return best_action_ind


num_episodes = 1000
epsilon = 1.0
batch_size = 32
discount = 0.99
buffer = ReplayBuffer(100000)
cur_frame = 0


last_100_ep_rewards = []
