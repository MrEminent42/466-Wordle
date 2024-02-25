import numpy as np
from simple_colors import *
from collections import Counter

SAMPLE_SIZE = 100
IDENTITY = np.eye(26)


@np.vectorize
def letter_to_int(letter: str) -> int:
    return ord(letter.lower()) - ord("a") + 1


@np.vectorize
def int_to_letter(value: int) -> str:
    if value == 0:
        return "0"
    return chr(int(value + ord("a") - 1))


def decode_word(encoded_word: np.ndarray) -> str:
    assert encoded_word.shape == (
        5,
        26,
    ), f"Invalid encoded word shape: {encoded_word.shape}"

    return "".join(int_to_letter(encoded_word))


def encode_word(word: str) -> np.ndarray:
    assert len(word) == 5, "Invalid word size."
    return letter_to_int(list(word.lower()))


np.random.seed(0)
words = np.loadtxt("./answers.txt", dtype=str)
words = np.random.choice(words, SAMPLE_SIZE)

encoded_words = np.array([encode_word(word) for word in words])


class State:
    def __init__(self, word_index: int):
        self._goal_word = encoded_words[word_index]
        self._board_letters = np.zeros((6, 5))
        self._board_mask = np.zeros((6, 5, 2))
        self._guess_number = 0
        self._has_won = False

    def __repr__(self):
        res = "Goal word: " + "".join(int_to_letter(self._goal_word)) + "\nBoard:\n"
        for row in range(6):
            for col in range(5):
                letter = int_to_letter(self._board_letters[row, col])

                if self._board_mask[row, col, 0] != 0:
                    res += yellow(str(letter).upper())
                elif self._board_mask[row, col, 1] != 0:
                    res += green(str(letter).upper())
                else:
                    res += black(str(letter).upper())
            res += "\n"

        return res

    @property
    def is_complete(self):
        return self._guess_number >= 6 or self._has_won

    @property
    def has_won(self):
        return self._has_won

    def guess(self, word: np.ndarray):
        assert word.shape == (5,) and word.dtype == int, f"Invalid word: {word}"
        assert not self.is_complete, "Game already over."

        counter = Counter(self._goal_word)
        won = True

        for i in range(5):
            guess = word[i]
            count = counter[guess]
            if count > 0:
                if self._goal_word[i] == guess:
                    self._board_mask[self._guess_number, i][1] = 1
                else:
                    self._board_mask[self._guess_number, i][0] = 1

                counter[guess] -= 1

            else:
                won = False

        self._board_letters[self._guess_number] = word
        self._guess_number += 1
        self._has_won = won


test_state = State(1)
test_state.guess(encoded_words[4])
test_state.guess(encoded_words[14])
test_state.guess(encoded_words[0])
test_state.guess(encoded_words[1])
print(test_state)
