def debug(*info):
    p = False
    if p:
        print(info)


from enum import IntEnum
from simple_colors import *


class Color(IntEnum):
    GREY = GRAY = 0
    YELLOW = 1
    GREEN = 2


class Tile:
    def __init__(self, character: str, color: Color):
        self.char = character
        self.color = color


class Board:
    def __init__(self):
        self.board = []

    def append_row(self, row: list[Tile]):
        self.board.append(row)

    def get_row(self, i) -> list[Tile]:
        return self.board[i]

    def get_rows(self) -> list[list[Tile]]:
        return self.board

    def get_num_rows(self) -> int:
        return len(self.board)


class WordleGame:
    def __init__(self, answer):
        self.answer = answer.upper()
        self.board = Board()
        self.is_complete = False
        self.win = False

    ## string representation of the Wordle board
    ## returns with color too!
    def __repr__(self):
        s = ""
        # in each line
        for i, line in enumerate(self.board.get_rows()):
            colors = [tile.color for tile in line]
            for i, tile in enumerate(line):
                # for i, char in enumerate(line):

                if tile.color == Color.GREY:
                    s += black(tile.char, "bold")
                elif tile.color == Color.YELLOW:
                    s += yellow(tile.char, "bold")
                else:
                    s += green(tile.char, "bold")
            s += "\n"
        return s

    ## Takes a five-letter guess, records this guess on the game's board.
    ## Returns the array of Colors with each index corresponding to the color of the letter at that index in the guess
    def guess(self, guess):
        tiles = []
        if len(guess) != 5:
            raise ValueError(
                'Wordle guess must be a 5-letter word. Could not guess with word "'
                + guess
                + '".'
            )
        # convert everything to upper case
        guess = guess.upper()
        # debug print
        debug("Your guess:", guess)
        debug("The answer:", self.answer)
        colors = self.get_colors(guess)
        # log guess to board
        tiles = [Tile(guess[i], colors[i]) for i in range(5)]
        self.board.append_row(tiles)

        # check for game over
        if self.board.get_num_rows() >= 6:
            self.is_complete = True
        elif guess == self.answer:
            print("WIN!")
            self.is_complete = self.win = True

        # give back list of colors
        return colors

    ## get the colors of a word guess
    ## input: string
    ## output: list of Colors, each color corresponding to the
    ##         appropriate game color of the letter at that index
    def get_colors(self, guess: str):
        ## grey by default
        colors = [Color.GREY for i in range(len(guess))]

        # count # occurrences of each of the letters in the correct answer
        occurrences_remaining = {}
        for char in self.answer:
            if char in occurrences_remaining:
                occurrences_remaining[char] += 1
            else:
                occurrences_remaining[char] = 1

        ## appropriately color the letters

        ## greens first
        ## if the character is in the correct place
        for i, char in enumerate(guess):
            if self.answer[i] == char:
                colors[i] = Color.GREEN
                occurrences_remaining[char] -= 1
                debug("Green:", char)

        ## yellows next
        ## if the character is in the word, but in the wrong place
        for i, char in enumerate(guess):
            ## skip if already colored greeen
            ## skip if all occurrences of this letter have been accounted for
            if (
                colors[i] == Color.GREEN
                or char not in occurrences_remaining
                or occurrences_remaining[char] == 0
            ):
                continue

            colors[i] = Color.YELLOW
            debug("Yellow:", char)
            # record that we have accounted for this occurence
            occurrences_remaining[char] -= 1

        return colors

    def is_complete(self):
        return self.is_complete

    def run_game(self):
        print("Welcome to Wordle-AI!")
        while not self.is_complete:
            self.guess(input("Guess: "))
            print(self)
        if self.win:
            print(
                "Congrats! You found the word in", self.board.get_num_rows(), "tries."
            )
        else:
            print("Darn! You didn't find the word. It was " + self.answer + ".")
