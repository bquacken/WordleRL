import numpy as np
import pandas as pd

from get_word_funcs import *
from word_functions import *


class Wordle:
    def __init__(self, answer = None):
        self.total_words, self.answer_words, self.acceptable_words = get_words(return_all_lists = True)
        if answer == None:
            self.answer = np.random.choice(self.answer_words)
        else:
            self.answer = answer
        self.guesses = []
        self.hints = []
        self.nguesses = 6
        self.nletters = 5
        self.guess_count = 0
        self.over = False
        self.win = 0
    
    def reset(self, answer = None):
        self.guess_count = 0
        self.guesses = []
        self.hints = []
        self.over = False
        self.win = 0
        if answer == None:
            self.answer = np.random.choice(self.answer_words)
        else:
            self.answer = answer
        
    def guess(self, word):
        if len(word) != self.nletters or word not in self.total_words:
            print('Invalid Guess, Try Again!')
            return [word, similarity_value(word, self.answer)]
        self.guess_count += 1
        self.guesses.append(word)
        hint = list(similarity_value(word, self.answer))
        self.hints.append(hint)
        if word == self.answer:
            self.win = 1
            self.over = True
            return [word, hint]
        else:
            if self.guess_count == 6:
                self.over = True
            return word, hint
