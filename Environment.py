from get_word_funcs import *
from word_functions import *
from Wordle import *

class Environment:
    reward_yellow = 0
    reward_green = 0
    reward_win = 10
    reward_lose = -10
    
    def __init__(self, answer = None):
        self.score = 0
        self.rewards = [0]
        self.total_words = get_words()
        self.wordle = Wordle(answer)
        self.num_guesses = 0
        self.state = np.zeros(443)
        self.state[0] = 6
        self.guess_to_state()
        self.action_space = np.array(range(len(self.total_words)))
        
    def reset(self, answer = None):
        self.score = 0
        self.rewards = [0]
        self.wordle.reset(answer)
        self.num_guesses = 0
        self.state = np.zeros(443)
        self.state[0] = 6
        self.guess_to_state()
        self.action_space = np.array(range(len(self.total_words)))
        

    def guess_to_state(self):
        """ 
        State is as follows:
        First position is number of guesses left
        Next 26 positions is whether letter has been attempted or not
        For positions 27-417, it is as follows:
        A: [No, Maybe, Yes] for first letter
        B: [No, Maybe, Yes] for first letter
        ...
        Until you go through all 5 letters of word.
        """
        if self.num_guesses == 0:
            self.state[0] = 6
            #Mark every letter in every position as maybe
            for i in range(130):
                self.state[53+1+3*i] = 1
        elif self.num_guesses > 0:
            self.state[0] = 6 - self.num_guesses
            guess = self.wordle.guesses[-1]
            hint = self.wordle.hints[-1]
            
            #No pass through
            for i in range(5):
                val = ord(guess[i]) - ord('a')
                if hint[i] == 0:
                    for j in range(5):
                        self.state[53 + 26*3*j + 3*val] = 1
                        self.state[53 + 26*3*j + 3*val+1] = 0
                        self.state[53 + 26*3*j + 3*val+2] = 0
                if hint[i] == 1:
                    self.state[27 + val] = 1
                    self.state[53 + 26*3*i + 3*val] = 1
                    self.state[53 + 26*3*i + 3*val + 1] = 0
            #Yes Pass Through
            for i in range(5):
                val = ord(guess[i]) - ord('a')
                self.state[1+val] = 1
                #If green for a certain letter, make sure all other letters cannot be in that place.
                if hint[i] == 2:
                    self.state[27 + val] = 1
                    for j in range(26):
                        self.state[53 + 26*3*i + 3*j + 1] = 0
                        self.state[53 + 26*3*i + 3*j] = 1
                    self.state[53 + 26*3*i + 3*val + 2] = 1
                    self.state[53 + 26*3*i + 3*val] = 0
                
        
    def step(self, action):
        self.num_guesses += 1
        guess, hint = self.wordle.guess(self.total_words[action])
        self.guess_to_state()
        reward = 0
        for i in range(len(hint)):
            if hint[i] == 1:
                reward+= self.reward_yellow
            elif hint[i] == 2:
                reward += self.reward_green
        if self.wordle.win:
            #Reward more for faster wins
            reward += (7 - self.num_guesses)*self.reward_win
        elif not self.wordle.win and self.wordle.over:
            reward += self.reward_lose
        if self.wordle.win and self.num_guesses == 1:
            reward = 0
        self.rewards.append(reward)
        state = self.state
        over = self.wordle.over
        return state, reward, over