import numpy as np
import pandas as pd
import os
import time
from math import exp

'''
get_words retrieves the word lists, gathered from the source code at https://www.nytimes.com/games/wordle/index.html
return_all_lists = True returns the following three arrays:
acceptable_words: The list of words which wordle accepts as an answer, but are not the answer
answer_words: The list of words which are possibly the answer, disjoit from acceptable_words
total_words: The concatenation of answer_words with acceptable_words'''
def get_words(return_all_lists = False):
    with open('answer_words.txt', 'r') as f:
        contents = f.read()
        words = contents.split('"')
        word1 = []
        for word in words:
            if word != ',':
                word1.append(word)
        answer_words = word1[1:-1]
    with open('acceptable_words.txt', 'r') as f:
        contents = f.read()
        words = contents.split('"')
        word1 = []
        for word in words:
            if word != ',':
                word1.append(word)
        acceptable_words = word1[1:-1]
    total_words = answer_words + acceptable_words
    total_words = np.array(total_words)
    answer_words = np.array(answer_words)
    acceptable_words = np.array(acceptable_words)
    if not return_all_lists:
        return total_words
    else:
        return total_words, answer_words, acceptable_words
'''
get_frequencies uses the dataset from https://www.kaggle.com/rtatman/english-word-frequency
this function retrieves the data, only keeps the words that are found in total_words, does a transformation
of the value counts as a weight for the 3b1b expected information strategy found at https://www.youtube.com/watch?v=v68zYyaEmEA&t=0s
'''
#The following code was hastily put together
def logistic(x):
    return 1/(1+exp(-x))

def get_freqencies():
    if not os.path.exists('wordfrequencylist.npy'):
        total_words = get_words()
        df = pd.read_csv('wordfreqkaggle.csv')
        freq_words = df['word']
        freq_counts = df['count']
        word_index = []
        for i in range(len(freq_words)):
            if type(freq_words[i]) != str:
                continue
            if len(freq_words[i]) != 5:
                continue
            if freq_words[i] not in total_words:
                continue
            word_index.append(i)
        freq_words = np.array(freq_words[word_index])
        freq_counts = np.array(freq_counts[word_index])
        arr = min(freq_counts)*np.ones(len(total_words), dtype=np.float32)
        for i in range(len(freq_words)):
            index = np.where(total_words == freq_words[i])[0][0]
            if index == 0:
                print(freq_words[i], freq_counts[i])
            arr[index] = freq_counts[i]
        arr.resize((len(arr), 1))
        arr = np.log10(arr)
        arr = arr - min(arr)
        arr = arr - (max(arr)-min(arr))/2
        arr = np.array([logistic(2*x) for x in arr])
        arr.resize((len(arr), 1))
        np.save('wordfrequencylist.npy', arr)
        freq = arr
    else:
        freq = np.load('wordfrequencylist.npy')
    return freq

def one_hot_words():
    total_words = get_words()
    if os.path.exists('one_hot_total_words.npy'):
        return np.load('one_hot_total_words.npy')
    one_hot_total_words = np.zeros((len(total_words), 130),dtype = np.uint8)
    for k in range(len(total_words)):
        word = total_words[k]
        for i in range(5):
            letter_index = 26*i + ord(word[i]) - ord('a')
            one_hot_total_words[k, letter_index] = 1
    np.save('one_hot_total_words', one_hot_total_words)
    return one_hot_total_words
    
def get_wordle_performance():
    if os.path.exists('Data/3b1b_performance.npy'):
        player_win_turns = np.load('Data/3b1b_performance.npy')
    else:
        player_win_turns = []
        player_win_score = 0
        env = Environment()
        p = Player()
        for answer in tqdm(answer_words):
            env.reset(answer)
            p.reset()
            while not env.wordle.over:
                action = np.where(total_words == p.exp_info_strategy())[0][0]
                env.step(action)
                p.guess(total_words[action], env.wordle.hints[-1])
            player_win_score += env.wordle.win
            player_win_turns.append(env.num_guesses)
        player_win_turns = np.array(player_win_turns)
        np.save('Data/3b1b_performance.npy', player_win_turns)
    return player_win_turns

