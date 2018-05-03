import pandas as pd
import nltk as nltk
from nltk import RegexpTokenizer
from nltk import TweetTokenizer
import os
import numpy as np
import re
import string
from nltk.corpus import stopwords
#from stanfordcorenlp import StanfordCoreNLP

def count_non_english_words(tokens):
    return len([token for token in tokens if not token.isalpha()])

def count_words(tokens):
    return len(tokens)

def count_longest_seq(tokens):
    max_length = 0
    length = 0
    for token in tokens:
        if not token.isalpha():
            if length > max_length:
                max_length = length
            length = 0
        else:
            length += len(token)
    return max_length

def count_num_chars(tokens):
    return sum([len(token) for token in tokens])