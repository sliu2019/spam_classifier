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

def count_all_caps_words(text):
    # Returns frequency of all caps words, not COUNT, as the function name says
    # Observation: technical ham emails can have a lot of acronyms. Also, some HTML makes things caps, but this can't detect it.
    return sum(token.isupper() for token in text) / len(text)

def find_avg_word_length(text):
    # Assumes input text is tokenized, and a list
    # Sum is implemented in C, and fast
    return sum(len(token) for token in text) / len(text)

def count_numbered_lists(text):
    # Runs on raw text

    count = 0
    # Matches "1." and "1)" type numbering
    matches = re.findall(r'(?:\d+(?:\.|\))\s.*\n?)+', text) #returns a list of strings
    for match in matches:
        if match[0] == "1":
            count += 1
            #print(match)
    return count

