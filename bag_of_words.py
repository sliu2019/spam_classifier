import os
import numpy as np
import re
import pandas as pd
import nltk as nltk
from nltk import RegexpTokenizer
from nltk import TweetTokenizer
import os
import re
import string
from nltk.corpus import stopwords
def generate_matrix(path, name, key_words, ham_flag):
    key_word_set = set(key_words)
    key_word_index = {}
    for i, word in enumerate(key_word_set):
        key_word_index[word] = i

    tknzr = TweetTokenizer()
    stop_words = set(stopwords.words('english'))
    X = np.zeros((1, len(key_word_set)))
    count = 0
    for filename in os.listdir(path):
        try:
            frequency = [0] * len(key_word_set)
            file = open(path + "/" + filename, "r", encoding='cp1252')
            email_content = ""
            email_subject = ""
            for line in file:
                m = re.search('^Subject:', line)
                if m:
                    m_span = m.span()
                    email_subject = line[m_span[1] + 1: -1]

                # We assume that a line break denotes where the email begins
                if line == "\n":
                    email_content = file.read()
                    break
            content_tknzd = tknzr.tokenize(email_content)
            subject_tknzd = tknzr.tokenize(email_subject)

            # De-punkt, stopwords removed, alphabetical words only
            content_cleaned = [token for token in content_tknzd if not token in stop_words]


            content_cleaned_lower = [token.lower() for token in content_cleaned]
            for token in content_cleaned_lower:
                if token in key_word_set:
                    frequency[key_word_index[token]] += 1
            freq_vector = np.array([frequency])
            X = np.vstack((X, freq_vector))
            count += 1
        except UnicodeDecodeError:
            continue
    if ham_flag:
        y = np.ones((count, 1))
    else:
        y = np.zeros((count, 1))
    X = X[1:, :]
    np.savez(name, X, y)
    print(path)

def main():
    import csv
    with open("key_words.csv", "r") as inputFile:
        reader = csv.reader(inputFile)
        key_words = next(reader)
    path = 'Data/easy_ham'
    generate_matrix(path, "ham_easy", key_words, True)
    path = 'Data/hard_ham'
    generate_matrix(path, 'ham_hard', key_words, True)
    path = 'Data/spam'
    generate_matrix(path, 'spam', key_words, False)

if __name__ == '__main__':
    main()