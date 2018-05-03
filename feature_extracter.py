
from bs4 import BeautifulSoup as BS
from nltk.stem import WordNetLemmatizer
from nltk import tokenize
import string

def word_counts(html):
    lem = WordNetLemmatizer()
    tokens = tokenize(BS(html).get_text())
    tokens = [lem.lemmatizer(token.lowercase) for token in tokens if token not in string.punctuation]

    return make_dict(tokens)

def make_dict(tokens):
    dct = {}
    for token in tokens:
        if token not in dct.keys():
            dct[token] = 0
        dct[token] += 1
    return dct

def link_count(html):
    soup = BS(html)
    return len(soup.find_all('a'))

def img_count(html):
    soup = BS(html)
    return len(soup.find_all('img'))

def junk_count(html):
    tokens = tokenize(BS(html).get_text())
    tokens = [token for token in tokens if not token.isalpha() and token not in string.punctuation]
    return len(tokens)


def word_count(html):
    tokens = tokenize(BS(html).get_text())
    return len([token for token in tokens if token not in string.punctuation])

def doc_length(html):
    tokens = tokenize(BS(html).get_text())
    return sum([len(token) for token in tokens if token not in string.punctuation])

def longest_word(html):
    tokens = tokenize(BS(html).get_text())
    return max([len(token) for token in tokens])

def ave_word_length(html):
    count = word_count(html)
    if count > 0:
        return doc_length(html)/count
    return 0

def punc_count(html):
    tokens = tokenize(BS(html.get_text()))
    count = 0
    for token in tokens:
        if len(token) > 1:
            for char in token:
                if char in string.punctuation:
                    count +=1
                    break

    return count

def test(html):
    print("Word counts:", word_counts(html))
    print("Link count:", link_count(html))
    print("Img count:", img_count(html))
    print("Junk count:", junk_count(html))
    print("Total word count:", word_count(html))
    print("Doc length", doc_length(html))
    print("Longest Word:", longest_word(html))
    print("Ave word length:", ave_word_length(html))
    print("Punc count:", punc_count(html))


