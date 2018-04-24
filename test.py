
from bs4 import BeautifulSoup as BS
from nltk.stem import WordNetLemmatizer
import string

def word_counts(html):
	lem = WordNetLemmatizer()
	tokens = word_tokenize(BS(html).get_text())
	tokens = [lem.lemmatizer(token.lowercase) in tokens if token not in string.punctuation]

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
	tokens = word_tokenize(BS(html).get_text())
	tokens = [token in tokens if not token.isalpha() and token not in string.punctuation]
	return len(tokens)


def word_count(html):
	tokens = word_tokenize(BS(html).get_text())
	return len([token in tokens if token not in string.punctuation])

def doc_length(html):
	tokens = word_tokenize(BS(html).get_text())
	return sum([len(token) in tokens if token not in string.punctuation])

def longest_word(html):
	tokens = word_tokenize(BS(html).get_text())
	return max([len(token) in tokens])

def ave_word_length(html):
	count = word_count(html)
	if count > 0:
		return doc_length(html)/count
	return 0

def punc_count(html):
	tokens = word_tokenize(BS(html.get_text()))
	count = 0
	for token in tokens:
		if len(token) > 1:
			for char in token:
				if char in string.punctuation:
					count +=1
					break

	return count

