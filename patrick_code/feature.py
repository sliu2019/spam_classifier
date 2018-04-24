import numpy as np 
import string
import re 
from feature_util import Trie, Trie_Node
def word_counter(title, content, list_of_words, frequency = True, normalize = True):
	assert(len(title) != 0 and len(content) != 0)
	title = title.translate(title.maketrans("", "", string.punctuation)).lower()
	content = content.translate(content.maketrans("", "", string.punctuation)).lower()
	title_tokens = re.split(" ", title)
	content_tokens = re.split(" ", content)
	title_vec = np.zeros(len(list_of_words))
	content_vec = np.zeros(len(list_of_words))
	indexes = {}
	i = 0
	for word in list_of_words:
		indexes[word] = i
		i += 1
	print(title_tokens)
	print(content_tokens)
	set_of_words = set(list_of_words)
	title_length = 0
	for token in title_tokens:
		title_length += len(token)
		if token in set_of_words:
			if frequency:
				title_vec[indexes[token]] += 1
			else:
				title_vec[indexes[token]] = 1
	content_length = 0
	for token in content_tokens:
		content_length += len(token)
		if token in set_of_words:
			if frequency:
				content_vec[indexes[token]] += 1
			else:
				content_vec[indexes[token]] = 1
	if normalize:
		title_vec = title_vec / title_length
		content_vec = content_vec / content_length
	return title_vec, content_vec

def cap_counter(title, content, normalize = True):
	assert(len(title) != 0 and len(content) != 0)
	title = title.translate(title.maketrans("", "", string.punctuation + " "))
	content = content.translate(content.maketrans("", "", string.punctuation + " "))
	output_vec = np.zeros(2)
	count = 0
	for i in title:
		if i.isupper():
			count += 1
		else:
			if output_vec[0] < count:
				output_vec[0] = count
			count = 0
	if output_vec[1] < count:
		output_vec[1] = count
	count = 0 
	for i in content:
		if i.isupper():
			count += 1
		else:
			if output_vec[1] < count:
				output_vec[1] = count
			count = 0
	if output_vec[1] < count:
		output_vec[1] = count
	if normalize:
		output_vec[0] = output_vec[0]/len(title)
		output_vec[1] = output_vec[1]/len(content)
	return output_vec

def bag_of_words_counter(title, content, bags_of_words, frequency = True, normalize = True):
	raise NotImplementedErrror()

def test_word_counter():
	title = "Greetings, get your free money!"
	content = "call 800-12345 to claim your free car and free XboX one!"
	key_words = ["money", "free"]
	t, c = word_counter(title, content, key_words)
	print(t)
	print(c)

def test_cap_counter():
	title = "War Recruitment"
	content = "do you knOW WHAT IT TAKES TO BE A GREAT CHAMP!!!! JOIN OUR BATTLE NOW"
	o = cap_counter(title, content)
	print(o)




test_word_counter()
test_cap_counter()