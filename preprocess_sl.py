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

def extract_features(path):
	""" The main function which reads all training data, and extracts all features, then writes them (in matrix form) to an output file.
		To use, make sure to pass it the path of the directory containing the text files. 
		Of course, the path is relative to the location of this file. 
		It currently cannot navigate into subdirectories, so you may need to call several times for several folders.  
	"""
	#nlp = StanfordCoreNLP(r'\stanford-english-corenlp-2017-06-09-models')
	tknzr = TweetTokenizer()
	stop_words = set(stopwords.words('english'))

	for filename in os.listdir(path):
		file = open(path + "/" + filename, "r")

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

		content_depunct = list(filter(lambda token: token not in string.punctuation, content_tknzd))

		# De-punkt, stopwords removed, alphabetical words only 
		content_cleaned = list(filter(lambda token: token not in stop_words, content_depunct))
		content_cleaned = [token for token in content_cleaned if token.isalpha()]

		content_cleaned_lower = [token.lower() for token in content_cleaned]
		print(">>>>>>>>>>>>>>>>>>>>>", filename, ">>>>>>>>>>>>>>>>>>>>")
		#print(content_depunct)
		
		# Call your helper functions here! Use the raw, depunct, or cleaned content/subject. 
		avg_word_len = find_avg_word_length(content_cleaned_lower)
		freq_all_caps_words = count_all_caps_words(content_cleaned)
		num_numbered_lists = count_numbered_lists(email_content)
		print(avg_word_len)
		print(freq_all_caps_words)
		print(num_numbered_lists)

		file.close()

		
		
		# Do we need to consider case where email_content is empty? 

		

	# TO DO: return features in matrix form, write this to a file
	return None

def main():
	# Make sure you fill in the correct path 
	path = 'Data/test'
	features = extract_features(path)
	#count_all_caps_words("The following words are ALL CAPS FOREVER. The following word is in CAPS. 4DOT CATS 9 LIVE")

if __name__ == '__main__':
	main()