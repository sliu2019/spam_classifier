import pandas as pd 
import nltk as nltk
from nltk import TweetTokenizer
import os
import numpy as np 
import re

def count_all_caps_words(text):
	#Only captures all caps words that lie between 2 word boundaries. i.e. it would not pick up on "3D" or "4EVA"
	match = re.findall(r"\b([A-Z]+)\b", text)
	#print(match)
	return len(match)
	
def find_avg_word_length(text):
	#Assumes input text is tokenized, and a list
	pass

def extract_features(path):
	tknzr = TweetTokenizer()
	for filename in os.listdir(path):
		# new_filename = filename.replace(".", "_") + ".txt"
		# os.rename(filename, new_filename)
		# file = open(new_filename, "r")
		file = open(path + "/" + filename, "r")

		email_content = ""
		email_subject = ""
		for line in file:
			m = re.search('^Subject:', line)
			if m:
				m_span = m.span()
				email_subject = line[m_span[1] + 1: -1]

			m = re.search("^Content-Transfer-Encoding:", line)
			if m:
				file.readline()
				email_content = file.read() 
				break
			if line == "\n":
				file.readline()
				email_content = file.read() 
				break
		email_content_tokenized = tknzr.tokenize(email_content)
		email_subject_tokenized = tknzr.tokenize(email_subject)
		print(email_subject_tokenized)
		print(email_content_tokenized)

		file.close()

		#Call some helper functions on email_content, email_subject. 
		#If email_content is empty for some reason, just don't call the helpers. 

		# print(email_content)
		# print(email_subject)

	#Later, return the matrix of features
	return None

def main():
	# Make sure you fill in the correct path 
	path = 'Data/test'
	features = extract_features(path)
	#count_all_caps_words("The following words are ALL CAPS FOREVER. The following word is in CAPS. 4DOT CATS 9 LIVE")

if __name__ == '__main__':
	main()


