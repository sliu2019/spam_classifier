from preprocess_sl import *
from extract_feature import *
from feature_extracter import *
from preprocess_pl import *
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
avg_word = True
freq_all_caps = True
count_num_list = True
count_non_english = True
word_count = True
longest_seq_char = True
char_count = True
bag_of_words = True


def extract_features(path):
    """ The main function which reads all training data, and extracts all features, then writes them (in matrix form) to an output file.
        To use, make sure to pass it the path of the directory containing the text files.
        Of course, the path is relative to the location of this file.
        It currently cannot navigate into subdirectories, so you may need to call several times for several folders.
    """
    #nlp = StanfordCoreNLP(r'\stanford-english-corenlp-2017-06-09-models')
    test = None
    tknzr = TweetTokenizer()
    stop_words = set(stopwords.words('english'))

    for filename in os.listdir(path):
        try:
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
            # print(email_content)

            content_tknzd = tknzr.tokenize(email_content)
            subject_tknzd = tknzr.tokenize(email_subject)

            content_depunct = list(filter(lambda token: token not in string.punctuation, content_tknzd))

            # De-punkt, stopwords removed, alphabetical words only
            content_cleaned = [token for token in content_tknzd if not token in stop_words]
            # content_cleaned = [token for token in content_cleaned if token.isalpha()]

            content_cleaned_lower = [token.lower() for token in content_cleaned]
            if test == None:
                test = content_cleaned_lower
            else:
                test = test + content_cleaned_lower
            print(">>>>>>>>>>>>>>>>>>>>>", filename, ">>>>>>>>>>>>>>>>>>>>")

            file.close()
        except UnicodeDecodeError:
            continue


        # Do we need to consider case where email_content is empty?
        # No. That means we have too much data missing.
    # vectorizer = TfidfVectorizer('content', 'cp1252', 'strict')
    # vectorizer.fit(test)
    # print(len(test))
    # a = vectorizer.transform(test)
    # print(a.todense().shape)
    # b = vectorizer.inverse_transform(a)
    # print(b)
    # c = 1
    dictionary = {}
    for token in test:
        if token not in dictionary:
            dictionary[token] = 1
        else:
            dictionary[token] += 1
    return Counter(dictionary).most_common(500)

def main():
    # Make sure you fill in the correct path
    path = 'Data/easy_ham'
    easy_ham_dic = [item[0] for item in extract_features(path)]
    path2 = 'Data/hard_ham'
    hard_ham_dic = [item[0] for item in extract_features(path2)]
    path3 = 'Data/spam'
    spam_dic = [item[0] for item in extract_features(path3)]
    final_set = set(easy_ham_dic).union(hard_ham_dic).union(spam_dic)
    final_list = list(final_set)
    # print(final_list)
    import csv
    with open("key_words.csv", "w") as outputFile:
        wr = csv.writer(outputFile)
        wr.writerow(final_list)
    #count_all_caps_words("The following words are ALL CAPS FOREVER. The following word is in CAPS. 4DOT CATS 9 LIVE")

if __name__ == '__main__':
    main()