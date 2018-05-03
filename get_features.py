from preprocess_sl import *
from extract_feature import *
from feature_extracter import *
from preprocess_pl import *

avg_word = True
freq_all_caps = True
count_num_list = True
count_non_english = True
word_count = True
longest_seq_char = True
char_count = True

key_words = ["money"]
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

        content_depunct = list(filter(lambda token: token not in string.punctuation, content_tknzd))

        # De-punkt, stopwords removed, alphabetical words only
        content_cleaned = list(filter(lambda token: token not in stop_words, content_depunct))
        content_cleaned = [token for token in content_cleaned if token.isalpha()]

        content_cleaned_lower = [token.lower() for token in content_cleaned]
        print(">>>>>>>>>>>>>>>>>>>>>", filename, ">>>>>>>>>>>>>>>>>>>>")
        # print(content_depunct)
        # print(subject_tknzd)
        # Call your helper functions here! Use the raw, depunct, or cleaned content/subject.
        out_list = []
        if avg_word:
            avg_word_len = find_avg_word_length(content_cleaned_lower)
            out_list.append((avg_word_len))
        if freq_all_caps:
            freq_all_caps_words = count_all_caps_words(content_cleaned)
            out_list.append(freq_all_caps_words)
        if count_num_list:
            num_numbered_lists = count_numbered_lists(email_content)
            out_list.append(num_numbered_lists)
        if count_non_english:
            non_english_count = count_non_english_words(content_cleaned)
            out_list.append(non_english_count)
        if word_count:
            word_counts = count_words(content_cleaned)
            out_list.append(word_counts)
        if longest_seq_char:
            longest_char_length = count_longest_seq(email_content)
            out_list.append(longest_char_length)
        if char_count:
            total_char_count = count_num_chars(content_cleaned)
            out_list.append(total_char_count)

        print(out_list)
        # print(avg_word_len)
        # print(freq_all_caps_words)
        # print(num_numbered_lists)

        file.close()



        # Do we need to consider case where email_content is empty?
        # No. That means we have too much data missing.


    # TO DO: return features in matrix form, write this to a file
    return None

def main():
    # Make sure you fill in the correct path
    path = 'Data/test'
    features = extract_features(path)
    #count_all_caps_words("The following words are ALL CAPS FOREVER. The following word is in CAPS. 4DOT CATS 9 LIVE")

if __name__ == '__main__':
    main()