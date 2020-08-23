import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import regex as re

url = 'https://www.gutenberg.org/files/2701/2701-h/2701-h.htm'


class Question2:
    stop_words = set(stopwords.words("english"))

    def __init__(self, url):
        self.r = requests.get(url)
        # Extract HTML from Response object and print
        self.html = self.r.text
        # Create a BeautifulSoup object from the HTML
        self.soup = BeautifulSoup(self.html, "html5lib")
        # Get soup title
        self.soup.title
        # Get soup title as string
        self.soup.title.string
        # Get the text out of the soup and print it
        self.text_string = self.soup.get_text()

    def text(self):
        return self.text_string

    def tokenize(self, text):
        return word_tokenize(text)

    def to_lowercase(self, tokens):
        return [word.lower() for word in tokens]

    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words]

    def fdistribution(self, tokenized_words):
        fdist = FreqDist(tokenized_words)
        fdist.plot(30, cumulative=False)
        plt.show()

    def tf_idf(self, without_sws):
        # Declare dict to hold tf of each word
        tf_dict_local = {}

        document_size = len(without_sws)

        # Remove punctuation from the input string
        bag_of_words = [word for word in without_sws if re.match("^\P{P}(?<!-)", word)]

        # Get unique words from the string without punctuation
        unique_words = set(bag_of_words)

        # Create dictionary with number of occurences of words in the corpus
        word_count_dict = dict.fromkeys(unique_words, 0)

        # Iterate through bag of words and record the number of times a word appears
        for word in bag_of_words:
            word_count_dict[word] += 1

        return word_count_dict

    def tf_idf_top10(self, document):
        pass


q2 = Question2(url)
text = q2.text()
tokenized_words = q2.tokenize(text)

lower_tokenized_words = q2.to_lowercase(tokenized_words)
without_stopwords = q2.remove_stopwords(lower_tokenized_words)

# Remove punctions from the without_stopwords variable
tf_dict = q2.tf_idf(without_stopwords)
print(tf_dict)

fdist = q2.fdistribution(without_stopwords)
