import math
# Scikit Learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import regex as re
from numpy import take

document_one = 'https://www.gutenberg.org/files/2701/2701-h/2701-h.htm'  # Moby Dick eBook
document_two = 'https://www.gutenberg.org/files/768/768-h/768-h.htm'  # Wuthering Heights eBook
document_three = 'https://en.wikipedia.org/wiki/Titanic'  # Wikipedia article on The Titanic
document_four = 'https://en.wikipedia.org/wiki/Wanggongchang_Explosion'  # Wikipedia article on the worst
# accidental explosion in human history
document_five = 'https://en.wikipedia.org/wiki/September_11_attacks'  # Wikipedia article on the 9/11 attacks


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

    def tf(self, without_sws):
        tf_dict_local = {}

        # Remove punctuation from the input string
        bag_of_words = [word for word in without_sws if re.match("^\P{P}(?<!-)", word)]

        document_size = len(bag_of_words)

        # Get unique words from the string without punctuation
        unique_words = set(bag_of_words)

        # Create dictionary with number of occurrences of words in the corpus
        word_count_dict = dict.fromkeys(unique_words, 0)

        # Iterate through bag of words and record the number of times a word appears
        for word in bag_of_words:
            word_count_dict[word] += 1

        # Calculate the tf of each word and add it to a dictionary
        for word, count in word_count_dict.items():
            tf_dict_local[word] = count / float(document_size)

        return tf_dict_local

    def idf(self, documents):
        document_size = len(documents)
        idf_dict = dict.fromkeys(documents[0].keys(), 0)
        for document in documents:
            for word, value in document.items():
                if value > 0:
                    idf_dict[word] = math.log(document_size / float(value))
        return idf_dict

    def tf_idf(self, tf_dict, idf):
        tf_idf_dict_local = {}
        for word, value in tf_dict.items():
            tf_idf_dict_local[word] = tf_dict[word] * idf[word]
        return tf_idf_dict_local

    def tf_idf_top10(self, document):
        # Sort the input dictionary and output as a list
        sorted_dict = sorted(document.items(), key=lambda x: x[1], reverse=True)

        # Trim the list and convert it back to a dictionary
        first10vals = dict(sorted_dict[:10])
        return first10vals


print("Getting content from websites...")
# Get document content from the internet using BS4 functions
document_one_content = Question2(document_one)
document_two_content = Question2(document_two)
document_three_content = Question2(document_three)
document_four_content = Question2(document_four)
document_five_content = Question2(document_five)

# Strip html elements and get the text from each of the documents
text_document_one = document_one_content.text()
text_document_two = document_two_content.text()
text_document_three = document_three_content.text()
text_document_four = document_four_content.text()
text_document_five = document_five_content.text()

# Tokenize each of the documents
tokenized_document_one = document_one_content.tokenize(text_document_one)
tokenized_document_two = document_two_content.tokenize(text_document_two)
tokenized_document_three = document_three_content.tokenize(text_document_three)
tokenized_document_four = document_four_content.tokenize(text_document_four)
tokenized_document_five = document_five_content.tokenize(text_document_five)

# Convert each of the tokenized documents to lowercase
lowercase_doc_one = document_one_content.to_lowercase(tokenized_document_one)
lowercase_doc_two = document_two_content.to_lowercase(tokenized_document_two)
lowercase_doc_three = document_three_content.to_lowercase(tokenized_document_three)
lowercase_doc_four = document_four_content.to_lowercase(tokenized_document_four)
lowercase_doc_five = document_five_content.to_lowercase(tokenized_document_five)

# Remove all stopwords from each of the documents
no_stopwords_doc_one = document_one_content.remove_stopwords(lowercase_doc_one)
no_stopwords_doc_two = document_two_content.remove_stopwords(lowercase_doc_two)
no_stopwords_doc_three = document_three_content.remove_stopwords(lowercase_doc_three)
no_stopwords_doc_four = document_four_content.remove_stopwords(lowercase_doc_four)
no_stopwords_doc_five = document_five_content.remove_stopwords(lowercase_doc_five)

# Evaluate the tf of each document
tf_doc_one = document_one_content.tf(no_stopwords_doc_one)
tf_doc_two = document_two_content.tf(no_stopwords_doc_two)
tf_doc_three = document_two_content.tf(no_stopwords_doc_three)
tf_doc_four = document_two_content.tf(no_stopwords_doc_four)
tf_doc_five = document_two_content.tf(no_stopwords_doc_five)

# Evaluate the idf of all documents
all_documents_idf = document_one_content.idf([tf_doc_one, tf_doc_two, tf_doc_three, tf_doc_four, tf_doc_five])

# Evaluate the tf-idf of each document
tf_idf_doc_one = document_one_content.tf_idf(tf_doc_one, all_documents_idf)
tf_idf_doc_two = document_two_content.tf_idf(tf_doc_two, all_documents_idf)
tf_idf_doc_three = document_three_content.tf_idf(tf_doc_three, all_documents_idf)
tf_idf_doc_four = document_four_content.tf_idf(tf_doc_four, all_documents_idf)
tf_idf_doc_five = document_five_content.tf_idf(tf_doc_five, all_documents_idf)

# Output the top 10 elements with the highest tf_idfs
print(document_one_content.tf_idf_top10(tf_idf_doc_one))
print(document_two_content.tf_idf_top10(tf_idf_doc_two))
print(document_three_content.tf_idf_top10(tf_idf_doc_three))
print(document_four_content.tf_idf_top10(tf_idf_doc_four))
print(document_five_content.tf_idf_top10(tf_idf_doc_five))

#Cosine Similarity
# Define the documents
documents = [document_one, document_two, document_three, document_four, document_five]
# Create the Document Term Matrix
count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(documents)

# OPTIONAL: Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies.
doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix,
                  columns=count_vectorizer.get_feature_names(),
                  index= [document_one, document_two, document_three, document_four, document_five])
df
# Compute Cosine Similarity

print(cosine_similarity(df, df))