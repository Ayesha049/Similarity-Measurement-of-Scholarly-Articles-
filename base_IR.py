"""
Name: Khalid Hasan
BearPass: M03543550
Assignment: 03

Summary:
1. A base file that contains all the necessary repetitive functions.
2. This base file is imported by other information retrieval models.

*** Make sure you have installed the Python's nltk and pandas package
"""
import string

import nltk as tk
from PyPDF2 import PdfReader
from nltk.corpus import stopwords

tk.download('punkt')
tk.download('stopwords')


class BaseIR(object):

    def __init__(self):
        self.stopwords = set(stopwords.words('english'))

    @staticmethod
    def meta_data():
        print(f"=================== CSC790-IR Project ==============")
        print("Title: Identifying similarities of research papers utilizing Natural Language Processing and Information Retrieval methods")
        print("Authors : AYESHA SIDDIQUA and KHALID HASAN")
        print("========================================================")

    @staticmethod
    def _read_one_words(file_path):
        """
        Read one words from the file and return them in a list of one words without a newline
        :param file_path: the relative path of the file
        :return: a list of words without a newline
        """
        with open(file_path, 'r') as file:
            one_words = [line.strip() for line in file.readlines()]

        return one_words

    @staticmethod
    def _read_document(file_path):
        def visitor_body(text, cm, tm, fontDict, fontSize):
            y = tm[5]
            if 50 < y < 750:
                parts.append(text)

        parts = []
        reader = PdfReader(file_path)
        for page in reader.pages:
            # extracting text from page
            page.extract_text(visitor_text=visitor_body)

        text_body = "".join(parts)

        return text_body

    def preprocess(self, text):
        """
        Read and process documents following:
        1. Tokenize a document and convert tokens to lowercase
        2. Remove punctuations and stopwords
        3. Stem the remaining tokens
        :param text: the text needs to be processed
        :return: a list of tokens without punctuations and stopwords
        """
        # Tokenize
        tokens = tk.word_tokenize(text)

        # Lower-case
        tokens_lower = [token.lower() for token in tokens if token]

        # Remove punctuation and special characters
        table = str.maketrans('', '', string.punctuation)
        tokens_lower_without_punc = [token.translate(table) for token in tokens_lower]

        # Remove punctuations and stopwords
        tokens_lower_without_punc_and_stop = [
            token for token in tokens_lower_without_punc if token not in self.stopwords
        ]

        # Stemming
        stemmer = tk.stem.PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in tokens_lower_without_punc_and_stop]

        return stemmed_tokens
