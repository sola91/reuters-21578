import string
import re 
from nltk.corpus import stopwords as sw
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize


class DocumentPreprocessor():

    def __init__(self, stemmer=None, stopwords=None, min_length=3):
        self.stemmer = stemmer or PorterStemmer()
        self.stopwords = stopwords or (sw.words('english') + list(string.punctuation))

    def __tokenize(self, text):
        min_length = 3
        words = map(lambda word: word.lower(), word_tokenize(text))
        words = [word for word in words if word not in self.stopwords]
        tokens = (list(map(lambda token: self.stemmer.stem(token), words)))
        p = re.compile('[a-zA-Z]+');
        filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length, tokens))
        return filtered_tokens
    
    
    #transfor
    def __