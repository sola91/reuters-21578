import re 
from nltk.corpus import stopwords as sw
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from itertools import chain

class ReutersPreprocessor():
    """
    Class used to preprocess the data 
    
    :stemmer: Stemmer class to be applied during tokenization
    :stopwords: Set of stopwords to be removed from all the documents
    :min_length: Minimum length of a valid token
    """
    def __init__(self, stemmer=PorterStemmer(), stopwords=sw.words('english'), min_length=3):
        self.stemmer = stemmer 
        self.stopwords = stopwords
        self.min_length = min_length
        
    def tokenize(self, text):
        """
        Given a text, returns a list of tokens according to the following rules:
        1 - apply a specified stemming algorithm, 
        2 - remove stop words and consider only alpha-numeric characters
        3 - consider only tokens with length >= min_length 
        """
        words = map(lambda word: word.lower(), word_tokenize(text))
        words = [word for word in words if word not in self.stopwords]
        tokens = (list(map(lambda token: PorterStemmer().stem(token),words)))
        filtered_tokens = list(filter (lambda token: re.match('[a-zA-Z]+', token) and len(token) >= self.min_length,tokens))
        return filtered_tokens

        
    def pre_process(self, documents):
        """
        Preprocess the reuters-21578 document collection and returns
        train and test.
        """
               
        train_documents = documents[(documents.lewissplit == "TRAIN")]
        
        train_category_list = [doc for doc in train_documents["topics"]]
        train_category_set =  set(chain(*train_category_list)) 
        test_documents = documents[(documents.lewissplit == "TEST")]

        #From the test set, we need to remove the topics that are not present in the train set
        test_documents.loc[:,"topics"] = test_documents.topics.apply(lambda x: [entry for entry in x if entry in train_category_set])
        
        #Vectorize the data using the tf-idf method
        vectorizer = TfidfVectorizer(tokenizer=self.tokenize,min_df=2)
        vec_train_documents = vectorizer.fit_transform(train_documents["text"])
        vec_test_documents = vectorizer.transform(test_documents["text"])
 
        # Transform labels for multi label classification
        mlb = MultiLabelBinarizer()
        train_labels = mlb.fit_transform([doc for doc in train_documents["topics"]])
        test_labels = mlb.transform([doc for doc in test_documents["topics"]])
        
        return vec_train_documents, vec_test_documents, train_labels, test_labels