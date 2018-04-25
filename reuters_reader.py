import os
import fnmatch
import itertools
from pandas import DataFrame
import re 
from html.parser import HTMLParser


"""
Parser of a single SGML file of the reuters-21578 collection. 
"""
class ReutersSGMLParser(HTMLParser):
    def __init__(self, verbose=0):
        HTMLParser.__init__(self)
        self._reset()

    def _reset(self):
        self.in_title = 0
        self.in_body = 0
        self.in_topics = 0
        self.in_topic_d = 0
        self.title = ""
        self.body = ""
        self.topics = []
        self.topic_d = ""
        self.lewissplit= ""

    def parse(self, fd):
        self.docs = []
        self.feed(fd)
        for doc in self.docs:
            yield doc
        self.close()
    
    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_title:
            self.title += data
        elif self.in_topic_d:
            self.topic_d += data

    def handle_starttag(self, tag, attributes):
        if tag == "reuters":
            attributes_dict = dict(attributes)
            self.lewissplit = attributes_dict["lewissplit"]
        elif tag == "title":
            self.in_title = 1
        elif tag == "body":
            self.in_body = 1        
        elif tag == "topics":
            self.in_topics = 1        
        elif tag == "d":
            self.in_topic_d = 1
    
    def handle_endtag(self, tag):
        if tag == "reuters":
            self.body = re.sub(r'\s+', r' ', self.body)
            self.docs.append({'title': self.title,
                          'body': self.body,
                          'topics': self.topics,
                          'lewissplit': self.lewissplit})
            self._reset()
        elif tag == "title":
            self.in_title = 0
        elif tag == "body":
            self.in_body = 0       
        elif tag == "topics":
            self.in_topics = 0      
        elif tag == "d":
            self.in_topic_d = 0
            self.topics.append(self.topic_d)
            self.topic_d = ""        



class ReutersReader():
    
    def __init__(self, data_path):
        self.data_path = data_path
        
        
    """
    Iterate through all the .sgm files and returns a generator cointaining all the documents
    in the router-21578 collection
    """    
    def __fetch_documents_generator(self):   
        for root, _dirnames, filenames in os.walk(self.data_path):
            for filename in fnmatch.filter(filenames, '*.sgm'):
                path = os.path.join(root, filename)
                parser = ReutersSGMLParser()
                for doc in parser.parse(open(path, encoding='utf-8', errors='replace').read()):
                    yield doc
    
    
    """
    Return a dataframe containing one row for each document that has at least
    one topic assigned.
    """
    def get_documents(self):
    
        doc_generator = self.__fetch_documents_generator()
        data = [('{title}\n\n{body}'.format(**doc), doc['topics'], doc["lewissplit"])
            for doc in itertools.chain(doc_generator)
            if doc['topics']]
    
        if not len(data):
            return DataFrame([])
        else:
            return DataFrame(data, columns=['text', 'topics','lewissplit'])
       