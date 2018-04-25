import re 
from html.parser import HTMLParser

"""Class used to parse a single SGML file or the reuters collection and yield a set of documents."""
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

