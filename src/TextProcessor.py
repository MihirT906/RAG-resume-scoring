import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

class TextProcessor:
    def __init__(self):
        self.index_terms = []
    
    def tokenize(self, raw_text):
        if isinstance(raw_text, float):
            raw_text = str(raw_text)
        raw_text = re.sub(r'[\n\t]', ' ', raw_text)
        raw_text = re.sub(r'\s+', ' ', raw_text).strip()
        self.index_terms = raw_text.split(" ")
        return self.index_terms
    
    def normalize(self):
        self.index_terms = [string.lower() for string in self.index_terms]
        
    def stopping(self):
        nltk.download('stopwords', quiet=True)
        sw_nltk = stopwords.words('english')
        self.index_terms = [string for string in self.index_terms if string not in sw_nltk]
    
    def lemmatize(self):
        nltk.download('wordnet', quiet=True)
        lemmatizer = WordNetLemmatizer()
        pattern = re.compile(r'[^a-zA-Z0-9\s]')
        #pattern = re.compile(r'[^a-zA-Z\s]')
        self.index_terms = [lemmatizer.lemmatize(pattern.sub('', string)) for string in self.index_terms]
        self.index_terms = [term for term in self.index_terms if term]
        
    def process_text(self, raw_text):
        self.tokenize(raw_text)
        self.normalize()
        self.stopping()
        self.lemmatize()
        return self.index_terms



# tp = TextProcessor()
# print(tp.tokenize('Belgium has great beer. They drink beer all the time.\t\tma\t\t\t\t\t\t\t     '))

