from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class TF_IDF:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.documents = None
        self.document_ids = None
    
    def add_documents(self, list_of_documents):
        list_of_documents = list_of_documents.dropna(subset=['body'])
        self.documents = list_of_documents['body'].tolist()
        self.document_ids = list_of_documents['id'].tolist()
        self.vectorizer.fit(self.documents)
    
    def search_job_desc(self, job_description):
        query_vector = self.vectorizer.transform([job_description])
        document_vectors = self.vectorizer.transform(self.documents)
        
        cosine_similarities = cosine_similarity(query_vector, document_vectors).flatten()
        document_scores = pd.DataFrame({'id': self.document_ids, 'score': cosine_similarities})
        document_scores = document_scores.sort_values(by='score', ascending=False)
        
        return document_scores
