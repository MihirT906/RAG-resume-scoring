from rank_bm25 import BM25Okapi
from TextProcessor import TextProcessor
import pandas as pd
class BM25:
    def __init__(self):
        self.bm25 = None
        self.documents = None
        self.document_ids = None
        self.text_processor = TextProcessor()
    
    def add_documents(self, list_of_documents):
        list_of_documents = list_of_documents.dropna(subset=['body'])
        self.documents = list_of_documents['body'].tolist()
        self.document_ids = list_of_documents['id'].tolist()
        
        tokenized_corpus = [self.text_processor.process_text(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def search_job_desc(self, job_description):
        query = self.text_processor.process_text(job_description)
        document_scores = self.bm25.get_scores(query)
        
        document_scores_df = pd.DataFrame({'id': self.document_ids, 'score': document_scores})
        document_scores_df = document_scores_df.sort_values(by='score', ascending=False)
        
        return document_scores_df