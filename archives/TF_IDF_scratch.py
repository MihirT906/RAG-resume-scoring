from archives.InvertedIndex import InvertedIndex 
from TextProcessor import TextProcessor 
import pandas as pd

from data_prep import load_data
class TF_IDF:
    def __init__(self):
        #self.query = None
        self.resume_list = None
        self.no_of_resumes = 0
        self.inverted_index = None
        #self.idf_scores = {}
    
    def add_documents(self, list_of_documents):
        self.resume_list = list_of_documents
        self.no_of_documents = len(list_of_documents)
        self.inverted_index = InvertedIndex().create_dict(self.resume_list)
        
    def process_job_description(self, job_description):
        tp = TextProcessor()
        index_terms = tp.process_text(job_description)
        return index_terms
    
    def compute_IDF_scores(self, query):
        idf_scores = {}
        index_terms = self.process_job_description(query)
        for index in index_terms:
            if(index in self.inverted_index):
                idf_scores[index] = self.no_of_documents/len(self.inverted_index[index])
            else:
                idf_scores[index] = 0
        
        return idf_scores
    
    def compute_score(self, document, idf_scores):
        target_docID = document['id']
        score = 0
        for index, idf in idf_scores.items():
            if(idf>0):
                tf = next((freq for docID_, freq in self.inverted_index[index] if docID_ == target_docID), 0)
                score+= tf*idf
        
        return score

    def search_job_desc(self, query):
        idf_scores = self.compute_IDF_scores(query)
        document_scores = pd.DataFrame(columns=['id', 'score'])
        #print(self.resume_list.head())
        for _,document in self.resume_list.iterrows():
            document_scores.loc[len(document_scores)] = {'id': document['id'], 'score': round(self.compute_score(document, idf_scores), 2)}
        
        document_scores = document_scores.sort_values(by='score', ascending=False)
    
        #sorted_ids = [doc_id for doc_id, score in document_scores if score > 0][:10]
        
        return document_scores
        print(sorted_ids)
    

