from sklearn.metrics.pairwise import cosine_similarity
from TextProcessor import TextProcessor
import numpy as np
import pandas as pd

import gensim.downloader as api
from gensim.models import Word2Vec


class word2vec:
    def __init__(self):
        self.resume_list = None
        self.no_of_resumes = 0
        self.vector_docs_list = None
        self.vector_docs_list = pd.DataFrame(columns=['id', 'vec'])
        self.model = api.load('glove-wiki-gigaword-100')
        
    def add_documents(self, list_of_documents):
        self.resume_list = list_of_documents
        self.no_of_documents = len(list_of_documents)

    def document_vectors(self):
        tp = TextProcessor()
        for _, resume in self.resume_list.iterrows():
            doc = tp.process_text(resume['body'])
            index_terms = [word for word in doc if word in self.model]
            if index_terms:
                vector_mean = np.mean(self.model[index_terms], axis=0)
                self.vector_docs_list.loc[len(self.vector_docs_list)] = {'id': resume['id'], 'vec': vector_mean}
            else:
                print(f"No valid terms found in the model for resume ID: {resume['id']}")
            #self.vector_docs_list.loc[len(self.vector_docs_list)] = {'id': resume['id'], 'vec': np.mean(self.model[index_terms], axis=0)}
            #self.vector_docs_list.append({'id': resume['id'], 'vec': np.mean(self.model[index_terms], axis=0)})
        return
    
    def process_job_description(self, job_description):
        tp = TextProcessor()
        index_terms = tp.process_text(job_description)
        vec_job_desc = [word for word in index_terms if word in self.model]
        self.vectorised_job_desc = np.mean(self.model[vec_job_desc], axis=0)
        return
    
    def compute_score(self):
        document_scores = pd.DataFrame(columns=['id', 'score'])
        for _,document in self.vector_docs_list.iterrows():
            document_scores.loc[len(document_scores)] = {'id': document['id'], 'score': cosine_similarity([self.vectorised_job_desc], [document['vec']])[0][0]}
            #scores.append((resume['id'], cosine_similarity([self.vectorised_job_desc], [resume['vec']])[0][0]))
 
        document_scores = document_scores.sort_values(by='score', ascending=False)
    
        #sorted_ids = [doc_id for doc_id, score in document_scores if score > 0][:10]
        
        return document_scores
    