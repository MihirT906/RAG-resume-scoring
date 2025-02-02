from TextProcessor import TextProcessor 
from collections import Counter 


class InvertedIndex:
    def __init__(self):
        self.document_list = []
        self.index_dict = {}
    
    def add_documents(self, list_of_documents):
        self.document_list = list_of_documents
        
    def create_dict(self, list_of_documents):
        self.add_documents(list_of_documents)
        for _, document in self.document_list.iterrows():
            self.add_to_dict(document['body'], document['id'])
        
        # self.show_index_dict()
        return self.index_dict
    
    def show_index_dict(self):
        for index, value in self.index_dict.items():
            print(f"{index}:{value}")
        
    
    def add_to_dict(self, doc_content, doc_ID):
        tp = TextProcessor()
        index_terms = tp.process_text(doc_content)    

        frequency_count = Counter(index_terms)
        
        for item, frequency in frequency_count.items():
            if item not in self.index_dict:
                self.index_dict[item] = [(doc_ID, frequency)]
            else:
                self.index_dict[item].append((doc_ID, frequency))
            
# document1 = {'id': 1, 'body': 'The big sharks of Belgium drink beer.'}
# document2 = {'id': 2, 'body': 'Belgium has great beer. They drink beer all the time.'}
# document_list = [document1, document2]
# io = InvertedIndex()
# print(io.create_dict(document_list))

        