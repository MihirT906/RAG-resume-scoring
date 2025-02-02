# from TF_IDF import TF_IDF
from data_prep import load_data, read_docx, load_csv
from word2vec import word2vec
from TF_IDF import TF_IDF
from BM25 import BM25


job_description = "Olympics preparation and athlete performances"

news_path = '../data/news-article-categories.csv'
news_articles = load_csv(news_path)

#TFIDF
tfidf = TF_IDF()
tfidf.add_documents(news_articles)
TFIDF_document_scores = tfidf.search_job_desc(job_description)
TFIDF_document_scores['rank'] = TFIDF_document_scores['score'].rank(method='min', ascending=False).astype(int)

#BM25
bm25 = BM25()
bm25.add_documents(news_articles)
BM25_document_scores = bm25.search_job_desc(job_description)
BM25_document_scores['rank'] = BM25_document_scores['score'].rank(method='min', ascending=False).astype(int)

#Word2Vec
w2v = word2vec()
w2v.add_documents(news_articles)
w2v.document_vectors()
w2v.process_job_description(job_description)
W2V_document_scores = w2v.compute_score()
W2V_document_scores['rank'] = W2V_document_scores['score'].rank(method='min', ascending=False).astype(int)



document_scores = TFIDF_document_scores.merge(W2V_document_scores, how='outer', on='id')
document_scores = document_scores.merge(BM25_document_scores, how='outer', on='id')
document_scores = document_scores.rename(columns={'score_x': 'TFIDF score', 'score_y': 'Word2Vec score', 'score': 'BM25 score'})
document_scores = document_scores.rename(columns={'rank_x': 'TFIDF rank', 'rank_y': 'Word2Vec rank', 'rank': 'BM25 rank'})
print(f"Total number of articles: {news_articles.shape[0]}")
document_ranks = document_scores[['id', 'TFIDF rank', 'Word2Vec rank', 'BM25 rank']]
document_scores = document_scores[['id', 'TFIDF score', 'Word2Vec score', 'BM25 score']]

print("----- RANKING WITH TFIDF -----")
print(document_ranks.sort_values(by='TFIDF rank', ascending=True)[:10])

print("----- RANKING WITH WORD2VEC -----")
print(document_ranks.sort_values(by='Word2Vec rank', ascending=True)[:10])

print("----- RANKING WITH BM25 -----")
print(document_ranks.sort_values(by='BM25 rank', ascending=True)[:10])


print("----- RANKING WITH TFIDF -----")
print(document_scores.sort_values(by='TFIDF score', ascending=False)[:10])

print("----- RANKING WITH WORD2VEC -----")
print(document_scores.sort_values(by='Word2Vec score', ascending=False)[:10])

print("----- RANKING WITH BM25 -----")
print(document_scores.sort_values(by='BM25 score', ascending=False)[:10])