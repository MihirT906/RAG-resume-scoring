import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau, spearmanr

class EvaluationEngine:
    def __init__(self, qrel_file_path):
        self.qrel_file_path = qrel_file_path
        self.qrel_map, self.ideal_ranking = self.read_qrel()
        
    def read_qrel(self):
        qrel_df = pd.read_csv(self.qrel_file_path)
        qrel_map = pd.Series(qrel_df.resume_score.values, index=qrel_df.id).to_dict()
        ideal_ranking = list(qrel_df.sort_values(by='resume_score', ascending=False)['id'])
        print("Ideal Ranking: ", ideal_ranking)
        return qrel_map, ideal_ranking
    
    def calculate_NDCG(self, ranking_df, k):
        """Calculate the NDCG for the given ranking DataFrame up to position k."""
        # Extract the ranking list from the DataFrame
        ranking = ranking_df['id'].tolist()
        
        # Get the relevance scores for the given ranking
        y_true = [self.qrel_map.get(id, 0) for id in ranking]
        
        # Generate the ideal ranking based on the qrel_map
        ideal_ranking = sorted(self.qrel_map.values(), reverse=True)
        y_true_ideal = ideal_ranking[:k]
        
        # Ensure both y_true and y_true_ideal are the same length by padding with zeros if necessary
        # if len(y_true) < k:
        #     y_true += [0] * (k - len(y_true))
        # else:
        #     y_true = y_true[:k]

        # if len(y_true_ideal) < k:
        #     y_true_ideal += [0] * (k - len(y_true_ideal))
        # else:
        #     y_true_ideal = y_true_ideal[:k]
        
        # Calculate the NDCG
        ndcg = ndcg_score([y_true_ideal], [y_true], k=k)
        
        return ndcg

    def calculate_kendall(self, ranking_df):
        """Calculate Kendall's Tau for the given ranking DataFrame."""
        # Get the ideal ranking order from qrel_map
        ideal_ranking = sorted(self.qrel_map.values(), reverse=True)
        
        # Get the predicted ranking order
        ranking = ranking_df['id'].tolist()
        predicted_ranking = [self.qrel_map.get(id, 0) for id in ranking]
        
        # Calculate Kendall's Tau
        kendall_tau, _ = kendalltau(ideal_ranking, predicted_ranking)
        
        return kendall_tau
    
    def calculate_spearman(self, ranking_df):
        """Calculate Spearman's Rank Correlation Coefficient for the given ranking DataFrame."""
        # Get the ideal ranking order from qrel_map
        ideal_ranking = sorted(self.qrel_map.values(), reverse=True)

        # Get the predicted ranking order
        ranking = ranking_df['id'].tolist()
        predicted_ranking = [self.qrel_map.get(id, 0) for id in ranking]
        
        # Calculate Spearman's Rank Correlation Coefficient
        spearman_corr, _ = spearmanr(ideal_ranking, predicted_ranking)
        
        return spearman_corr
    
    def calculate_precision_at_k(self, ranking_df, k):

        precision_at_k = 0.0
        
        first_k_elements = list(self.qrel_map.keys())[:k]
        # Sort ranking by score descending to simulate a ranked list
        sorted_ranking = ranking_df.sort_values(by='score', ascending=False)
        
        # Get the top-k document IDs
        top_k_ids = sorted_ranking.head(k)['id'].tolist()
        
        # Count relevant documents in the top-k ranking
        num_relevant_at_k = sum(1 for id in top_k_ids if id in first_k_elements)
        
        # Calculate precision at k
        precision_at_k = num_relevant_at_k / k if k > 0 else 0
        
        return precision_at_k
    
    def calculate_MAP(self, ranking_df):
        map_score = 0.0
        num_queries = len(ranking_df)
        
        # Calculate P@k for each query and average over all queries
        for k in range(1, len(ranking_df) + 1):
            map_score += self.calculate_precision_at_k(ranking_df, k)
        
        # Average MAP across all queries
        if num_queries > 0:
            map_score /= num_queries
        
        return map_score
