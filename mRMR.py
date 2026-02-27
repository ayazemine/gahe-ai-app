import pandas as pd
import numpy as np
from sklearn.feature_selection import f_regression
from scipy.stats import pearsonr


class mRMRFeatureSelector:
    """
    Minimum Redundancy Maximum Relevance (mRMR) Feature Selector
    """
    def __init__(self, n_features=None, relevance_method='f_score', redundancy_method='pearson'):
        self.n_features = n_features
        self.relevance_method = relevance_method
        self.redundancy_method = redundancy_method
        self.selected_features_ = None
        self.feature_scores_ = None
        self.feature_rankings_ = None
        
    def _calculate_relevance(self, X, y):
        """Calculate relevance scores (feature-target correlation)"""
        if self.relevance_method == 'f_score':
            f_scores, _ = f_regression(X, y)
            return f_scores
        elif self.relevance_method == 'pearson':
            relevance_scores = []
            for i in range(X.shape[1]):
                corr, _ = pearsonr(X[:, i], y)
                relevance_scores.append(abs(corr))
            return np.array(relevance_scores)
        else:
            raise ValueError("relevance_method must be 'f_score' or 'pearson'")
    
    def _calculate_redundancy(self, X, selected_features, candidate_feature):
        """Calculate redundancy score (average correlation with selected features)"""
        if len(selected_features) == 0:
            return 0.0
        
        redundancies = []
        for selected_idx in selected_features:
            if self.redundancy_method == 'pearson':
                corr, _ = pearsonr(X[:, candidate_feature], X[:, selected_idx])
                redundancies.append(abs(corr))
            else:
                # Could add other redundancy methods here
                corr, _ = pearsonr(X[:, candidate_feature], X[:, selected_idx])
                redundancies.append(abs(corr))
        
        return np.mean(redundancies)
    
    def fit(self, X, y):
        """
        Fit the mRMR feature selector
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        n_samples, n_total_features = X.shape
        
        if self.n_features is None:
            self.n_features = min(int(n_total_features * 0.5), 50)  # Select 50% of features or max 50
        
        self.n_features = min(self.n_features, n_total_features)
        
        print(f"mRMR: Selecting {self.n_features} features from {n_total_features}")
        
        # Calculate relevance scores for all features
        relevance_scores = self._calculate_relevance(X, y)
        
        # Initialize
        selected_features = []
        remaining_features = list(range(n_total_features))
        feature_scores = []
        
        # Select first feature with highest relevance
        first_feature_idx = np.argmax(relevance_scores)
        selected_features.append(first_feature_idx)
        remaining_features.remove(first_feature_idx)
        feature_scores.append({
            'feature_idx': first_feature_idx,
            'relevance': relevance_scores[first_feature_idx],
            'redundancy': 0.0,
            'mrmr_score': relevance_scores[first_feature_idx]
        })
        
        # Iteratively select remaining features
        for _ in range(self.n_features - 1):
            if not remaining_features:
                break
                
            best_score = -np.inf
            best_feature = None
            best_info = None
            
            for candidate_idx in remaining_features:
                # Calculate relevance
                relevance = relevance_scores[candidate_idx]
                
                # Calculate redundancy
                redundancy = self._calculate_redundancy(X, selected_features, candidate_idx)
                
                # Calculate mRMR score
                mrmr_score = relevance - redundancy
                
                if mrmr_score > best_score:
                    best_score = mrmr_score
                    best_feature = candidate_idx
                    best_info = {
                        'feature_idx': candidate_idx,
                        'relevance': relevance,
                        'redundancy': redundancy,
                        'mrmr_score': mrmr_score
                    }
            
            if best_feature is not None:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                feature_scores.append(best_info)
        
        self.selected_features_ = np.array(selected_features)
        self.feature_scores_ = feature_scores
        
        # Create feature rankings
        self.feature_rankings_ = np.zeros(n_total_features)
        for rank, feature_idx in enumerate(selected_features):
            self.feature_rankings_[feature_idx] = rank + 1
        
        print(f"mRMR: Selected features: {self.selected_features_}")
        print(f"mRMR: Average relevance: {np.mean([s['relevance'] for s in feature_scores]):.4f}")
        print(f"mRMR: Average redundancy: {np.mean([s['redundancy'] for s in feature_scores]):.4f}")
        print(f"mRMR: Average mRMR score: {np.mean([s['mrmr_score'] for s in feature_scores]):.4f}")
        
        return self
    
    def transform(self, X):
        """Transform features by selecting only the chosen features"""
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet.")
        
        return X[:, self.selected_features_]
    
    def fit_transform(self, X, y):
        """Fit the selector and transform the features"""
        return self.fit(X, y).transform(X)
    
    def get_feature_importance(self):
        """Get feature importance information"""
        if self.feature_scores_ is None:
            return None
        
        importance_df = pd.DataFrame(self.feature_scores_)
        importance_df = importance_df.sort_values('mrmr_score', ascending=False)
        return importance_df
