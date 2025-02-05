import numpy as np
from sklearn.preprocessing import normalize

class CustomFeatureSelector:
    def __init__(self, estimator, number_logs=None, threshold=0.1, apply_normalization=True):
        self.estimator = estimator
        self.number_logs = number_logs  # Pode ser None, será definido depois
        self.threshold = threshold  # Armazena corretamente
        self.apply_normalization = apply_normalization  # Opção para normalização
        self.feature_importances_ = None
        self.support_ = None
    
    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.feature_importances_ = np.array(self.estimator.feature_importances_)
        
        # Define number_logs dinamicamente se não for passado
        if self.number_logs is None:
            self.number_logs = max(1, min(X.shape[1] // 2, len(self.feature_importances_) // 2))  # Metade das features, mínimo 1
        
        return self
    
    def transform(self, X):
        X = np.array(X)  # Garante que X é um array numpy
        
        max_pairs = (X.shape[1] // 2) * 2  # Garante um número válido de pares
        
        # Itera sobre pares de features e mantém a mais importante
        for i in range(0, min(self.number_logs * 2, max_pairs), 2):
            if i + 1 >= X.shape[1]:  # Evita indexação fora dos limites
                break
            
            if self.feature_importances_[i] > self.feature_importances_[i + 1]:
                if self.apply_normalization:
                    X[:, i + 1] = normalize(X[:, i + 1].reshape(1, -1))  # Normaliza antes de remover
                mask = np.ones(X.shape[1], dtype=bool)
                mask[i + 1] = False  # Remove a feature menos importante
            else:
                if self.apply_normalization:
                    X[:, i] = normalize(X[:, i].reshape(1, -1))  # Normaliza antes de remover
                mask = np.ones(X.shape[1], dtype=bool)
                mask[i] = False
            X = X[:, mask]
            self.feature_importances_ = self.feature_importances_[mask]
        
        # Mantém apenas features acima do threshold
        selected_features = self.feature_importances_ > self.threshold
        self.support_ = selected_features
        return X[:, selected_features]
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
    
    def get_support(self):

        if self.support_ is None:

            raise RuntimeError("You must fit the selector before getting support")

        return self.support_
