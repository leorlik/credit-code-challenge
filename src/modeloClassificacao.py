from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class ModeloClassificacao(BaseEstimator, ClassifierMixin):
    """
    Implementação de um modelo de classificação utilizando Regressão Logística com pré-processamento de dados.

    Parâmetros
    ----------
    None

    Atributos
    ----------
    model : LogisticRegression
        Modelo de Regressão Logística
    adasyn : ADASYN
        Técnica de oversampling ADASYN
    pipeline : Pipeline
        Pipeline de pré-processamento

    Métodos
    -------
    fit(X, y)
        Treina o modelo com os dados de entrada
    predict(X)
        Realiza a predição com os dados de entrada
    predict_proba(X)
        Realiza a predição de probabilidade com os dados de entrada
    """
    def __init__(self):
        
        self.model = LogisticRegression(**{'C': 0.1, 'max_iter': 100, 'penalty': 'l2', 'solver': 'liblinear'}, 
                                        random_state=98)

        self.adasyn = ADASYN(sampling_strategy='minority')
        
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),          
            ('pca', PCA(n_components=0.95))
        ])
    
    def fit(self, X, y):

        X_transformed = self.pipeline.fit_transform(X)
        
        X_resampled, y_resampled = self.adasyn.fit_resample(X_transformed, y)
        

        self.model.fit(X_resampled, y_resampled)
        
        return self
    
    def predict(self, X):

        X_transformed = self.pipeline.transform(X)
        
        return self.model.predict(X_transformed)
    
    def predict_proba(self, X):

        X_transformed = self.pipeline.transform(X)

        return self.model.predict_proba(X_transformed)