import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu, shapiro
from scipy.spatial import distance

class DistributionAgent:
    def __init__(self, df: pd.DataFrame, label: str = "y", p_value_threshold: float = 0.05):
        """
        Agente para análise de distribuições de variáveis numéricas.
        
        Parâmetros:
            df (pd.DataFrame): DataFrame contendo os dados.
            label (str): Nome da coluna que contém o rótulo binário.
            p_value_threshold (float): Nível de significância para rejeitar H0 (default 0.05).
        """
        self.df = df
        self.label = label
        self.p_value_threshold = p_value_threshold
        self._validate_label()
        self.numeric_cols = self._get_numeric_columns()
    
    def _validate_label(self):
        if self.label not in self.df.columns:
            raise ValueError(f"A coluna de label '{self.label}' não está no DataFrame.")
        
        unique_values = self.df[self.label].dropna().unique()
        if len(unique_values) != 2:
            raise ValueError("O label deve ser binário (com apenas dois valores únicos).")
        
        self.group_0, self.group_1 = unique_values
    
    def _get_numeric_columns(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.label in numeric_cols:
            numeric_cols.remove(self.label)  # Remover a coluna do label caso esteja no conjunto de colunas numéricas
        return numeric_cols
    
    def compare_distributions(self, method: str = "both") -> dict:
        """
        Compara as distribuições das variáveis numéricas usando KS Test, Mann-Whitney U ou ambos.
        
        Parâmetros:
            method (str): Método de teste a ser usado ("ks", "mw" ou "both").
        
        Retorno:
            dict: Chaves são os nomes das colunas e valores são True (distribuições diferentes) ou False (não diferentes).
        """
        if method not in ["ks", "mw", "both"]:
            raise ValueError("O método deve ser 'ks', 'mw' ou 'both'.")
        
        results = {}
        
        for col in self.numeric_cols:
            sample_0 = self.df[self.df[self.label] == self.group_0][col].dropna()
            sample_1 = self.df[self.df[self.label] == self.group_1][col].dropna()
            
            if len(sample_0) > 0 and len(sample_1) > 0:
                ks_p_value, mw_p_value = None, None
                
                if method in ["ks", "both"]:
                    _, ks_p_value = ks_2samp(sample_0, sample_1)
                
                if method in ["mw", "both"]:
                    _, mw_p_value = mannwhitneyu(sample_0, sample_1, alternative='two-sided')
                
                if method == "both":
                    results[col] = (ks_p_value < self.p_value_threshold) or (mw_p_value < self.p_value_threshold)
                elif method == "ks":
                    results[col] = ks_p_value < self.p_value_threshold
                else:  # method == "mw"
                    results[col] = mw_p_value < self.p_value_threshold
            else:
                results[col] = None  # Caso não haja dados suficientes
        
        return results
    
    def test_normality(self) -> dict:
        """
        Testa a normalidade de todas as colunas numéricas usando o teste de Shapiro-Wilk.
        
        Retornao
            dict: Chaves são os nomes das colunas e valores são True (segue distribuição normal) ou False (não segue).
        """
        normality_results = {}
        
        for col in self.numeric_cols:
            sample = self.df[col].dropna()
            if len(sample) > 3:  # Shapiro-Wilk requer pelo menos 3 valores
                _, p_value = shapiro(sample)
                normality_results[col] = p_value > self.p_value_threshold
            else:
                normality_results[col] = None  # Amostras muito pequenas para testar
        
        return normality_results
    
    def count_outliers(self) -> pd.Series:
        """
        Conta a quantidade de variáveis que são outliers para cada linha do DataFrame.
        
        Critério: Um valor é considerado outlier se estiver fora do intervalo [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
        
        Retorno:
            pd.Series: Série com a contagem de outliers para cada linha do DataFrame.
        """
        outlier_counts = pd.Series(0, index=self.df.index)
        
        for col in self.numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_counts += outliers.astype(int)
        
        return outlier_counts
    
    def get_outliers(self, limiar: int = 3) -> pd.DataFrame:
        """
        Retorna um DataFrame contendo apenas as linhas que contêm outliers usando a distância de Mahalanobis.
        
        Retorna:
            pd.DataFrame: DataFrame com as linhas que contêm outliers.
        """

        # Calculando a distância de Mahalanobis para todas as amostras
        cov_matrix = np.cov(self.df.T)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        mean_vector = self.df.mean(axis=0)

        # Calculando a distância de Mahalanobis para cada linha (amostra)
        distances = self.df.apply(lambda row: distance.mahalanobis(row, mean_vector, inv_cov_matrix), axis=1)

        # Definindo um limiar para a distância (usando distribuição qui-quadrado)
        threshold = np.percentile(distances, 100 - limiar)  # Por exemplo, 97% para determinar outliers
        outliers = self.df[distances > threshold]
        return outliers
