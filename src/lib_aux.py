import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def count_outliers(df: pd.DataFrame, normalize: bool =False) -> dict:
    """
    Função que conta os outliers em um DataFrame

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado
    normalize : bool, default=False
        Se True, retorna a porcentagem de outliers em relação ao total de linhas

    Retorno
    -------
    dict
        Dicionário com o nome de cada coluna e a quantidade de outliers
    """

    outliers = {}

    ### Conta os outliers em cada coluna
    for column in df.select_dtypes(include=[np.number]).columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers[column] = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()

    ### Se normalize, divide pelo total de linhas (Porcentagem)
    if normalize:
        for key in outliers.keys():
            outliers[key] = outliers[key] / len(df)

    return outliers

def get_columns_to_eliminate_by_corr(corr_matrix: pd.DataFrame, threshold: float = 0.8) -> list:
    """
    Função que retorna as colunas a serem eliminadas de um DataFrame baseado na correlação. Se uma coluna possui uma correlação acima de 0.8 
    com duas ou mais outras colunas, ela é considerada redundante e é indicada para eliminação.

    Parâmetros
    ----------
    corr_matrix : pd.DataFrame
        Matriz de correlação das variáveis
    threshold : float, default=0.8
        Valor de correlação a partir do qual as variáveis são consideradas altamente correlacionadas

    Retorno
    -------
    list
        Lista com o nome das colunas a serem eliminadas
    dict
        Dicionário com o nome das colunas e a quantidade de vezes que ela aparece em uma correlação acima de 0.8
    """


    ### Adquire o triangulo superior da matriz de correlação
    abs_corr_matrix = corr_matrix.abs()
    upper_triangle = abs_corr_matrix.where(np.triu(np.ones(abs_corr_matrix.shape), k=1).astype(bool))

    corr_appearances = {} ### Dicionário para contar quantas vezes uma coluna aparece em uma correlação acima de 0.8
    to_drop_columns = [] ### Lista para armazenar as colunas a serem eliminadas

    max_corr_value = 1

    ### Itera enquanto houver correlações acima do threshold
    while(max_corr_value > threshold):

        ### Adquire a correlação máxima e o index
        max_corr = upper_triangle.stack().idxmax()
        max_corr_value = upper_triangle.stack().max()

        print(f"A correlação absoluta entre {max_corr} é {max_corr_value}")
        upper_triangle.loc[max_corr] = np.nan ### Remove a correlação máxima para nao repetir

        for column in max_corr:

            ### Se apareceu pela segunda vez, adiciona a lista de colunas a serem eliminadas
            if column in corr_appearances:
                to_drop_columns.append(column)
                upper_triangle.loc[column] = np.nan
            ### Se não, adiciona ao dicionário
            else:
                corr_appearances[column] = 1

    return to_drop_columns, corr_appearances

def show_heatmap(df_corr_key: pd.DataFrame, size: tuple = (8, 6)) -> None:
    """
    Função que plota um mapa de calor da correlação das variáveis de um DataFrame

    Parâmetros
    ----------
    df_corr_key : pd.DataFrame
        DataFrame a ser analisado
    size : tuple, default=(8, 6)
        Tamanho da figura

    Retorno
    -------
    None
        Mostra o mapa de calor da correlação no notebook
    """
    plt.figure(figsize=size)
    sns.heatmap(df_corr_key.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Mapa de calor da correlação das variáveis do DataFrame')
    plt.show()

def na_percentage(row) -> str:
    """
    Função que retorna a porcentagem de valores nulos em uma linha
    
    Parâmetros
    ----------
    row : pd.Series
        Linha do DataFrame
        
    Retorno
    -------
    str
        Porcentagem de valores nulos na linha
    """
    return str(np.round((row.isna().sum() / len(row) * 100), 1)) + '%'

def fill_na_by(df: pd.DataFrame, column: str = "None", method: str = 'mean') -> pd.DataFrame:
    """
    Função que preenche valores nulos em um DataFrame. Se column é "None", preenche com a média, mediana ou moda total. Se não, 
    preenche com a média, mediana ou moda em relação à coluna escolhida.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado
    column : str, default="None"
        Coluna a ser agrupada para preencher os valores nulos
    method : str, default='mean'
        Método de preenchimento dos valores nulos

    Retorno
    -------
    pd.DataFrame
        DataFrame com os valores nulos preenchidos
    """

    ### Se a coluna for None, preenche com a média, mediana ou moda total
    if column == "None":
        if method == 'mean':
            df = df.fillna(df.mean())
        elif method == 'median':
            df = df.fillna(df.median())
        elif method == 'mode':
            df = df.fillna(df.mode().iloc[0])
        else:
            raise ValueError("Método inválido")
        
    else:

        ### Agrupa o DataFrame pela coluna e tira a média, mediana e moda
        grouped_df = df.groupby(column).agg(['mean', 'median', 
                                             #### Moda
                                             lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan]).\
                                             rename(columns={"<lambda>": "mode"})

        ### Pega as colunas a serem preenchidas
        columns_to_fill = [x for x in df.columns if x not in [column]]

        ### Preenche os valores nulos de acordo com o método escolhido em relação à coluna escolhida
        if method in ['mean', 'median', 'mode']:
            for grouped_var in grouped_df.index:
                for coluna_alvo in columns_to_fill:
                    df.loc[df[coluna_alvo].isna() & (df[column] == grouped_var), coluna_alvo] = grouped_df.loc[grouped_var, (coluna_alvo, method)]
        else:
            raise ValueError("Método inválido")
        
    return df

def get_skew_and_variance(df: pd.DataFrame) -> pd.DataFrame: 
    """
    Função que retorna o deslocamento da distribuição e a variância de cada coluna de um DataFrame

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado

    Retorno
    -------
    pd.DataFrame
        DataFrame com o nome da coluna, o deslocamento da distribuição e a variância
    """

    columns_list = []
    skewness_list = []
    variance_list = []
    for column in df.columns:

        skewness = df[column].dropna().skew()
        variance = df[column].dropna().var()

        columns_list.append(column)
        skewness_list.append(skewness)
        variance_list.append(np.round(variance, 3))

    return pd.DataFrame({"Column": columns_list, "Skewness": skewness_list, "Variance": variance_list})