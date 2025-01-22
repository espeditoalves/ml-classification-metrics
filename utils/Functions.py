import pandas as pd
from typing import Dict

def confusion_matriz(df: pd.DataFrame, target: str, y_pred: str) -> Dict[str, int]:
    """
    Calcula a matriz de confusão para um conjunto de dados fornecido.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados reais e as previsões.
    target (str): Nome da coluna que contém os valores reais.
    y_pred (str): Nome da coluna que contém os valores previstos.

    Retorna:
    Dict[str, int]: 
            Um dicionário contendo os valores de 
            Verdadeiros Positivos (VP), Falsos Positivos (FP), 
            Verdadeiros Negativos (VN) e Falsos Negativos (FN).
    """
    VP = df.query(f'{target} == 1 and {y_pred} == 1').shape[0]
    VN = df.query(f'{target} == 0 and {y_pred} == 0').shape[0]
    FP = df.query(f'{target} == 0 and {y_pred} == 1').shape[0]
    FN = df.query(f'{target} == 1 and {y_pred} == 0').shape[0]
    
    return {
        'Verdadeiros Positivos (VP ou TP)': VP,
        'Falsos Positivos (FP)': FP,
        'Verdadeiros Negativos (VN ou TN)': VN,
        'Falsos Negativos (FN)': FN
    }