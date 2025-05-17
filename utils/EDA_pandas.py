import numpy as np
import pandas as pd
from typing import Tuple
# Definindo as categorias de preditividade com base no IV
def categorize_iv(iv_value):
    if iv_value < 0.02:
        return "Not useful for prediction"
    elif 0.02 <= iv_value < 0.1:
        return "Weak predictive Power"
    elif 0.1 <= iv_value < 0.3:
        return "Medium predictive Power"
    elif 0.3 <= iv_value < 0.5:
        return "Strong predictive Power"
    else:
        return "Suspicious Predictive Power"

# Aplicando a função ao DataFrame iv
# iv['Predictiveness'] = iv['IV'].apply(categorize_iv)


def iv_woe(
    data: pd.DataFrame, 
    target: str, 
    bins: int = 10, 
    show_woe: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates the Weight of Evidence (WOE) and Information Value (IV) for all independent variables in a DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        target (str): Column name of the binary target variable.
        bins (int): Number of bins to use for WOE calculation (default is 10).
        show_woe (bool): If True, print the WOE table for each variable (default is True).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - DataFrame with IV values for each independent variable.
            - DataFrame with WOE values for each bin of each independent variable.

    Reference:  https://lucastiagooliveira.github.io/datascience/iv/woe/python/2020/12/15/iv_woe.html      
    Reference: https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
    Reference: https://www.linkedin.com/pulse/information-value-uma-excelente-t%C3%A9cnica-para-de-j%C3%BAlia-de-moura-ertel/
    Reference: https://www.analyticsvidhya.com/blog/2021/06/understand-weight-of-evidence-and-information-value/
    Reference: https://onlinelibrary.wiley.com/doi/10.1155/2013/848271
    """
    
    # Empty Dataframe
    newDF, woeDF = pd.DataFrame(), pd.DataFrame()
    
    # Extract Column Names
    cols = data.columns
    
    # Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars])) > 10):
            binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d = d0.groupby("x", as_index=False, observed=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        # Adicionamos um pequeno valor (neste caso, 0.5) para garantir que não haja divisões por zero e evitar que o logaritmo de zero seja calculado.
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events'] / d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)

        temp = pd.DataFrame({"Variable": [ivars], "IV": [d['IV'].sum()]}, columns=["Variable", "IV"])
        newDF = pd.concat([newDF, temp], axis=0)
        woeDF = pd.concat([woeDF, d], axis=0)
        
        newDF = newDF.sort_values(by='IV', ascending=False)
        newDF['Predictiveness'] = newDF['IV'].apply(categorize_iv)

        # Show WOE Table
        if show_woe:
            print(d)
    
    return newDF, woeDF