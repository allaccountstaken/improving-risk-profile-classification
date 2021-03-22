from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score
from azureml.core.run import Run
import pandas as pd
import numpy as np
import argparse
import joblib
import os





def load_data(path="https://raw.githubusercontent.com/allaccountstaken/automl_v_hyperdrive/main/data/camel_data_after2010Q3.csv"):

    '''
        Returns a DataFrame with bank instances and CAMELS features as columns.
        Parameters:
        file_name (cvs file): expects a path to cvs file.
        Default is 'data/camel_data_after2010Q3.csv'
        
        Returns:
        df with index_col=0
        '''
    df = pd.read_csv(path, index_col=0)

    return df

def clean_data(df):
    '''
        Returns X number of CAMELS features and y with Target.
        
        Parameters:
        dataset (DataFrame): expects a DataFrame with CAMELS features and Target column.
        Example: the dataset should at least have the following feture columns df[['EQTA', 'EQTL', 'LLRTA', 'LLRGL', 'OEXTA', 'INCEMP', 'ROA', 'ROE', 'TDTL', 'TDTA', 'TATA']], as well as 'Target' to build target vector y.
        
        Returns:
        X features and reshaped y target vector
        '''
    pd.set_option('use_inf_as_na', True)
    df.dropna(inplace=True)
    X = df[['EQTA', 'EQTL', 'LLRTA', 'LLRGL', 'OEXTA', 'INCEMP', 'ROA', 'ROE', 'TDTL', 'TDTA', 'TATA']].copy()
    y = df['Target'].values.reshape(-1, 1)

    return X, y



def test_oos_performance(model, oos_reports):
    pass

def plot_oos_performance(ins_df, oos_df):
    pass
