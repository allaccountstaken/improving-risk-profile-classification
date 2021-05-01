# These are helpers functions used in the project to operate on CAMELS datasets
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





def load_data(file_path):

    '''
        Returns a DataFrame with bank instances and CAMELS features as columns.
        Parameters:
        file_name (cvs file): expects a path to cvs file.
        Default is 'data/camel_data_after2010Q3.csv'
        
        Returns:
        df with index_col=0
        '''
    df = pd.read_csv(file_path, index_col=0)

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

def main():
    
    # Prepare parser for parameters to tune
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--n_estimators', type=int, default=20, help="Maximum number of iterations to converge, similar to max_iter in Logistic Regression")
    parser.add_argument('--max_features', type=int, default=5, help="Number of features to consider in one pass, i.e. how large could the tree grow")
    parser.add_argument('--max_depth', type=int, default=2, help="Maximum number of splits, i.e. how bushy could the tree grow")
    args = parser.parse_args()
    
    # Prepare the dataset to match the expected format
    path = 'https://raw.githubusercontent.com/allaccountstaken/automl_v_hyperdrive/main/data/camel_data_after2010Q3.csv'
    ds = load_data(path)
    X, y = clean_data(ds)
    #Consider for internal datasets:
    #from azureml.data.dataset_factory import TabularDatasetFactory
    #ds = TabularDatasetFactory.from_delimited_files(path)

    # Perorm train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), train_size=0.7, random_state=123)

    # Additionally consider scalling (this may not be important for tree-based models)
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    # Prepare Azure run context to log tunning progress
    run = Run.get_context()
    run.log("Learning rate:", np.float(args.learning_rate))
    run.log("Number of estimators:", np.int(args.n_estimators))
    run.log("Number of features:", np.int(args.max_features))
    run.log("Max tree depth:", np.int(args.max_depth))
    
    # Instanciated and fit GBM classifier using sklearn library
    model = GradientBoostingClassifier(learning_rate=args.learning_rate,
                                       n_estimators=args.n_estimators,
                                       max_features=args.max_features,
                                       max_depth=args.max_depth,
                                       random_state=123)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Report performance metrics of the trained model using testing subset
    recall = recall_score(y_test, y_pred, average='binary')
    run.log("Recall", np.round(np.float(recall), 5))
    #accuracy = model.score(X_test_scaled, y_test) 
    #run.log("Accuracy", np.float(accuracy))


    # The code below can be used to store the model for later consumption
    #os.makedirs("outputs", exist_ok=True)
    #joblib.dump(value=model, './outputs/model.joblib')

if __name__ == '__main__':
    main()



def test_oos_performance(model, oos_reports):
    pass

def plot_oos_performance(ins_df, oos_df):
    pass
