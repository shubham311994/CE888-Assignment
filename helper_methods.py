import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

RANDOM_STATE = 42


def preprocess_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset with required deletion and conversions.
    :param dataset: pandas dataframe object containing the data
    :return: pandas dataframe object containing the data after preprocessing
    """

    dataset.drop(columns=['car'], inplace=True)

    # changing the dataset type of the categorical columns to category from object dtype
    dataset_object = dataset.select_dtypes(include=['object']).copy()

    for col in dataset_object.columns:
        dataset[col] = dataset[col].astype('category')

    # dropping toCoupon_GEQ5min column from the data
    dataset.drop(columns=['toCoupon_GEQ5min'], inplace=True)

    category_mapping = {'less1': 0, '1~3': 1, 'never': 2, '4~8': 3, 'gt8': 4}

    dataset['Bar'] = dataset['Bar'].map(category_mapping)
    dataset['CoffeeHouse'] = dataset['CoffeeHouse'].map(category_mapping)
    dataset['CarryAway'] = dataset['CarryAway'].map(category_mapping)
    dataset['RestaurantLessThan20'] = dataset['RestaurantLessThan20'].map(category_mapping)
    dataset['Restaurant20To50'] = dataset['Restaurant20To50'].map(category_mapping)

    iterative_imp = IterativeImputer(estimator=RandomForestClassifier(),
                                     initial_strategy='most_frequent',
                                     max_iter=10, random_state=42)
    categorical_data = \
        dataset[['Bar', 'CoffeeHouse', 'Restaurant20To50', 'CarryAway', 'RestaurantLessThan20']]

    categorical_data = iterative_imp.fit_transform(categorical_data)
    dataset[['Bar', 'CoffeeHouse', 'Restaurant20To50', 'CarryAway', 'RestaurantLessThan20']] = \
        categorical_data

    high_cardinality = []
    for each in dataset_object.columns:
        if len(dataset[each].unique()) >= 3:
            high_cardinality.append(each)

    low_cardinality_columns = list(set(dataset_object.columns) - set(high_cardinality))

    label_enc = LabelEncoder()
    for each in low_cardinality_columns:
        dataset[each] = label_enc.fit_transform(dataset[each])

    dataset = pd.get_dummies(dataset, columns=high_cardinality, drop_first=True, prefix=high_cardinality)

    return dataset


def get_xgboost_params() -> dict:
    """
    Define and return the grid search params for Random Forest.
    :return: params dictionary
    """
    xgboost_params = dict(max_depth=range(3, 10, 1),
                          n_estimators=[100, 200, 300, 400, 500],
                          learning_rate=[0.1, 0.2, 0.3],
                          colsample_bytree=[0.5, 0.6, 0.7, 0.8, 0.9],
                          subsample=[0.6, 0.7, 0.8, 0.9]
                          )

    return xgboost_params


def get_lightgbm_params():
    """
    Define and return lightgbm parameters for grid search.

    :return: params dictionary
    """
    param_grid = {
        "n_estimators": [300, 500, 700, 1000],
        "learning_rate": [0.01, 0.3],
        "num_leaves": [20, 40, 60],
        "max_depth": [-1, 3, 6, 9],
    }

    return param_grid


def read_data(path: str) -> pd.DataFrame:
    """
    Read the data at the given path using pandas.
    :param path: path of the data file
    :return: pandas dataframe
    """
    data = pd.read_csv(path)

    return data


def train_test_set(dataset: pd.DataFrame):
    """
    Divide the dataset into train, test
    :param dataset:
    :return: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = \
        train_test_split(dataset.drop(columns=['Y']), dataset['Y'],
                         test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test


def get_random_forest_params() -> dict:
    """
    Define and return the grid search params for Random Forest.
    :return: params dictionary
    """
    parameters = {"n_estimators": [200, 300, 500, 700],
                               "criterion": ['gini', 'entropy'],
                               "max_depth": [None, 15, 20],
                               "min_samples_leaf": [1, 3, 9],
                               "min_samples_split": [2, 4, 6],
                               "max_features": ['auto', 'log2']}

    return parameters


def get_logistic_params():
    """
    Define and return the grid search params for Logistic Regression.
    :return: params dictionary
    """
    param_grid = [
        {'penalty': ['l2', 'none'],
         'C': [0.075,  0.5, 1, 10],
         'solver': ['newton-cg', 'liblinear'],
         'max_iter': [100, 1000, 2500, 5000]
         }
    ]

    return param_grid


def get_best_params(estimator_object, data: pd.DataFrame, parameters: dict):
    """
    Perform the Grid Search on the dataset with given estimators and parameters.
    :param estimator_object: model object for grid search
    :param data: dataset to fit on the grid search
    :param parameters: param grid for each model
    :return: grid_search_CV: object
    """
    grid_search_CV = GridSearchCV(estimator=estimator_object,
                                  param_grid=parameters, cv=StratifiedKFold(n_splits=10))
    grid_search_CV.fit(data.drop(columns=['Y']), data['Y'])
    print('##################################')
    print('Best Model Parameter with score')
    print(f'Best Parameters -> {grid_search_CV.best_params_}')
    print(f'Best Score -> {grid_search_CV.best_score_}')
    print('##################################')
    print('##################################')

    return grid_search_CV
