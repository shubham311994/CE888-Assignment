from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metric

from helper_methods import (preprocess_dataset, train_test_set,
                            get_random_forest_params, get_lightgbm_params, get_logistic_params,
                            get_best_params, RANDOM_STATE, read_data)

import lightgbm as lgb


def main(path: str):
    """
    Run all the commands and methods for the data preprocessing to modelling.
    :param path: path where the csv file is stored.
    :return:
    """
    data = preprocess_dataset(read_data(path))
    x_train, x_test, y_train, y_test = train_test_set(dataset=data)
    model_list = [RandomForestClassifier(), lgb.LGBMClassifier(device="gpu", random_state=RANDOM_STATE),
                  LogisticRegression()]
    parameters = [get_random_forest_params(), get_lightgbm_params(), get_logistic_params()]
    for each in range(len(model_list)):
        grid_search_cv_object = get_best_params(estimator_object=model_list[each], data=data,
                                                parameters=parameters[each])
        print('Taking the best parameters from Grid Search and building the model.')
        print('##################################')
        best_param_model = grid_search_cv_object.best_estimator_
        best_param_model.fit(x_train, y_train)
        best_param_model_predictions = best_param_model.predict(x_test)
        print(metric.classification_report(y_test, best_param_model_predictions))
        print('##################################')
        print('##################################')


if __name__ == '__main__':
    main(str('in-vehicle-coupon-recommendation.csv'))
