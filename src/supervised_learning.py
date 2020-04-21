import math
from sklearn.metrics import classification_report

import numpy as np, pandas as pd
import ast
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import KFold
import warnings
import pickle
from sklearn.utils import resample

warnings.filterwarnings('ignore')
import spacy

import xgboost as xgb
from nltk.stem.lancaster import LancasterStemmer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

en_nlp = spacy.load('en')
st = LancasterStemmer()

features_csv_path = "data/train-v2.0_detect_sent_fast_text.csv"
features_csv_with_root_matching_path = "data/train-v2.0_detect_sent_root_matching_fast_text.csv"

validation_csv_path_with_root_matching = "data/dev-v2.0_detect_sent_root_matching_fast_text.csv"
validation_csv_path = "data/dev-v2.0_detect_sent_fast_text.csv"
save_models = True


def load_data(path):
    data = pd.read_csv(path).reset_index(drop=True)

    print(data.shape)

    print(data.head(3))

    print(ast.literal_eval(data["sentences"][0]))

    data_total = data[data["sentences"].apply(lambda x: len(ast.literal_eval(x))) < 11].reset_index(drop=True)
    return data_total


def create_features(data):
    train = pd.DataFrame()

    for k in range(len(data["euclidean_dis"])):
        dis = ast.literal_eval(data["euclidean_dis"][k])
        for i in range(len(dis)):
            train.loc[k, "column_euc_" + "%s" % i] = dis[i]

    print("Finished")

    for k in range(len(data["cosine_sim"])):
        dis = ast.literal_eval(data["cosine_sim"][k].replace("nan", "1"))
        for i in range(len(dis)):
            train.loc[k, "column_cos_" + "%s" % i] = dis[i]

    train["target"] = data["target"]
    return train


def create_concatenated(training):
    training.apply(max, axis=0)

    subset1 = training.iloc[:, :10].fillna(60)
    subset2 = training.iloc[:, 10:].fillna(1)

    print(subset1.head(3))

    print(subset2.head(3))

    # train2 = pd.concat([subset1, subset2], axis=1, join_axes=[subset1.index])
    train2 = pd.concat([subset1, subset2], axis=1, join='outer')

    print(train2.head(3))

    train2.apply(max, axis=0)
    return train2


# TODO: make predictions on validation dataset
# make new model: train on entire training dataset
# as np arrays
# validation_accuracy = metrics.accuracy_score(validation_output, mul_lr.predict(validation_data))
# print validation accuracy
def validate_logistic_regression(X, Y, vX, vY):
    model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    model.fit(X, Y)
    validation_accuracy = metrics.accuracy_score(vY, model.predict(vX))
    target_names = ['class -1', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8',
                    'class 9',
                    'class 10']
    report = classification_report(vY, model.predict(vX), target_names=target_names)
    validation_f1 = metrics.f1_score(vY, model.predict(vX), average='micro')

    return validation_accuracy, report, validation_f1


def validate_random_forest(X, Y, vX, vY):
    rf = RandomForestClassifier(min_samples_leaf=8, n_estimators=60)
    rf.fit(X, Y)
    validation_accuracy = metrics.accuracy_score(vY, rf.predict(vX))
    target_names = ['class -1', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8',
                    'class 9',
                    'class 10']
    report = classification_report(vY, rf.predict(vX), target_names=target_names)
    validation_f1 = metrics.f1_score(vY, rf.predict(vX), average='micro')

    return validation_accuracy, report, validation_f1


def validate_XGB(X, Y, vX, vY):
    model = xgb.XGBClassifier()
    param_dist = {"max_depth": [3, 5, 10],
                  "min_child_weight": [1, 5, 10],
                  "learning_rate": [0.07, 0.1, 0.2],
                  }

    # run randomized search
    grid_search = GridSearchCV(model, param_grid=param_dist, cv=3,
                               verbose=5, n_jobs=-1)
    grid_search.fit(X, Y)

    print('Best Estimator', grid_search.best_estimator_)
    xg = xgb.XGBClassifier(max_depth=5)
    xg.fit(X, Y)
    validation_accuracy = metrics.accuracy_score(vY, xg.predict(vX))
    target_names = ['class -1', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8',
                    'class 9',
                    'class 10']
    report = classification_report(vY, xg.predict(vX), target_names=target_names)
    validation_f1 = metrics.f1_score(vY, xg.predict(vX), average='micro')

    return validation_accuracy, report, validation_f1


def log_reg_fit(training_standardised, validation_standardised):
    ### Fitting Multinomial Logistic Regression

    ### Standardize
    skf = KFold(n_splits=10)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(training_standardised.iloc[:, :-1])
    Y = training_standardised.iloc[:, -1]
    acc_array = np.zeros((10, 100))
    f1_micro_array = np.zeros((10, 100))
    ind = 0
    for train_index, test_index in skf.split(X, Y):
        mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
        mul_lr.fit(X[train_index], Y[train_index])
        for m in range(100):
            new_data, new_output = resample(X[test_index], Y[test_index], replace=True)
            accuracy = metrics.accuracy_score(new_output, mul_lr.predict(new_data))
            f1_micro = metrics.f1_score(new_output, mul_lr.predict(new_data), average='micro')
            acc_array[ind][m] = accuracy
            f1_micro_array[ind][m] = f1_micro
        ind += 1

    vX = scaler.fit_transform(validation_standardised.iloc[:, :-1])
    vY = validation_standardised.iloc[:, -1]
    validation_accuracy, report, validation_f1 = validate_logistic_regression(X, Y, vX, vY)

    final_acc_array = np.mean(acc_array, axis=0)
    final_f1_micro_array = np.mean(f1_micro_array, axis=0)
    sample_mean = np.average(final_acc_array)
    sample_f1_micro = np.average(final_f1_micro_array)
    sum_std_err = 0
    sum_std_err_micro = 0
    for each in final_acc_array:
        sum_std_err += (each - sample_mean) ** 2
    std_error = math.sqrt(sum_std_err / (len(final_acc_array) - 1))
    for each in final_f1_micro_array:
        sum_std_err_micro += (each - sample_f1_micro) ** 2
    std_error_micro = math.sqrt(sum_std_err_micro / (len(final_f1_micro_array) - 1))

    print('Multinimoal Logistic Regression Accuracy is', sample_mean)
    print('Multinimoal Logistic Regression Standard Error is', std_error)
    print('Multinimoal Logistic Regression Confidence Interval is: ', sample_mean - std_error, ' to: ',
          sample_mean + std_error)
    print("Validation accuracy:", validation_accuracy)
    print('Multinimoal Logistic Regression F-1 Micro is', sample_f1_micro)
    print('Multinimoal Logistic Regression F-1 Micro Standard Error is', std_error_micro)
    print('Multinimoal Logistic Regression  F-1 Micro Confidence Interval is: ', sample_f1_micro - std_error_micro,
          ' to: ',
          sample_f1_micro + std_error_micro)
    print("Validation F-1 Micro:", validation_f1)
    print('Classification report', report)


def get_columns_from_root(train):
    for i in range(train.shape[0]):
        if len(ast.literal_eval(train["root_match_idx"][i])) == 0:
            pass

        else:
            for item in ast.literal_eval(train["root_match_idx"][i]):
                train.loc[i, "column_root_" + "%s" % item] = 1
    return train


def process_root_data(predicted1, train2):
    predicted_new = get_columns_from_root(predicted1)
    subset3 = predicted_new[
        ["column_root_0", "column_root_1", "column_root_2", "column_root_3", "column_root_4", "column_root_5", \
         "column_root_6", "column_root_7", "column_root_8", "column_root_9"]]
    subset3.fillna(0, inplace=True)
    train3 = pd.concat([subset3, train2], axis=1, join='outer')
    train3 = train3[
        ["column_root_0", "column_root_1", "column_root_2", "column_root_3", "column_root_4", "column_root_5", \
         "column_root_6", "column_root_7", "column_root_8", "column_root_9", "column_cos_0", "column_cos_1", \
         "column_cos_2", "column_cos_3", "column_cos_4", "column_cos_5", \
         "column_cos_6", "column_cos_7", "column_cos_8", "column_cos_9", "target"]]
    dataset = train3.iloc[:, :-1].to_numpy()
    output = train3.iloc[:, -1].to_numpy()

    return dataset, output


## For Model with root matching
def log_reg_root(X, Y, vX, vY):
    skf = KFold(n_splits=10)
    acc_array_log = np.zeros((10, 100))
    acc_array_random = np.zeros((10, 100))
    acc_array_xg = np.zeros((10, 100))
    f1_micro_array_log = np.zeros((10, 100))
    f1_micro_array_random = np.zeros((10, 100))
    f1_micro_array_xg = np.zeros((10, 100))
    ind = 0
    for train_index, test_index in skf.split(X, Y):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
        mul_lr.fit(X_train, Y_train)

        rf = RandomForestClassifier(min_samples_leaf=8, n_estimators=60)
        rf.fit(X_train, Y_train)

        model = xgb.XGBClassifier()
        param_dist = {"max_depth": [3, 5, 10],
                      "min_child_weight": [1, 5, 10],
                      "learning_rate": [0.07, 0.1, 0.2],
                      }

        # run randomized search
        grid_search = GridSearchCV(model, param_grid=param_dist, cv=3,
                                   verbose=5, n_jobs=-1)
        grid_search.fit(X_train, Y_train)

        print('Best Estimator', grid_search.best_estimator_)

        xg = xgb.XGBClassifier(max_depth=5)
        xg.fit(X_train, Y_train)

        for m in range(100):
            new_data, new_output = resample(X_test, Y_test, replace=True)
            accuracy_log = metrics.accuracy_score(new_output, mul_lr.predict(new_data))
            accuracy_random = metrics.accuracy_score(new_output, rf.predict(new_data))
            accuracy_xg = metrics.accuracy_score(new_output, xg.predict(new_data))
            f1_micro_log = metrics.f1_score(new_output, mul_lr.predict(new_data), average='micro')
            f1_micro_random = metrics.f1_score(new_output, rf.predict(new_data), average='micro')
            f1_micro_xg = metrics.f1_score(new_output, xg.predict(new_data), average='micro')
            acc_array_log[ind][m] = accuracy_log
            acc_array_random[ind][m] = accuracy_random
            acc_array_xg[ind][m] = accuracy_xg
            f1_micro_array_log[ind][m] = f1_micro_log
            f1_micro_array_random[ind][m] = f1_micro_random
            f1_micro_array_xg[ind][m] = f1_micro_xg

        ind += 1

    lr_validation_accuracy, report_lr, f1_lr = validate_logistic_regression(X, Y, vX, vY)
    rf_validation_accuracy, report_rf, f1_rf = validate_random_forest(X, Y, vX, vY)
    xgb_validation_accuracy, report_xgb, f1_xgb = validate_XGB(X, Y, vX, vY)

    final_acc_array_log = np.mean(acc_array_log, axis=0)
    final_acc_array_random = np.mean(acc_array_random, axis=0)
    final_acc_array_xg = np.mean(acc_array_xg, axis=0)
    final_f1_micro_array_log = np.mean(f1_micro_array_log, axis=0)
    final_f1_micro_array_random = np.mean(f1_micro_array_random, axis=0)
    final_f1_micro_array_xg = np.mean(f1_micro_array_xg, axis=0)
    sample_mean_log = np.average(final_acc_array_log)
    sample_mean_random = np.average(final_acc_array_random)
    sample_mean_xg = np.average(final_acc_array_xg)
    sample_f1_log = np.average(final_f1_micro_array_log)
    sample_f1_random = np.average(final_f1_micro_array_random)
    sample_f1_xg = np.average(final_f1_micro_array_xg)
    sum_std_err_log = 0
    sum_std_err_random = 0
    sum_std_err_xg = 0
    sum_std_err_log_f1 = 0
    sum_std_err_random_f1 = 0
    sum_std_err_xg_f1 = 0
    for each in final_acc_array_log:
        sum_std_err_log += (each - sample_mean_log) ** 2
    std_error_log = math.sqrt(sum_std_err_log / (len(final_acc_array_log) - 1))
    for each in final_acc_array_random:
        sum_std_err_random += (each - sample_mean_random) ** 2
    std_error_random = math.sqrt(sum_std_err_random / (len(final_acc_array_random) - 1))
    for each in final_acc_array_xg:
        sum_std_err_xg += (each - sample_mean_xg) ** 2
    std_error_xg = math.sqrt(sum_std_err_xg / (len(final_acc_array_xg) - 1))

    for each in final_f1_micro_array_log:
        sum_std_err_log_f1 += (each - sample_f1_log) ** 2
    std_error_log_f1 = math.sqrt(sum_std_err_log_f1 / (len(final_f1_micro_array_log) - 1))
    for each in final_f1_micro_array_random:
        sum_std_err_random_f1 += (each - sample_f1_random) ** 2
    std_error_random_f1 = math.sqrt(sum_std_err_random_f1 / (len(final_f1_micro_array_random) - 1))
    for each in final_f1_micro_array_xg:
        sum_std_err_xg_f1 += (each - sample_f1_xg) ** 2
    std_error_xg_f1 = math.sqrt(sum_std_err_xg_f1 / (len(final_f1_micro_array_xg) - 1))

    print('Multinimoal Logistic Regression Accuracy with root matching is: ', sample_mean_log)
    print('Multinimoal Logistic Regression Standard Error with root matching is: ', std_error_log)
    print('Multinimoal Logistic Regression Confidence Interval with root matching is: ',
          sample_mean_log - std_error_log, ' to: ',
          sample_mean_log + std_error_log)
    print("Multinomial Logistic regression validation accuracy:", lr_validation_accuracy)

    print('Multinimoal Logistic Regression F1 with root matching is: ', sample_f1_log)
    print('Multinimoal Logistic Regression F1 Standard Error with root matching is: ', std_error_log_f1)
    print('Multinimoal Logistic Regression F1 Confidence Interval with root matching is: ',
          sample_f1_log - std_error_log_f1, ' to: ',
          sample_f1_log + std_error_log_f1)
    print("Multinomial Logistic regression validation F1:", f1_lr)
    print("Multinomial Logistic Regression Report", report_lr)

    print('Random Forest Accuracy with root matching is: ', sample_mean_random)
    print('Random Forest Standard Error with root matching is: ', std_error_random)
    print('Random Forest Confidence Interval with root matching is: ',
          sample_mean_random - std_error_random, ' to: ',
          sample_mean_random + std_error_random)
    print("Random forest validation accuracy:", rf_validation_accuracy)

    print('Random Forest F1 with root matching is: ', sample_f1_random)
    print('Random Forest F1 Standard Error with root matching is: ', std_error_random_f1)
    print('Random Forest F1 Confidence Interval with root matching is: ',
          sample_f1_random - std_error_random_f1, ' to: ',
          sample_f1_random + std_error_random_f1)
    print("Random forest validation F1:", f1_rf)
    print("Random Forest Report", report_rf)

    print('XG Boost Accuracy with root matching is: ', sample_mean_xg)
    print('XG Boost Standard Error with root matching is: ', std_error_xg)
    print('XG Boost Confidence Interval with root matching is: ',
          sample_mean_xg - std_error_xg, ' to: ',
          sample_mean_xg + std_error_xg)
    print("XGB validation accuracy:", xgb_validation_accuracy)

    print('XG Boost F1 with root matching is: ', sample_f1_xg)
    print('XG Boost F1 Standard Error with root matching is: ', std_error_xg_f1)
    print('XG Boost F1 Confidence Interval with root matching is: ',
          sample_f1_xg - std_error_xg_f1, ' to: ',
          sample_f1_xg + std_error_xg_f1)
    print("XGB validation F1:", f1_xgb)
    print("XGB Report", report_xgb)


data_usage = load_data(features_csv_with_root_matching_path)
validation_usage = load_data(validation_csv_path_with_root_matching)

train_set = create_features(data_usage)
validation_set = create_features(validation_usage)

# del data

print(train_set.head(3))

# train.fillna(10000, inplace=True)

print(train_set.head(3).transpose())

training_standardised = create_concatenated(train_set)
validation_standardised = create_concatenated(validation_set)

# log_reg_fit(training_standardised, validation_standardised)

## To run Model with root matching
root_dataset, root_output = process_root_data(data_usage, training_standardised)
validation_root_dataset, validation_root_output = process_root_data(validation_usage, validation_standardised)

log_reg_root(root_dataset, root_output, validation_root_dataset, validation_root_output)
